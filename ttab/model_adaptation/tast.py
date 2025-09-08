# -*- coding: utf-8 -*-
import copy
import functools
from typing import List

import torch
import torch.nn as nn
import ttab.loads.define_dataset as define_dataset
import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer
import torch.nn.functional as F


class BatchEnsemble(nn.Module):
    def __init__(self, indim, outdim, ensemble_size, init_mode):
        super().__init__()
        self.ensemble_size = ensemble_size

        self.in_features = indim
        self.out_features = outdim

        # register parameters
        self.register_parameter(
            "weight", nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        )
        self.register_parameter("bias", nn.Parameter(torch.Tensor(self.out_features)))

        self.register_parameter(
            "alpha_be", nn.Parameter(torch.Tensor(self.ensemble_size, self.in_features))
        )
        self.register_parameter(
            "gamma_be",
            nn.Parameter(torch.Tensor(self.ensemble_size, self.out_features)),
        )

        use_ensemble_bias = True
        if use_ensemble_bias and self.bias is not None:
            delattr(self, "bias")
            self.register_parameter("bias", None)
            self.register_parameter(
                "ensemble_bias",
                nn.Parameter(torch.Tensor(self.ensemble_size, self.out_features)),
            )
        else:
            self.register_parameter("ensemble_bias", None)

        # initialize parameters
        self.init_mode = init_mode
        self.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D1 = x.size()
        r_x = x.unsqueeze(0).expand(self.ensemble_size, B, D1)  #
        r_x = r_x.view(self.ensemble_size, -1, D1)
        r_x = r_x * self.alpha_be.view(self.ensemble_size, 1, D1)
        r_x = r_x.view(-1, D1)

        w_r_x = nn.functional.linear(r_x, self.weight, self.bias)

        _, D2 = w_r_x.size()
        s_w_r_x = w_r_x.view(self.ensemble_size, -1, D2)
        s_w_r_x = s_w_r_x * self.gamma_be.view(self.ensemble_size, 1, D2)
        if self.ensemble_bias is not None:
            s_w_r_x = s_w_r_x + self.ensemble_bias.view(self.ensemble_size, 1, D2)
        s_w_r_x = s_w_r_x.view(-1, D2)

        return s_w_r_x

    def reset(self):
        init_details = [0, 1]
        initialize_tensor(self.weight, self.init_mode, init_details)
        initialize_tensor(self.alpha_be, self.init_mode, init_details)
        initialize_tensor(self.gamma_be, self.init_mode, init_details)
        if self.ensemble_bias is not None:
            initialize_tensor(self.ensemble_bias, "zeros")
        if self.bias is not None:
            initialize_tensor(self.bias, "zeros")


def initialize_tensor(
    tensor: torch.Tensor,
    initializer: str,
    init_values: List[float] = [],
) -> None:

    if initializer == "zeros":
        nn.init.zeros_(tensor)

    elif initializer == "ones":
        nn.init.ones_(tensor)

    elif initializer == "uniform":
        nn.init.uniform_(tensor, init_values[0], init_values[1])

    elif initializer == "normal":
        nn.init.normal_(tensor, init_values[0], init_values[1])

    elif initializer == "random_sign":
        with torch.no_grad():
            tensor.data.copy_(
                2.0
                * init_values[1]
                * torch.bernoulli(torch.zeros_like(tensor) + init_values[0])
                - init_values[1]
            )
    elif initializer == "xavier_normal":
        torch.nn.init.xavier_normal_(tensor)

    elif initializer == "kaiming_normal":
        torch.nn.init.kaiming_normal_(tensor)
    else:
        raise NotImplementedError(f"Unknown initializer: {initializer}")


class TAST(BaseAdaptation):
    """
    T3A: Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization,
    https://openreview.net/forum?id=e_yvNqkJKAW&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2021%2FConference%2FAuthors%23your-submissions),
    https://github.com/matsuolab/T3A

    T3A adjusts a trained linear classifier with the following procedure:
    (1) compute a pseudo-prototype representation for each class.
    (2) classify each sample based on its distance to the pseudo-prototypes.
    """

    def __init__(self, meta_conf, model: nn.Module):
        super(TAST, self).__init__(meta_conf, model)

    def _prior_safety_check(self):

        assert self._meta_conf.top_M > 0, "top_M must be correctly specified"
        assert (
            self._meta_conf.debug is not None
        ), "The state of debug should be specified"
        assert self._meta_conf.n_train_steps > 0, "Adaptation steps requires >= 1."

    def _initialize_model(self, model):
        """Configure model for adaptation."""
        # In T3A, no update on model params.
        model.requires_grad_(False)
        model.eval()

        return model.to(self._meta_conf.device)

    def _initialize_trainable_parameters(self):
        """
        During adaptation, T3A doesn't need to update params in network.
        """
        self._adapt_module_names = []
        self._classifier_layers = []

        # TODO: make this more general
        # Problem description: the naming and structure of classifier layers may change.
        # Be careful: the following list may not cover all cases when using model outside of this library.
        freezed_module_name = ["fc", "classifier", "head"]

        for named_module, module in self._model.named_children():
            if named_module in freezed_module_name:
                assert isinstance(module, nn.Linear)
                self._classifier_layers.append(module)
            elif named_module == "classifiers":
                self._classifier_layers.append(module[0])
            else:
                self._adapt_module_names.append(named_module)

        self.warmup_supports = self._classifier_layers[-1].weight.data.to(
            self._meta_conf.device
        )
        self._num_classes = self._classifier_layers[-1].weight.data.size(0)

    def _initialize_optimizer(self, params) -> torch.optim.Optimizer:
        """In T3A, no optimizer is used."""
        pass

    def _post_safety_check(self):
        is_training = self._model.training
        assert not is_training, "T3A does not need train mode: call model.eval()."

        param_grads = [p.requires_grad for p in (self._model.parameters())]
        has_any_params = any(param_grads)
        assert not has_any_params, "adaptation does not need trainable params."
        is_training = self.mlps.training
        assert is_training

    def initialize(self, seed: int):
        """Initialize the algorithm."""
        self._model = self._initialize_model(model=copy.deepcopy(self._base_model))
        self._initialize_trainable_parameters()
        self._auxiliary_data_cls = define_dataset.ConstructAuxiliaryDataset(
            config=self._meta_conf
        )

        warmup_prob = self.warmup_supports
        for module in self._classifier_layers:
            warmup_prob = module(warmup_prob)

        self.warmup_ent = adaptation_utils.softmax_entropy(warmup_prob)
        self.warmup_labels = nn.functional.one_hot(
            warmup_prob.argmax(1), num_classes=self._num_classes
        ).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent

        self.top_M = self._meta_conf.top_M
        self.softmax = nn.Softmax(-1)

        self.filter_K = self._meta_conf.top_M
        self.steps = self._meta_conf.gamma
        self.num_ensemble = self._meta_conf.num_ensemble
        self.lr = self._meta_conf.lr
        self.tau = self._meta_conf.tau
        self.init_mode = self._meta_conf.init_mode
        self.num_classes = self._num_classes
        self.k = self._meta_conf.k

        self.mlps = BatchEnsemble(
            self._model.fc.in_features,
            self._model.fc.in_features // 4,
            self.num_ensemble,
            self.init_mode,
        ).to(self._meta_conf.device)
        self.optimizer = torch.optim.SGD(self.mlps.parameters(), lr=self._meta_conf.lr, momentum=0.9)
        # only use one-step adaptation.
        if self._meta_conf.n_train_steps > 1:
            self._meta_conf.n_train_steps = 1

    def one_adapt_step(
        self,
        model: torch.nn.Module,
        batch: Batch,
        timer: Timer,
        random_seed: int = None,
    ):
        """adapt the model in one step."""
        with timer("forward"):
            with fork_rng_with_seed(random_seed):
                feas = model.forward_head(
                    model.forward_features(batch._x), pre_logits=True
                )
                # feas = model.base_model(batch._x)
                feas = feas.view(feas.size(0), -1)

            y_hat = self._model(batch._x)
            label_hat = nn.functional.one_hot(
                y_hat.argmax(1), num_classes=self._num_classes
            ).float()
            ent = adaptation_utils.softmax_entropy(y_hat)

            # prediction.
            assert (
                self.supports.device == feas.device
            ), "Supports and features should be on the same device."
            self.supports = torch.cat([self.supports, feas])
            self.labels = torch.cat([self.labels, label_hat])
            self.ent = torch.cat([self.ent, ent])

            supports, labels = self.select_supports()
            # supports = nn.functional.normalize(supports, dim=1)
            # weights = supports.T @ (labels)
            # adapted_y = feas @ nn.functional.normalize(weights, dim=0)
            # loss = adaptation_utils.softmax_entropy(adapted_y).mean(0)
            for _ in range(self.steps):
                adapted_y = self.forward_and_adapt(feas, supports, labels)
            loss = adaptation_utils.softmax_entropy(adapted_y).mean(0)
        return {"loss": loss.item(), "yhat": adapted_y, "grads": None}

    def adapt_and_eval(
        self,
        episodic: bool,
        metrics: Metrics,
        model_selection_method: BaseSelection,
        current_batch: Batch,
        previous_batches: List[Batch],
        logger: Logger,
        timer: Timer,
    ):
        """The key entry of test-time adaptation."""
        # some simple initialization.
        log = functools.partial(logger.log, display=self._meta_conf.debug)

        # model_selection method is defined but not used in T3A.
        log(f"\tinitialize selection method={model_selection_method.name}.")
        model_selection_method.initialize()

        # adaptation.
        with timer("test_adaptation"):
            log(f"\tadapt the model for {self._meta_conf.n_train_steps} steps.")
            for _ in range(self._meta_conf.n_train_steps):
                adaptation_result = self.one_adapt_step(
                    self._model, current_batch, timer, random_seed=self._meta_conf.seed
                )

        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, adaptation_result["yhat"])
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    adaptation_result["yhat"],
                    current_batch._y,
                    current_batch._g,
                    is_training=False,
                )

    def generate_representation(self, batch: Batch):
        """
        The classifier adaptation needs the feature representations as inputs.
        """

        assert (
            not self._model.training
        ), "The generation process needs model.eval() mode."

        inputs = batch._x
        # targets = batch._y

        feas = self._model.forward_features(inputs)
        feas = feas.view(inputs.size(0), -1)

        target_hat = self._model(inputs)
        ent = adaptation_utils.softmax_entropy(target_hat)
        label_hat = target_hat.argmax(1).float()

        return feas, target_hat, label_hat, ent

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        top_M = self.top_M
        # if top_M == -1:
        #     indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).to(self._meta_conf.device)
        for i in range(self._num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat == i][indices2][:top_M])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        return self.supports, self.labels

    @torch.enable_grad()
    def forward_and_adapt(self, z, supports, labels):
        # targets : pseudo labels, outputs: for prediction
        with torch.no_grad():
            targets, outputs = self.target_generation(z, supports, labels)

        self.optimizer.zero_grad()

        loss = None
        logits = self.compute_logits(z, supports, labels, self.mlps)

        for ens in range(self.num_ensemble):
            if loss is None:
                loss = F.kl_div(logits[ens].log_softmax(-1), targets[ens])
            else:
                loss += F.kl_div(logits[ens].log_softmax(-1), targets[ens])

        loss.backward()
        self.optimizer.step()

        return outputs  # outputs

    def compute_logits(self, z, supports, labels, mlp):
        """
        :param z: unlabeled test examples
        :param supports: support examples
        :param labels: labels of support examples
        :param mlp: multiple projection heads
        :return: classification logits of z
        """
        B, dim = z.size()
        N, dim_ = supports.size()

        mlp_z = mlp(z)
        mlp_supports = mlp(supports)

        assert dim == dim_

        logits = torch.zeros(self.num_ensemble, B, self.num_classes).to(z.device)
        for ens in range(self.num_ensemble):
            temp_centroids = (
                labels / (labels.sum(dim=0, keepdim=True) + 1e-12)
            ).T @ mlp_supports[ens * N : (ens + 1) * N]

            # normalize
            temp_z = torch.nn.functional.normalize(
                mlp_z[ens * B : (ens + 1) * B], dim=1
            )
            temp_centroids = torch.nn.functional.normalize(temp_centroids, dim=1)

            logits[ens] = self.tau * temp_z @ temp_centroids.T  # [B,C]

        return logits

    def cosine_distance_einsum(self, X, Y):
        # X, Y [n, dim], [m, dim] -> [n,m]
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)
        XX = torch.einsum("nd, nd->n", X, X)[:, None]  # [n, 1]
        YY = torch.einsum("md, md->m", Y, Y)  # [m]
        XY = 2 * torch.matmul(X, Y.T)  # [n,m]
        return XX + YY - XY

    def target_generation(self, z, supports, labels):
        # retrieve k nearest neighbors. from "https://github.com/csyanbin/TPN/blob/master/train.py"
        dist = self.cosine_distance_einsum(z, supports)
        W = torch.exp(-dist)  # [B, N]

        temp_k = (
            self.filter_K
            if self.filter_K != -1
            else supports.size(0) // self.num_classes
        )
        k = min(self.k, temp_k)

        values, indices = torch.topk(W, k, sorted=False)  # [B, k]
        topk_indices = torch.zeros_like(W).scatter_(
            1, indices, 1
        )  # [B, N] 1 for topk, 0 for else
        temp_labels = self.compute_logits(
            supports, supports, labels, self.mlps
        )  # [ens, N, C]
        temp_labels_targets = F.one_hot(
            temp_labels.argmax(-1), num_classes=self.num_classes
        ).float()  # [ens, N, C]
        temp_labels_outputs = torch.softmax(temp_labels, -1)  # [ens, N, C]

        topk_indices = topk_indices.unsqueeze(0).repeat(
            self.num_ensemble, 1, 1
        )  # [ens, B, N]

        # targets for pseudo labels. we use one-hot class distribution
        targets = torch.bmm(topk_indices, temp_labels_targets)  # [ens, B, C]
        targets = targets / (targets.sum(2, keepdim=True) + 1e-12)  # [ens, B, C]
        # targets = targets.mean(0)  # [B,C]

        # outputs for prediction
        outputs = torch.bmm(topk_indices, temp_labels_outputs)  # [ens, B, C]
        outputs = outputs / (outputs.sum(2, keepdim=True) + 1e-12)  # [ens, B, C]
        outputs = outputs.mean(0)  # [B,C]

        return targets, outputs

    def reset_warmup(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

    @property
    def name(self):
        return "tast"
