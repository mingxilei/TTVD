# -*- coding: utf-8 -*-
import copy
import functools
from typing import List

import torch
import torch.nn as nn
import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer
from math import copysign

import torch.nn.functional as F
import math

class TTVD(BaseAdaptation):

    def __init__(self, meta_conf, model: nn.Module, n=4):
        super(TTVD, self).__init__(meta_conf, model)
        self.vd_centers = torch.load(meta_conf.ttvd_centers, map_location=torch.device('cpu')).to(meta_conf.device)
        self.n = n
        self.alpha = meta_conf.ttvd_alpha

    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""
        model.train()
        # disable grad, to (re-)enable only what specified adaptation method updates
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # bn module always uses batch statistics, in both training and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
        return model.to(self._meta_conf.device)

    def _initialize_trainable_parameters(self):
        """
        Collect the affine scale + shift parameters from norm layers.

        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        self._adapt_module_names = []
        adapt_params = []
        adapt_param_names = []

        for name_module, module in self._model.named_modules():
            if isinstance(
                module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
            ):  # only bn is used in the paper.
                self._adapt_module_names.append(name_module)
                for name_parameter, parameter in module.named_parameters():
                    if name_parameter in ["weight", "bias"]:
                        adapt_params.append(parameter)
                        adapt_param_names.append(f"{name_module}.{name_parameter}")

        assert (
            len(self._adapt_module_names) > 0
        ), "TENT needs some adaptable model parameters."
        return adapt_params, adapt_param_names

    def one_adapt_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        timer: Timer,
        random_seed: int = None,
    ):
        """adapt the model in one step."""
        with timer("forward"):
            with fork_rng_with_seed(random_seed):
                inputs_cls = batch._x
                inputs_cls, targets_ssh = adaptation_utils.rotate_batch(
                    batch._x, 'expand', self._meta_conf.device
                )

                # pd loss
                features = model.forward_features(inputs_cls)
                pd_centers = model.fc.weight / 2
                if model.fc.bias is not None:
                    pd_radius = torch.sum(pd_centers ** 2, dim=1) + model.fc.bias
                else:
                    pd_radius = torch.sum(pd_centers ** 2, dim=1)
                dists_pd = torch.cdist(features.unsqueeze(1), pd_centers.unsqueeze(0)).squeeze() ** 2 - pd_radius
                y_hat = self.joint_influence(dists_pd, len(batch), self.n, self.alpha)

                # vd loss
                dists = torch.cdist(features.unsqueeze(1), self.vd_centers.unsqueeze(0)).squeeze()
                y_hat_p = self.joint_influence(dists, len(batch), self.n, self.alpha)

                idx = self.multi_diagram_filtering(y_hat, y_hat_p)

            loss = (
                +adaptation_utils.softmax_entropy(-dists_pd[idx]).mean(0)
                + adaptation_utils.softmax_entropy(-dists[idx]).mean(0)
            )

            # apply fisher regularization when enabled
            if self.fishers is not None:
                ewc_loss = 0
                for name, param in model.named_parameters():
                    if name in self.fishers:
                        ewc_loss += (
                            self._meta_conf.fisher_alpha
                            * (
                                self.fishers[name][0]
                                * (param - self.fishers[name][1]) ** 2
                            ).sum()
                        )
                loss += ewc_loss

        with timer("backward"):
            loss.backward()
            grads = dict(
                (name, param.grad.clone().detach())
                for name, param in model.named_parameters()
                if param.grad is not None
            )
            optimizer.step()
            optimizer.zero_grad()
        return {
            "optimizer": copy.deepcopy(optimizer).state_dict(),
            "loss": loss.item(),
            "grads": grads,
            "yhat": y_hat,
        }

    def run_multiple_steps(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        model_selection_method: BaseSelection,
        nbsteps: int,
        timer: Timer,
        random_seed: int = None,
    ):
        for step in range(1, nbsteps + 1):
            adaptation_result = self.one_adapt_step(
                model,
                optimizer,
                batch,
                timer,
                random_seed=random_seed,
            )

            model_selection_method.save_state(
                {
                    "model": copy.deepcopy(model).state_dict(),
                    "step": step,
                    "lr": self._meta_conf.lr,
                    **adaptation_result,
                },
                current_batch=batch,
            )

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
        if episodic:
            log("\treset model to initial state during the test time.")
            self.reset()

        log(f"\tinitialize selection method={model_selection_method.name}.")
        model_selection_method.initialize()

        # evaluate the per batch pre-adapted performance. Different with no adaptation.
        if self._meta_conf.record_preadapted_perf:
            with timer("evaluate_preadapted_performance"):
                self._model.eval()
                with torch.no_grad():
                    # yhat = self._model(current_batch._x)
                    inputs_cls, _ = adaptation_utils.rotate_batch(
                    current_batch._x, 'expand', self._meta_conf.device
                )
                    yhat = self._model(inputs_cls)
                self._model.train()
                metrics.eval_auxiliary_metric(
                    current_batch._y, yhat, metric_name="preadapted_accuracy_top1"
                )

        # adaptation.
        with timer("test_time_adaptation"):
            nbsteps = self._get_adaptation_steps(index=len(previous_batches))
            log(f"\tadapt the model for {nbsteps} steps with lr={self._meta_conf.lr}.")
            self.run_multiple_steps(
                model=self._model,
                optimizer=self._optimizer,
                batch=current_batch,
                model_selection_method=model_selection_method,
                nbsteps=nbsteps,
                timer=timer,
                random_seed=self._meta_conf.seed,
            )

        # select the optimal checkpoint, and return the corresponding prediction.
        with timer("select_optimal_checkpoint"):
            optimal_state = model_selection_method.select_state()
            log(
                f"\tselect the optimal model ({optimal_state['step']}-th step and lr={optimal_state['lr']}) for the current mini-batch.",
            )

            self._model.load_state_dict(optimal_state["model"])
            model_selection_method.clean_up()

            if self._oracle_model_selection:
                # oracle model selection needs to save steps
                self.oracle_adaptation_steps.append(optimal_state["step"])
                # update optimizer.
                self._optimizer.load_state_dict(optimal_state["optimizer"])

        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, optimal_state["yhat"])
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    optimal_state["yhat"],
                    current_batch._y,
                    current_batch._g,
                    is_training=False,
                )

        # stochastic restore part of model parameters if enabled.
        if self._meta_conf.stochastic_restore_model:
            self.stochastic_restore()

    @property
    def name(self):
        return "ttvd"

    def joint_influence(self, distances, batch_size, n, alpha):
        """Computes CIVD joint influence"""

        y_hat = 0
        for i in range(4):
            y_hat = y_hat - distances[i*batch_size:(i+1)*batch_size, i::n].pow(alpha) * copysign(1, alpha)
        return y_hat

    def multi_diagram_filtering(self, y_hat1, y_hat2):
        preds1, preds2 = y_hat1.max(1)[1], y_hat2.max(1)[1]
        idx = preds1 == preds2
        idx = torch.cat([idx*(k+1) for k in range(4)])
        return idx
    
    def pd(self, model):
        pd_centers = model.fc.weight / 2
        if model.fc.bias is not None:
            pd_radius = torch.sum(pd_centers ** 2, dim=1) + model.fc.bias
        else:
            pd_radius = torch.sum(pd_centers ** 2, dim=1)
        return pd_centers, pd_radius
