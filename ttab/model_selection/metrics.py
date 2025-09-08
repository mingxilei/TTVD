# -*- coding: utf-8 -*-
import torch
from ttab.utils.stat_tracker import RuntimeTracker

task2metrics = {"classification": ["cross_entropy", "accuracy_top1", "ece_loss"]}
auxiliary_metrics_dict = {
    "preadapted_cross_entropy": "cross_entropy",
    "preadapted_accuracy_top1": "accuracy_top1",
    "preadapted_ece": "ece_loss",
}


class Metrics(object):
    def __init__(self, scenario) -> None:
        self._conf = scenario
        self._init_metrics()

    def _init_metrics(self) -> None:
        self._metrics = task2metrics[self._conf.task]
        self.tracker = RuntimeTracker(metrics_to_track=self._metrics)
        self._primary_metrics = self._metrics[0]

    def init_auxiliary_metric(self, metric_name: str):
        self._metrics.append(metric_name)
        self.tracker.add_stat(metric_name)

    @torch.no_grad()
    def eval(self, y: torch.Tensor, y_hat: torch.Tensor) -> None:
        results = dict()
        for metric_name in self._metrics:
            if not metric_name in auxiliary_metrics_dict.keys():
                results[metric_name] = eval(metric_name)(y, y_hat)
            else:
                continue
        self.tracker.update_metrics(results, n_samples=y.size(0))
        return results

    @torch.no_grad()
    def eval_auxiliary_metric(
        self, y: torch.Tensor, y_hat: torch.Tensor, metric_name: str
    ):
        assert (
            metric_name in self._metrics
        ), "The target metric must be in the list of metrics."
        results = dict()
        results[metric_name] = eval(auxiliary_metrics_dict[metric_name])(y, y_hat)
        self.tracker.update_metrics(results, n_samples=y.size(0))
        return results


"""list some common metrics."""


def _accuracy(target, output, topk):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / batch_size).item()


def accuracy_top1(target, output, topk=1):
    """Computes the precision@k for the specified values of k"""
    if target.shape[0] != output.shape[0]: return accuracy_top1_agg(target, output, topk)
    return _accuracy(target, output, topk)


def accuracy_top5(target, output, topk=5):
    """Computes the precision@k for the specified values of k"""
    return _accuracy(target, output, topk)


cross_entropy_loss = torch.nn.CrossEntropyLoss()


def cross_entropy(target, output):
    """Cross entropy loss"""
    if target.shape[0] != output.shape[0]: return cross_entropy_agg(target, output)
    return cross_entropy_loss(output, target).item()


def accuracy_top1_agg(target, output, topk=1):
    """Computes the precision@k for the specified values of k"""
    agg_outputs = 0

    for i in range(4):
        agg_outputs = agg_outputs + output[i*len(target):(i+1)*len(target), i::4] / 4
    return _accuracy(target, agg_outputs, topk)

def cross_entropy_agg(target, output):
    agg_outputs = 0
    for i in range(4):
        agg_outputs = agg_outputs + output[i*len(target):(i+1)*len(target), i::4] / 4
    return cross_entropy_loss(agg_outputs, target).item()


def ece_loss(target, output):
    if target.shape[0] != output.shape[0]: return ece_loss_agg(target, output)
    output = torch.nn.functional.softmax(output, dim=1)
    return ECELoss().loss(output=output.detach().cpu().numpy(), labels=target.detach().cpu().numpy())

def ece_loss_agg(target, output):
    agg_outputs = 0
    for i in range(4):
        agg_outputs = agg_outputs + output[i*len(target):(i+1)*len(target), i::4] / 4
    agg_outputs = torch.nn.functional.softmax(agg_outputs, dim=1)
    return ECELoss().loss(p=output.detach().cpu().numpy(), labels=target.detach().cpu().numpy())
