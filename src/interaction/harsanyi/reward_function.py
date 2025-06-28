import numpy as np
import torch
from typing import List


def _get_log_odds(logits, dim):
    """
    Let p = softmax(values, dim=1)[:, dim], return log (p / (1 - p))
    :param logits: tensor of shape (N, class_number)
    :param dim: int, the dimension to calculate log-odds
    :return values: tensor of shape (N, ), the log-odds.
    """
    class_number = logits.shape[1]
    values = logits[:, dim] - torch.logsumexp(logits[:, np.arange(class_number) != dim], dim=1)
    return values


def get_reward(outputs, selected_dim, **kwargs):
    """
    Given logits, calculate reward score for interaction computation
    :param outputs: tensor of shape (N, class_number)
    :param selected_dim: str, the dimension to calculate reward score
        - "0": the first dimension
        - "gt": the specified dimension (not necessarily the ground-truth dimension)
        - "gt-log-odds": the log-odds of the specified dimension
        - "gt-log-odds-sample=1000": the log-odds of the specified dimension, where the logits are sampled from 1000 classes
        - "gt-log-odds-t=temperature": the log-odds of the specified dimension, where the logits are divided by temperature
        - "max-log-odds": not implemented
        - "gt-logistic-odds": the log-odds of the specified dimension for binary classification
        - "gt-prob-logistic-odds": the log-odds of the specified dimension for binary classification and when outputs are probabilities
        - None: the outputs
    :param kwargs: dict, additional arguments
    :return values: tensor of shape (N, ), the reward score
    """
    if selected_dim == "0":
        values = outputs[:, 0]
    elif selected_dim == "gt":
        assert "gt" in kwargs.keys()
        gt = kwargs["gt"]
        values = outputs[:, gt]  # select the specified dimension (not necessarily the ground-truth dimension)
    elif selected_dim == "gt-log-odds":
        assert "gt" in kwargs.keys()
        gt = kwargs["gt"]
        values = _get_log_odds(outputs, gt)
    elif selected_dim.startswith("gt-log-odds-sample="): # todo: 后面需要再跟zjp确认一下这个有没有写错
        assert "gt" in kwargs
        assert "sample" in kwargs, "The sampled dimensions must include gt" # sample must include gt??
        gt = kwargs["gt"]
        sample = kwargs["sample"]
        num_sample_dims = int(selected_dim.split("=")[-1])
        assert isinstance(sample, List)
        assert len(sample) == num_sample_dims
        # ensure that gt is in sample
        assert gt in sample
        gt_index_in_sample = sample.index(gt)

        sampled_logits = outputs[:, sample]
        values = _get_log_odds(sampled_logits, gt_index_in_sample)

    elif selected_dim.startswith("gt-log-odds-t="):
        assert "gt" in kwargs.keys()
        gt = kwargs["gt"]
        temperature = float(selected_dim.split("=")[-1])
        values = _get_log_odds(outputs / temperature, gt)
    elif selected_dim == "max-log-odds":
        raise NotImplementedError  # todo
        # eps = 1e-7
        # values = torch.softmax(values, dim=1)
        # values = values[:, torch.argmax(values[-1])]
        # values = torch.log(values / (1 - values + eps) + eps)
    elif selected_dim == "gt-logistic-odds":
        # this is for the case of binary classification (using logistic regression)
        # In binary classification, the output is the probability p(y=1|x)=sigmoid(z), where z is the logit.
        # The log-odds is log(p/(1-p)) = z if y=1, and -z if y=0.
        assert "gt" in kwargs
        gt = kwargs["gt"]
        assert gt == 0 or gt == 1
        assert len(outputs.shape) == 2 and outputs.shape[1] == 1
        if gt == 1:
            values = outputs[:, 0]
        else:
            values = -outputs[:, 0]

    elif selected_dim == "gt-prob-logistic-odds":
        # this is for the case of binary classification and when outputs are probabilities
        assert "gt" in kwargs
        gt = kwargs["gt"]
        assert gt == 0 or gt == 1
        assert len(outputs.shape) == 2 and outputs.shape[1] == 1
        probs = outputs[:, 0]
        if gt == 0:
            probs = 1 - probs
        else:
            probs = probs
        # eps = 1e-7
        values = torch.special.logit(probs) # todo: need to specify an eps?
    elif selected_dim == None:
        values = outputs
    else:
        raise Exception(f"Unknown [selected_dim] {selected_dim}.")

    return values


if __name__ == '__main__':
    # test the _log_odds function is self-compatible when dim is None or not
    # set seed
    torch.manual_seed(0)
    logits = torch.randn(4,10)
    values = _get_log_odds(logits, dim=1)
    print(values)