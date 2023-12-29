import torch
import torch.nn.functional as F


def focal_ce(input, target, alpha=None, gamma=2, reduction="mean", label_smoothing=0.0):
    """
    input: [N, C], float32
    target: [N, ], int64
    """
    assert 0 <= label_smoothing < 1

    if input.ndim > 2:
        # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
        c = input.shape[1]
        input = input.permute(0, *range(2, input.ndim), 1).reshape(-1, c)
        # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
        target = target.view(-1)

    # compute weighted cross entropy term: -alpha * log(pt)
    # (alpha is already part of self.nll_loss)
    log_p = F.log_softmax(input, dim=-1)
    ce = F.nll_loss(log_p, target, weight=alpha, reduction="none")
    if label_smoothing != 0:
        confidence = 1.0 - label_smoothing
        smoothing = label_smoothing
        smooth_loss = -log_p.mean(dim=-1)
        ce = confidence * ce + smoothing * smooth_loss

    # get true class column from each row
    all_rows = torch.arange(len(input))
    log_pt = log_p[all_rows, target]

    # compute focal term: (1 - pt)^gamma
    pt = log_pt.exp()
    focal_term = (1 - pt) ** gamma

    # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
    loss = focal_term * ce



    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()


    return loss
