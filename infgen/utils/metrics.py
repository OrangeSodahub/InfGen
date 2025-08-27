import torch
import os
import itertools
import multiprocessing as mp
import torch.nn.functional as F
from pathlib import Path
from torch.nn import CrossEntropyLoss
from torch_scatter import gather_csr
from torch_scatter import segment_csr
from torchmetrics import Metric
from typing import Optional, Tuple, Dict, List


__all__ = ['minADE', 'minFDE', 'TokenCls', 'StateAccuracy', 'GridOverlapRate']


class CustomCrossEntropyLoss(CrossEntropyLoss):

    def __init__(self, label_smoothing=0.0, reduction='mean'):
        super(CustomCrossEntropyLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, input, target):
        num_classes = input.size(1)

        log_probs = F.log_softmax(input, dim=1)
        
        with torch.no_grad():
            smooth_target = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
            smooth_target = smooth_target * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        
        loss = -torch.sum(log_probs * smooth_target, dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def topk(
        max_guesses: int,
        pred: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
        joint: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    max_guesses = min(max_guesses, pred.size(1))
    if max_guesses == pred.size(1):
        if prob is not None:
            prob = prob / prob.sum(dim=-1, keepdim=True)
        else:
            prob = pred.new_ones((pred.size(0), max_guesses)) / max_guesses
        return pred, prob
    else:
        if prob is not None:
            if joint:
                if ptr is None:
                    inds_topk = torch.topk((prob / prob.sum(dim=-1, keepdim=True)).mean(dim=0, keepdim=True),
                                           k=max_guesses, dim=-1, largest=True, sorted=True)[1]
                    inds_topk = inds_topk.repeat(pred.size(0), 1)
                else:
                    inds_topk = torch.topk(segment_csr(src=prob / prob.sum(dim=-1, keepdim=True), indptr=ptr,
                                                       reduce='mean'),
                                           k=max_guesses, dim=-1, largest=True, sorted=True)[1]
                    inds_topk = gather_csr(src=inds_topk, indptr=ptr)
            else:
                inds_topk = torch.topk(prob, k=max_guesses, dim=-1, largest=True, sorted=True)[1]
            pred_topk = pred[torch.arange(pred.size(0)).unsqueeze(-1).expand(-1, max_guesses), inds_topk]
            prob_topk = prob[torch.arange(pred.size(0)).unsqueeze(-1).expand(-1, max_guesses), inds_topk]
            prob_topk = prob_topk / prob_topk.sum(dim=-1, keepdim=True)
        else:
            pred_topk = pred[:, :max_guesses]
            prob_topk = pred.new_ones((pred.size(0), max_guesses)) / max_guesses
        return pred_topk, prob_topk


def topkind(
        max_guesses: int,
        pred: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
        joint: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_guesses = min(max_guesses, pred.size(1))
    if max_guesses == pred.size(1):
        if prob is not None:
            prob = prob / prob.sum(dim=-1, keepdim=True)
        else:
            prob = pred.new_ones((pred.size(0), max_guesses)) / max_guesses
        return pred, prob, None
    else:
        if prob is not None:
            if joint:
                if ptr is None:
                    inds_topk = torch.topk((prob / prob.sum(dim=-1, keepdim=True)).mean(dim=0, keepdim=True),
                                           k=max_guesses, dim=-1, largest=True, sorted=True)[1]
                    inds_topk = inds_topk.repeat(pred.size(0), 1)
                else:
                    inds_topk = torch.topk(segment_csr(src=prob / prob.sum(dim=-1, keepdim=True), indptr=ptr,
                                                       reduce='mean'),
                                           k=max_guesses, dim=-1, largest=True, sorted=True)[1]
                    inds_topk = gather_csr(src=inds_topk, indptr=ptr)
            else:
                inds_topk = torch.topk(prob, k=max_guesses, dim=-1, largest=True, sorted=True)[1]
            pred_topk = pred[torch.arange(pred.size(0)).unsqueeze(-1).expand(-1, max_guesses), inds_topk]
            prob_topk = prob[torch.arange(pred.size(0)).unsqueeze(-1).expand(-1, max_guesses), inds_topk]
            prob_topk = prob_topk / prob_topk.sum(dim=-1, keepdim=True)
        else:
            pred_topk = pred[:, :max_guesses]
            prob_topk = pred.new_ones((pred.size(0), max_guesses)) / max_guesses
        return pred_topk, prob_topk, inds_topk


def valid_filter(
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
                                                       torch.Tensor, torch.Tensor]:
    if valid_mask is None:
        valid_mask = target.new_ones(target.size()[:-1], dtype=torch.bool)
    if keep_invalid_final_step:
        filter_mask = valid_mask.any(dim=-1)
    else:
        filter_mask = valid_mask[:, -1]
    pred = pred[filter_mask]
    target = target[filter_mask]
    if prob is not None:
        prob = prob[filter_mask]
    valid_mask = valid_mask[filter_mask]
    if ptr is not None:
        num_nodes_batch = segment_csr(src=filter_mask.long(), indptr=ptr, reduce='sum')
        ptr = num_nodes_batch.new_zeros((num_nodes_batch.size(0) + 1,))
        torch.cumsum(num_nodes_batch, dim=0, out=ptr[1:])
    else:
        ptr = target.new_tensor([0, target.size(0)])
    return pred, target, prob, valid_mask, ptr


def new_batch_nms(pred_trajs, dist_thresh, num_ret_modes=6):
    """

    Args:
        pred_trajs (batch_size, num_modes, num_timestamps, 7)
        pred_scores (batch_size, num_modes):
        dist_thresh (float):
        num_ret_modes (int, optional): Defaults to 6.

    Returns:
        ret_trajs (batch_size, num_ret_modes, num_timestamps, 5)
        ret_scores (batch_size, num_ret_modes)
        ret_idxs (batch_size, num_ret_modes)
    """
    batch_size, num_modes, num_timestamps, num_feat_dim = pred_trajs.shape
    pred_goals = pred_trajs[:, :, -1, :]
    dist = (pred_goals[:, :, None, 0:2] - pred_goals[:, None, :, 0:2]).norm(dim=-1)
    nearby_neighbor = dist < dist_thresh
    pred_scores = nearby_neighbor.sum(dim=-1) / num_modes

    sorted_idxs = pred_scores.argsort(dim=-1, descending=True)
    bs_idxs_full = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_modes)
    sorted_pred_scores = pred_scores[bs_idxs_full, sorted_idxs]
    sorted_pred_trajs = pred_trajs[bs_idxs_full, sorted_idxs]  # (batch_size, num_modes, num_timestamps, 7)
    sorted_pred_goals = sorted_pred_trajs[:, :, -1, :]  # (batch_size, num_modes, 7)

    dist = (sorted_pred_goals[:, :, None, 0:2] - sorted_pred_goals[:, None, :, 0:2]).norm(dim=-1)
    point_cover_mask = (dist < dist_thresh)

    point_val = sorted_pred_scores.clone()  # (batch_size, N)
    point_val_selected = torch.zeros_like(point_val)  # (batch_size, N)

    ret_idxs = sorted_idxs.new_zeros(batch_size, num_ret_modes).long()
    ret_trajs = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes, num_timestamps, num_feat_dim)
    ret_scores = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes)
    bs_idxs = torch.arange(batch_size).type_as(ret_idxs)

    for k in range(num_ret_modes):
        cur_idx = point_val.argmax(dim=-1)  # (batch_size)
        ret_idxs[:, k] = cur_idx

        new_cover_mask = point_cover_mask[bs_idxs, cur_idx]  # (batch_size, N)
        point_val = point_val * (~new_cover_mask).float()  # (batch_size, N)
        point_val_selected[bs_idxs, cur_idx] = -1
        point_val += point_val_selected

        ret_trajs[:, k] = sorted_pred_trajs[bs_idxs, cur_idx]
        ret_scores[:, k] = sorted_pred_scores[bs_idxs, cur_idx]

    bs_idxs = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_ret_modes)

    ret_idxs = sorted_idxs[bs_idxs, ret_idxs]
    return ret_trajs, ret_scores, ret_idxs


def batch_nms(pred_trajs, pred_scores,
              dist_thresh, num_ret_modes=6,
              mode='static', speed=None):
    """

    Args:
        pred_trajs (batch_size, num_modes, num_timestamps, 7)
        pred_scores (batch_size, num_modes):
        dist_thresh (float):
        num_ret_modes (int, optional): Defaults to 6.

    Returns:
        ret_trajs (batch_size, num_ret_modes, num_timestamps, 5)
        ret_scores (batch_size, num_ret_modes)
        ret_idxs (batch_size, num_ret_modes)
    """
    batch_size, num_modes, num_timestamps, num_feat_dim = pred_trajs.shape

    sorted_idxs = pred_scores.argsort(dim=-1, descending=True)
    bs_idxs_full = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_modes)
    sorted_pred_scores = pred_scores[bs_idxs_full, sorted_idxs]
    sorted_pred_trajs = pred_trajs[bs_idxs_full, sorted_idxs]  # (batch_size, num_modes, num_timestamps, 7)
    sorted_pred_goals = sorted_pred_trajs[:, :, -1, :]  # (batch_size, num_modes, 7)

    if mode == "speed":
        scale = torch.ones(batch_size).to(sorted_pred_goals.device)
        lon_dist_thresh = 4 * scale
        lat_dist_thresh = 0.5 * scale
        lon_dist = (sorted_pred_goals[:, :, None, [0]] - sorted_pred_goals[:, None, :, [0]]).norm(dim=-1)
        lat_dist = (sorted_pred_goals[:, :, None, [1]] - sorted_pred_goals[:, None, :, [1]]).norm(dim=-1)
        point_cover_mask = (lon_dist < lon_dist_thresh[:, None, None]) & (lat_dist < lat_dist_thresh[:, None, None])
    else:
        dist = (sorted_pred_goals[:, :, None, 0:2] - sorted_pred_goals[:, None, :, 0:2]).norm(dim=-1)
        point_cover_mask = (dist < dist_thresh)

    point_val = sorted_pred_scores.clone()  # (batch_size, N)
    point_val_selected = torch.zeros_like(point_val)  # (batch_size, N)

    ret_idxs = sorted_idxs.new_zeros(batch_size, num_ret_modes).long()
    ret_trajs = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes, num_timestamps, num_feat_dim)
    ret_scores = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes)
    bs_idxs = torch.arange(batch_size).type_as(ret_idxs)

    for k in range(num_ret_modes):
        cur_idx = point_val.argmax(dim=-1)  # (batch_size)
        ret_idxs[:, k] = cur_idx

        new_cover_mask = point_cover_mask[bs_idxs, cur_idx]  # (batch_size, N)
        point_val = point_val * (~new_cover_mask).float()  # (batch_size, N)
        point_val_selected[bs_idxs, cur_idx] = -1
        point_val += point_val_selected

        ret_trajs[:, k] = sorted_pred_trajs[bs_idxs, cur_idx]
        ret_scores[:, k] = sorted_pred_scores[bs_idxs, cur_idx]

    bs_idxs = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_ret_modes)

    ret_idxs = sorted_idxs[bs_idxs, ret_idxs]
    return ret_trajs, ret_scores, ret_idxs


def batch_nms_token(pred_trajs, pred_scores,
                    dist_thresh, num_ret_modes=6,
                    mode='static', speed=None):
    """
    Args:
        pred_trajs (batch_size, num_modes, num_timestamps, 7)
        pred_scores (batch_size, num_modes):
        dist_thresh (float):
        num_ret_modes (int, optional): Defaults to 6.

    Returns:
        ret_trajs (batch_size, num_ret_modes, num_timestamps, 5)
        ret_scores (batch_size, num_ret_modes)
        ret_idxs (batch_size, num_ret_modes)
    """
    batch_size, num_modes, num_feat_dim = pred_trajs.shape

    sorted_idxs = pred_scores.argsort(dim=-1, descending=True)
    bs_idxs_full = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_modes)
    sorted_pred_scores = pred_scores[bs_idxs_full, sorted_idxs]
    sorted_pred_goals = pred_trajs[bs_idxs_full, sorted_idxs]  # (batch_size, num_modes, num_timestamps, 7)

    if mode == "nearby":
        dist = (sorted_pred_goals[:, :, None, 0:2] - sorted_pred_goals[:, None, :, 0:2]).norm(dim=-1)
        values, indices = torch.topk(dist, 5, dim=-1, largest=False)
        thresh_hold = values[..., -1]
        point_cover_mask = dist < thresh_hold[..., None]
    else:
        dist = (sorted_pred_goals[:, :, None, 0:2] - sorted_pred_goals[:, None, :, 0:2]).norm(dim=-1)
        point_cover_mask = (dist < dist_thresh)

    point_val = sorted_pred_scores.clone()  # (batch_size, N)
    point_val_selected = torch.zeros_like(point_val)  # (batch_size, N)

    ret_idxs = sorted_idxs.new_zeros(batch_size, num_ret_modes).long()
    ret_goals = sorted_pred_goals.new_zeros(batch_size, num_ret_modes, num_feat_dim)
    ret_scores = sorted_pred_goals.new_zeros(batch_size, num_ret_modes)
    bs_idxs = torch.arange(batch_size).type_as(ret_idxs)

    for k in range(num_ret_modes):
        cur_idx = point_val.argmax(dim=-1)  # (batch_size)
        ret_idxs[:, k] = cur_idx

        new_cover_mask = point_cover_mask[bs_idxs, cur_idx]  # (batch_size, N)
        point_val = point_val * (~new_cover_mask).float()  # (batch_size, N)
        point_val_selected[bs_idxs, cur_idx] = -1
        point_val += point_val_selected

        ret_goals[:, k] = sorted_pred_goals[bs_idxs, cur_idx]
        ret_scores[:, k] = sorted_pred_scores[bs_idxs, cur_idx]

    bs_idxs = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_ret_modes)

    ret_idxs = sorted_idxs[bs_idxs, ret_idxs]
    return ret_goals, ret_scores, ret_idxs


class TokenCls(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(TokenCls, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               valid_mask: Optional[torch.Tensor] = None) -> None:
        target = target[..., None]
        acc = (pred[:, :self.max_guesses] == target).any(dim=1) * valid_mask
        self.sum += acc.sum()
        self.count += valid_mask.sum()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class minMultiFDE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(minMultiFDE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
        self.sum += torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                               target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                               p=2, dim=-1).min(dim=-1)[0].sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class minFDE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(minFDE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses
        self.eval_timestep = 70

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True) -> None:
        eval_timestep = min(self.eval_timestep, pred.shape[1]) - 1
        self.sum += ((torch.norm(pred[:, eval_timestep-1:eval_timestep] - target[:, eval_timestep-1:eval_timestep], p=2, dim=-1) *
                      valid_mask[:, eval_timestep-1].unsqueeze(1)).sum(dim=-1)).sum()
        self.count += valid_mask[:, eval_timestep-1].sum()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class minMultiADE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(minMultiADE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               min_criterion: str = 'FDE') -> None:
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        if min_criterion == 'FDE':
            inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
            inds_best = torch.norm(
                pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2), p=2, dim=-1).argmin(dim=-1)
            self.sum += ((torch.norm(pred_topk[torch.arange(pred.size(0)), inds_best] - target, p=2, dim=-1) *
                          valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1)).sum()
        elif min_criterion == 'ADE':
            self.sum += ((torch.norm(pred_topk - target.unsqueeze(1), p=2, dim=-1) *
                          valid_mask.unsqueeze(1)).sum(dim=-1).min(dim=-1)[0] / valid_mask.sum(dim=-1)).sum()
        else:
            raise ValueError('{} is not a valid criterion'.format(min_criterion))
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class minADE(Metric):

    def __init__(self,
                 max_guesses: int = 6,
                 **kwargs) -> None:
        super(minADE, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.max_guesses = max_guesses
        self.eval_timestep = 70

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               valid_mask: Optional[torch.Tensor] = None,
               keep_invalid_final_step: bool = True,
               min_criterion: str = 'ADE') -> None:
        # pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        # pred_topk, _ = topk(self.max_guesses, pred, prob)
        # if min_criterion == 'FDE':
        #     inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=self.device)).argmax(dim=-1)
        #     inds_best = torch.norm(
        #         pred[torch.arange(pred.size(0)), :, inds_last] -
        #         target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2), p=2, dim=-1).argmin(dim=-1)
        #     self.sum += ((torch.norm(pred[torch.arange(pred.size(0)), inds_best] - target, p=2, dim=-1) *
        #                   valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1)).sum()
        # elif min_criterion == 'ADE':
        #     self.sum += ((torch.norm(pred - target.unsqueeze(1), p=2, dim=-1) *
        #                   valid_mask.unsqueeze(1)).sum(dim=-1).min(dim=-1)[0] / valid_mask.sum(dim=-1)).sum()
        # else:
        #     raise ValueError('{} is not a valid criterion'.format(min_criterion))
        eval_timestep = min(self.eval_timestep, pred.shape[1])
        self.sum += ((torch.norm(pred[:, :eval_timestep] - target[:, :eval_timestep], p=2, dim=-1) * valid_mask[:, :eval_timestep]).sum(dim=-1) / pred.shape[1]).sum()
        self.count += valid_mask[:, :eval_timestep].any(dim=-1).sum()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class AverageMeter(Metric):

    def __init__(self, **kwargs) -> None:
        super(AverageMeter, self).__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, val: torch.Tensor) -> None:
        self.sum += val.sum()
        self.count += val.numel()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class StateAccuracy(Metric):

    def __init__(self, state_token: Dict[str, int], **kwargs) -> None:
        super().__init__(**kwargs)
        self.invalid_state = int(state_token['invalid'])
        self.valid_state = int(state_token['valid'])
        self.enter_state = int(state_token['enter'])
        self.exit_state = int(state_token['exit'])

        self.add_state('valid', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('valid_count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('invalid', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('invalid_count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               state_idx: torch.Tensor,
               valid_mask: Optional[torch.Tensor] = None) -> None:

        num_agent, num_step = state_idx.shape

        # check the evaluation outputs
        for a in range(num_agent):
            bos_idx = torch.where(state_idx[a] == self.enter_state)[0]
            eos_idx = torch.where(state_idx[a] == self.exit_state)[0]
            bos = 0
            eos = num_step - 1
            if len(bos_idx) > 0:
                bos = bos_idx[0]
                self.invalid += (state_idx[a, :bos] == self.invalid_state).sum()
                self.invalid_count += len(state_idx[a, :bos])
            if len(eos_idx) > 0:
                eos = eos_idx[0]
                self.invalid += (state_idx[a, eos + 1:] == self.invalid_state).sum()
                self.invalid_count += len(state_idx[a, eos + 1:])
            self.valid += (state_idx[a, bos + 1 : eos] == self.valid_state).sum()
            self.valid_count += len(state_idx[a, bos + 1 : eos])

        # check the tokenization
        if valid_mask is not None:

            state_idx = state_idx.roll(shifts=1, dims=1)

            for a in range(num_agent):
                bos_idx = torch.where(state_idx[a] == self.enter_state)[0]
                eos_idx = torch.where(state_idx[a] == self.exit_state)[0]
                bos = 0
                eos = num_step - 1
                if len(bos_idx) > 0:
                    bos = bos_idx[0]
                    self.invalid += (valid_mask[a, :bos] == 0).sum()
                    self.invalid_count += len(valid_mask[a, :bos])
                if len(eos_idx) > 0:
                    eos = eos_idx[-1]
                    self.invalid += (valid_mask[a, eos + 1:] != 0).sum()
                    self.invalid_count += len(valid_mask[a, eos + 1:])
                self.invalid += (((state_idx[a, bos : eos + 1] > 0) != valid_mask[a, bos : eos + 1])[valid_mask[a, bos : eos + 1] == 0]).sum()
                self.invalid_count += (valid_mask[a, bos : eos + 1] == 0).sum()
                self.valid += (((state_idx[a, bos : eos + 1] > 0) != valid_mask[a, bos : eos + 1])[valid_mask[a, bos : eos + 1] == 1]).sum()
                self.valid_count += (valid_mask[a, bos : eos + 1] == 1).sum()

    def compute(self) -> Dict[str, torch.Tensor]:
        return {'valid': self.valid / self.valid_count,
                'invalid': self.invalid / self.invalid_count,
                }

    def __repr__(self):
        head = "Results of " + self.__class__.__name__
        results = self.compute()
        body = [
            "valid: {}".format(results['valid']),
            "invalid: {}".format(results['invalid']),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class GridOverlapRate(Metric):

    def __init__(self, num_step, state_token, seed_size, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_step = num_step
        self.enter_state = int(state_token['enter'])
        self.seed_size = seed_size
        self.add_state('num_overlap_t', default=torch.zeros(num_step).long(), dist_reduce_fx='sum')
        self.add_state('num_insert_agent_t', default=torch.zeros(num_step).long(), dist_reduce_fx='sum')
        self.add_state('num_total_agent_t', default=torch.zeros(num_step).long(), dist_reduce_fx='sum')
        self.add_state('num_exceed_seed_t', default=torch.zeros(num_step).long(), dist_reduce_fx='sum')

    def update(self,
               state_token: torch.Tensor,
               grid_index: torch.Tensor) -> None:

        for t in range(self.num_step):
            inrange_mask_t = grid_index[:, t] != -1
            insert_mask_t = (state_token[:, t] == self.enter_state) & inrange_mask_t
            self.num_total_agent_t[t] += inrange_mask_t.sum()
            self.num_insert_agent_t[t] += insert_mask_t.sum()
            self.num_exceed_seed_t[t] += int(insert_mask_t.sum() >= self.seed_size)

            occupied_grids = set(grid_index[:, t][(grid_index[:, t] != -1) & (state_token[:, t] != self.enter_state)].tolist())
            to_inserted_grids = grid_index[:, t][(grid_index[:, t] != -1) & (state_token[:, t] == self.enter_state)].tolist()
            while to_inserted_grids:
                grid_index_t_i = to_inserted_grids.pop()
                if grid_index_t_i in occupied_grids:
                    self.num_overlap_t[t] += 1
                occupied_grids.add(grid_index_t_i)

    def compute(self) -> Dict[str, torch.Tensor]:
        overlap_rate_t = self.num_overlap_t / self.num_insert_agent_t
        overlap_rate_t.nan_to_num_()
        return {'num_overlap_t': self.num_overlap_t,
                'num_insert_agent_t': self.num_insert_agent_t,
                'num_total_agent_t': self.num_total_agent_t,
                'overlap_rate_t': overlap_rate_t,
                'num_exceed_seed_t': self.num_exceed_seed_t,
                }

    def __repr__(self):
        head = "Results of " + self.__class__.__name__
        results = self.compute()
        body = [
            "num_overlap_t: {}".format(results['num_overlap_t'].tolist()),
            "num_insert_agent_t: {}".format(results['num_insert_agent_t'].tolist()),
            "num_total_agent_t: {}".format(results['num_total_agent_t'].tolist()),
            "overlap_rate_t: {}".format(results['overlap_rate_t'].tolist()),
            "num_exceed_seed_t: {}".format(results['num_exceed_seed_t'].tolist()),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class NumInsertAccuracy(Metric):

    def __init__(self, state_token: Dict[str, int], **kwargs) -> None:
        super().__init__(**kwargs)
        self.invalid_state = int(state_token['invalid'])
        self.valid_state = int(state_token['valid'])
        self.enter_state = int(state_token['enter'])
        self.exit_state = int(state_token['exit'])

        self.add_state('valid', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('valid_count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('invalid', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('invalid_count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               state_idx: torch.Tensor,
               valid_mask: Optional[torch.Tensor] = None) -> None:

        num_agent, num_step = state_idx.shape

        # check the evaluation outputs
        for a in range(num_agent):
            bos_idx = torch.where(state_idx[a] == self.enter_state)[0]
            eos_idx = torch.where(state_idx[a] == self.exit_state)[0]
            bos = 0
            eos = num_step - 1
            if len(bos_idx) > 0:
                bos = bos_idx[0]
                self.invalid += (state_idx[a, :bos] == self.invalid_state).sum()
                self.invalid_count += len(state_idx[a, :bos])
            if len(eos_idx) > 0:
                eos = eos_idx[0]
                self.invalid += (state_idx[a, eos + 1:] == self.invalid_state).sum()
                self.invalid_count += len(state_idx[a, eos + 1:])
            self.valid += (state_idx[a, bos + 1 : eos] == self.valid_state).sum()
            self.valid_count += len(state_idx[a, bos + 1 : eos])

        # check the tokenization
        if valid_mask is not None:

            state_idx = state_idx.roll(shifts=1, dims=1)

            for a in range(num_agent):
                bos_idx = torch.where(state_idx[a] == self.enter_state)[0]
                eos_idx = torch.where(state_idx[a] == self.exit_state)[0]
                bos = 0
                eos = num_step - 1
                if len(bos_idx) > 0:
                    bos = bos_idx[0]
                    self.invalid += (valid_mask[a, :bos] == 0).sum()
                    self.invalid_count += len(valid_mask[a, :bos])
                if len(eos_idx) > 0:
                    eos = eos_idx[-1]
                    self.invalid += (valid_mask[a, eos + 1:] != 0).sum()
                    self.invalid_count += len(valid_mask[a, eos + 1:])
                self.invalid += (((state_idx[a, bos : eos + 1] > 0) != valid_mask[a, bos : eos + 1])[valid_mask[a, bos : eos + 1] == 0]).sum()
                self.invalid_count += (valid_mask[a, bos : eos + 1] == 0).sum()
                self.valid += (((state_idx[a, bos : eos + 1] > 0) != valid_mask[a, bos : eos + 1])[valid_mask[a, bos : eos + 1] == 1]).sum()
                self.valid_count += (valid_mask[a, bos : eos + 1] == 1).sum()

    def compute(self) -> Dict[str, torch.Tensor]:
        return {'valid': self.valid / self.valid_count,
                'invalid': self.invalid / self.invalid_count,
                }

    def __repr__(self):
        head = "Results of " + self.__class__.__name__
        results = self.compute()
        body = [
            "valid: {}".format(results['valid']),
            "invalid: {}".format(results['invalid']),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
