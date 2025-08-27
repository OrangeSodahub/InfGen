import logging
import time
import os
import yaml
import easydict
import math
import torch
import torch.nn as nn
from rich.console import Console
from typing import Any, List, Optional, Mapping

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


CONSOLE = Console(width=128)


def check_nan_inf(t, s):
    assert not torch.isinf(t).any(), f"{s} is inf, {t}"
    assert not torch.isnan(t).any(), f"{s} is nan, {t}"


def safe_list_index(ls: List[Any], elem: Any) -> Optional[int]:
    try:
        return ls.index(elem)
    except ValueError:
        return None


def angle_between_2d_vectors(
        ctr_vector: torch.Tensor,
        nbr_vector: torch.Tensor) -> torch.Tensor:
    return torch.atan2(ctr_vector[..., 0] * nbr_vector[..., 1] - ctr_vector[..., 1] * nbr_vector[..., 0],
                       (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1))


def angle_between_3d_vectors(
        ctr_vector: torch.Tensor,
        nbr_vector: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.cross(ctr_vector, nbr_vector, dim=-1).norm(p=2, dim=-1),
                       (ctr_vector * nbr_vector).sum(dim=-1))


def side_to_directed_lineseg(
        query_point: torch.Tensor,
        start_point: torch.Tensor,
        end_point: torch.Tensor) -> str:
    cond = ((end_point[0] - start_point[0]) * (query_point[1] - start_point[1]) -
            (end_point[1] - start_point[1]) * (query_point[0] - start_point[0]))
    if cond > 0:
        return 'LEFT'
    elif cond < 0:
        return 'RIGHT'
    else:
        return 'CENTER'


def wrap_angle(
        angle: torch.Tensor,
        min_val: float = -math.pi,
        max_val: float = math.pi) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


def load_config_act(path):
    """ load config file"""
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return easydict.EasyDict(cfg)


def load_config_init(path):
    """ load config file"""
    path = os.path.join('init/configs', f'{path}.yaml')
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


class Logging:

    def make_log_dir(self, dirname='logs'):
        now_dir = os.path.dirname(__file__)
        path = os.path.join(now_dir, dirname)
        path = os.path.normpath(path)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def get_log_filename(self):
        filename = "{}.log".format(time.strftime("%Y-%m-%d-%H%M%S", time.localtime()))
        filename = os.path.join(self.make_log_dir(), filename)
        filename = os.path.normpath(filename)
        return filename

    def log(self, level='DEBUG', name="simagent"):
        logger = logging.getLogger(name)
        level = getattr(logging, level)
        logger.setLevel(level)
        if not logger.handlers:
            sh = logging.StreamHandler()
            fh = logging.FileHandler(filename=self.get_log_filename(), mode='a',encoding="utf-8")
            fmt = logging.Formatter("%(asctime)s-%(levelname)s-%(filename)s-Line:%(lineno)d-Message:%(message)s")
            sh.setFormatter(fmt=fmt)
            fh.setFormatter(fmt=fmt)
            logger.addHandler(sh)
            logger.addHandler(fh)
        return logger

    def add_log(self, logger, level='DEBUG'):
        level = getattr(logging, level)
        logger.setLevel(level)
        if not logger.handlers:
            sh = logging.StreamHandler()
            fh = logging.FileHandler(filename=self.get_log_filename(), mode='a',encoding="utf-8")
            fmt = logging.Formatter("%(asctime)s-%(levelname)s-%(filename)s-Line:%(lineno)d-Message:%(message)s")
            sh.setFormatter(fmt=fmt)
            fh.setFormatter(fmt=fmt)
            logger.addHandler(sh)
            logger.addHandler(fh)
        return logger


# Adapted from 'CatK'
class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes
        with their rank prefixed in the log message.

        :param name: The name of the logger. Default is ``__name__``.
        :param rank_zero_only: Whether to force all logs to only occur on the rank zero process. Default is `False`.
        :param extra: (Optional) A dict-like object which provides contextual information. See `logging.LoggerAdapter`.
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(
        self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs
    ) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message with the rank
        of the process it's being logged from. If `'rank'` is provided, then the log will only
        occur on that rank/process.

        :param level: The level to log at. Look at `logging.__init__.py` for more information.
        :param msg: The message to log.
        :param rank: The rank to log at.
        :param args: Additional args to pass to the underlying logging function.
        :param kwargs: Any additional keyword args to pass to the underlying logging function.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError(
                    "The `rank_zero_only.rank` needs to be set before use"
                )
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)



def weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, (nn.GRU, nn.GRUCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)


def pos2posemb(pos, num_pos_feats=128, temperature=10000):

    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    D = pos.shape[-1]
    pos_dims = []
    for i in range(D):
        pos_dim_i = pos[..., i, None] / dim_t
        pos_dim_i = torch.stack((pos_dim_i[..., 0::2].sin(), pos_dim_i[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_dims.append(pos_dim_i)
        posemb = torch.cat(pos_dims, dim=-1)

    return posemb