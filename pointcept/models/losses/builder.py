"""
Criteria Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import inspect
from pointcept.utils.registry import Registry

LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        self.accepts_dict = [] # لیست جدید برای نگهداری وضعیت
        for loss_cfg in self.cfg:
            c = LOSSES.build(cfg=loss_cfg)
            self.criteria.append(c)
            # یک بار در ابتدا بررسی می‌کنیم
            sig = inspect.signature(c.forward)
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            self.accepts_dict.append('input_dict' in sig.parameters or accepts_kwargs)

    def __call__(self, pred, target, **kwargs):
        if len(self.criteria) == 0:
            # loss computation occur in model
            return pred
        loss = 0
        for c in self.criteria:
            sig = inspect.signature(c.forward)
        
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if 'input_dict' in sig.parameters or accepts_kwargs:
                loss += c(pred, target, **kwargs)
            else:
                loss += c(pred, target)
        return loss


def build_criteria(cfg):
    return Criteria(cfg)
