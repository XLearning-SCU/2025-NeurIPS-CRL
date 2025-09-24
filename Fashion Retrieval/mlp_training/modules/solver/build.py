import torch
from torch.optim.lr_scheduler import StepLR


def build_optimizer(cfg, model):
    if not cfg.MODEL.LOCAL.ENABLE:
        parameters = [{'params': filter(lambda p: p.requires_grad, model.parameters())}]
    else:
        parameters = [
            {'params': filter(lambda p: p.requires_grad, model.choices['local']['basenet'].parameters())},
            {'params': filter(lambda p: p.requires_grad, model.choices['local']['attnnet'].parameters())},
            {'params': filter(lambda p: p.requires_grad, model.choices['global'].parameters()), 'lr': cfg.SOLVER.BASE_LR_SLOW}
        ]
    optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(
        parameters,
        lr=cfg.SOLVER.BASE_LR
    )
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    return StepLR(
        optimizer,
        step_size=cfg.SOLVER.STEP_SIZE,
        gamma=cfg.SOLVER.DECAY_RATE
    )