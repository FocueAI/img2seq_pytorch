import torch


def configure_optimizers(model, init_lr, weight_decay, milestones, gamma):
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    return optimizer, scheduler
