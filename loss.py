import torch
import torch.nn as nn
import torch.nn.functional as F


def advLoss(source, target, device):

    sourceLabel = torch.ones(len(source))
    targetLabel = torch.zeros(len(target))
    Loss = nn.BCELoss()
    if device == 'cuda':
        Loss = Loss.cuda()
        sourceLabel, targetLabel = sourceLabel.cuda(), targetLabel.cuda()
    #print("sd={}\ntd={}".format(source, target))
    loss = Loss(source, sourceLabel) + Loss(target, targetLabel)
    return loss*0.5


if __name__ == "__main__":
    pass