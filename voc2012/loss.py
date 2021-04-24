import torch
import torch.nn as nn
from torch.nn.functional import multilabel_soft_margin_loss


class MyLoss(nn.Module):
    def __init__(self, lambd, beta):
        super(MyLoss, self).__init__()
        self.lambd = lambd
        self.beta = beta

    def forward(self, pred, truth, semantic, res_semantic, feature, deep_semantic):
        loss_cross_entropy = multilabel_soft_margin_loss(pred, truth)
        loss_cosine_distance = torch.mean(torch.cosine_similarity(semantic, res_semantic, dim=1))
        loss_restructure = torch.norm(torch.matmul(pred, deep_semantic) - feature)
        loss_sparse = torch.norm(pred)
        loss = (loss_cross_entropy + self.beta * (loss_restructure + self.lambd * loss_sparse))/loss_cosine_distance
        return loss
