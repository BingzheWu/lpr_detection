import torch
import numpy as np
import torch.nn.functional as F

def logloss(pr, pr_pred, szs, eps = 10e-10):
    """
    pr: the probability indicates wheather an object is exsited in a sub-region. 
    pr_pred: prediction value obtained by the neural network
    """
    b, h, w, ch = szs
    pr_pred = -torch.log(pr_pred)
    pr_pred = pr_pred * pr
    pr_pred = torch.reshape(pr_pred, (b, h*w*ch))
    pr_pred = torch.sum(pr_pred, 1)
    return pr_pred
def l1(gt, pred, szs):
    b, h, w, ch = szs
    res = torch.reshape(gt-pred, (b, h*w*ch))
    res = torch.abs(res)
    res = torch.sum(res, 1)
    return res
def loss(pred, target):
    """
    pred: pred values by the neural network, which includes pts for the plate and the objectness in a sub region.(shape[b,h,w,8])
    target:[b,h,w,9]
    """
    b,w,h,c = pred.size()
    obj_probs_true = target[...,0]
    obj_probs_pred = pred[..., 0]
    non_obj_probs_true = 1. - obj_probs_true
    non_obj_probs_pred = pred[...,1]
    affine_pred = pred[..., 2:]
    pts_true = target[..., 1:]
    affinex = torch.stack([torch.max(affine_pred[...,0], torch.Tensor([0.]).cuda()), affine_pred[...,1], affine_pred[..., 2]], 3)
    affiney = torch.stack([affine_pred[..., 3], torch.max(affine_pred[..., 4], torch.Tensor([0.]).cuda()), affine_pred[..., 5]],3)
    v = 0.5
    base = torch.Tensor([[[[-v,-v,1., v,-v,1., v,v,1., -v,v,1.]]]])
    base = base.repeat(b, h, w, 1)
    affiney.cuda()
    affinex.cuda()
    base.cuda()
    pts = []
    for i in range(0, 12, 3):
        row = base[..., i:(i+3)]
        row = row.cuda()
        ptsx = torch.sum(affinex*row, 3)
        ptsy = torch.sum(affiney*row, 3)
        pts_xy = torch.stack([ptsx, ptsy],3 )
        pts.append(pts_xy)
    pts = torch.cat(pts, 3)
    flags = torch.reshape(obj_probs_true, (b, h, w, 1))
    res = 1*l1(pts_true*flags, pts*flags, (b, h, w, 4*2))
    res += 1.*logloss(obj_probs_true, obj_probs_pred, (b, h, w, 1))
    res += 1.*logloss(non_obj_probs_true, non_obj_probs_pred, (b, h, w, 1))
    return res.mean()
def test_loss():
    target = torch.zeros((1, 26, 26, 9))
    pred = torch.zeros((1, 26, 26, 8))
    loss(pred, target)
if __name__ == '__main__':
    test_loss()