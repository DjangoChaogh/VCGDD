
# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import cv2
from utils1 import rect_2_cxy_wh, cxy_wh_2_rect, get_subwindow_tracking, sobel, im_to_torch
import torch.autograd as autograd
from scipy import ndimage
from skimage.restoration import denoise_wavelet
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from os.path import realpath, dirname, join, isdir, exists
import os

import matplotlib.pyplot as plt
import random

# import time


def generate_anchor(total_stride, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    size = total_stride * total_stride
    count = 0
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size / 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


class TrackerConfig(object):
    # These are the default hyper-params for DaSiamRPN 0.3827
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127  # input z size
    instance_size = 271  # input x size (search region)
    total_stride = 8
    score_size = (instance_size-exemplar_size)/total_stride+1
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    # adaptive change search region #
    adaptive = True

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.score_size = (self.instance_size - self.exemplar_size) / self.total_stride + 1

def response_level_loss(delta, score, gt, target_pos, scale_z, p, gpu_str):
    score_temp = score.permute(1, 2, 3, 0).contiguous().view(2, -1)
    score = torch.transpose(score_temp, 0, 1)  # [1085, 2]
    delta1 = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()  # [4, 1085]

    # calculate proposals, encode the distance between last bbox and anchors
    gt_cen = rect_2_cxy_wh(gt)
    gt_cen = np.tile(gt_cen, (p.anchor.shape[0], 1))
    gt_cen[:, 0] = ((gt_cen[:, 0] - target_pos[0]) * scale_z - p.anchor[:, 0]) / p.anchor[:, 2]
    gt_cen[:, 1] = ((gt_cen[:, 1] - target_pos[1]) * scale_z - p.anchor[:, 1]) / p.anchor[:, 3]
    gt_cen[:, 2] = np.log(gt_cen[:, 2] * scale_z) / p.anchor[:, 2]
    gt_cen[:, 3] = np.log(gt_cen[:, 3] * scale_z) / p.anchor[:, 3]

    # create pseudo proposals randomly, encode the perturbed distance between last bbox and anchors by perturbing coordinate of last bbox
    gt_cen_pseudo = rect_2_cxy_wh(gt)
    gt_cen_pseudo = np.tile(gt_cen_pseudo, (p.anchor.shape[0], 1))

    rate_xy1 = np.random.uniform(0.3, 0.5)
    rate_xy2 = np.random.uniform(0.3, 0.5)
    rate_wd = np.random.uniform(0.7, 0.9)

    gt_cen_pseudo[:, 0] = ((gt_cen_pseudo[:, 0] - target_pos[0] - rate_xy1 * gt_cen_pseudo[:, 2]) * scale_z - p.anchor[
                                                                                                              :,
                                                                                                              0]) / p.anchor[
                                                                                                                    :,
                                                                                                                    2]
    gt_cen_pseudo[:, 1] = ((gt_cen_pseudo[:, 1] - target_pos[1] - rate_xy2 * gt_cen_pseudo[:, 3]) * scale_z - p.anchor[
                                                                                                              :,
                                                                                                              1]) / p.anchor[
                                                                                                                    :,
                                                                                                                    3]
    gt_cen_pseudo[:, 2] = np.log(gt_cen_pseudo[:, 2] * rate_wd * scale_z) / p.anchor[:, 2]
    gt_cen_pseudo[:, 3] = np.log(gt_cen_pseudo[:, 3] * rate_wd * scale_z) / p.anchor[:, 3]

    # decode the output reg score to top-left-wh bbox
    delta[0, :] = (delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]) / scale_z + target_pos[0]
    delta[1, :] = (delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]) / scale_z + target_pos[1]
    delta[2, :] = (np.exp(delta[2, :]) * p.anchor[:, 2]) / scale_z
    delta[3, :] = (np.exp(delta[3, :]) * p.anchor[:, 3]) / scale_z
    location = np.array([delta[0] - delta[2] / 2, delta[1] - delta[3] / 2, delta[2], delta[3]])

    label = overlap_ratio(location, gt)  # it is problematic to set the predicted bbox from last frame as gt

    # set thresold to define positive and negative samples, following the training step
    iou_hi = 0.6
    iou_low = 0.3

    # make labels
    y_pos = np.where(label > iou_hi, 1, 0)
    y_pos = torch.from_numpy(y_pos).to(gpu_str).long()
    y_neg = np.where(label < iou_low, 0, 1)
    y_neg = torch.from_numpy(y_neg).to(gpu_str).long()
    pos_index = np.where(y_pos.cpu() == 1)
    neg_index = np.where(y_neg.cpu() == 0)
    index = np.concatenate((pos_index, neg_index), axis=1)

    # make pseudo lables
    y_pos_pseudo = np.where(label > iou_hi, 0, 1)
    y_pos_pseudo = torch.from_numpy(y_pos_pseudo).to(gpu_str).long()
    y_neg_pseudo = np.where(label < iou_low, 1, 0)
    y_neg_pseudo = torch.from_numpy(y_neg_pseudo).to(gpu_str).long()

    y_truth = y_pos
    y_pseudo = y_pos_pseudo

    # calculate classification loss
    loss_truth_cls = -F.cross_entropy(score[index], y_truth[index])
    loss_pseudo_cls = -F.cross_entropy(score[index], y_pseudo[index])
    loss_cls = (loss_truth_cls - loss_pseudo_cls) * (1)

    # calculate regression loss
    loss_truth_reg = -rpn_smoothL1(delta1, gt_cen, y_pos, gpu_str)
    loss_pseudo_reg = -rpn_smoothL1(delta1, gt_cen_pseudo, y_pos, gpu_str)
    loss_reg = (loss_truth_reg - loss_pseudo_reg) * (5)  # lambada_(reg)=5

    # final adversarial loss
    loss = loss_cls + loss_reg
    return loss

def rtaa_attack(net, x_init, x, gt, target_pos, target_sz, scale_z, p, gpu_str, eps=10, alpha=1, iteration=10, x_val_min=0, x_val_max=255):
    x = Variable(x.data)
    x_adv = Variable(x_init.data, requires_grad=True)

    alpha = eps * 1.0 / iteration  # the max perturbation of pixel value is 10 in RTAA
    losses = []

    for i in range(iteration):
        delta, score, adv_feat = net(x_adv)   # obtain the regression score and classification score
        loss = response_level_loss(delta, score, gt, target_pos, scale_z, p, gpu_str)

        # statistic the convergence of the loss 9.30
        losses.append(loss.cpu().detach().numpy())
        # statistic the convergence of the loss end

        # calculate the derivative
        net.zero_grad()  # clear the gradient of network parameter
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        loss.backward(retain_graph=True)

        adv_grad = where((x_adv.grad > 0) | (x_adv.grad < 0), x_adv.grad, 0)
        adv_grad = torch.sign(adv_grad)
        x_adv = x_adv - alpha * adv_grad  # 因为loss的目标是最小化，所以是减号

        x_adv = where(x_adv > x + eps, x + eps, x_adv)  # clip the value of perturbation that is over the epsilon ball to the boundary of ball
        x_adv = where(x_adv < x - eps, x - eps, x_adv)  # clip the value of perturbation that is over the epsilon ball to the boundary of ball
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        x_adv = Variable(x_adv.data, requires_grad=True)

    return x_adv, losses

def rtaa_bpda_attack(net, x_init, x, gt, target_pos, target_sz, scale_z, p, denoiser, new_config, ic, gpu_str, eps=10, alpha=1, iteration=10, x_val_min=0, x_val_max=255):
    x = Variable(x.data)
    x_adv = Variable(x_init.data, requires_grad=True)

    alpha = eps * 1.0 / iteration  # the max perturbation of pixel value is 10 in RTAA
    losses = []

    for i in range(iteration):
        # purification
        if new_config.purification.ic_select:
            x_crop_pure_dict = denoiser.denoise(x_adv, x, ic=ic)
        else:
            x_crop_pure_dict = denoiser.denoise(x_adv, x)
        x_pure = (x_crop_pure_dict[0][0] * 255.0).to(new_config.device.clf_device)
        x_pure = Variable(x_pure.data, requires_grad=True)

        delta, score, adv_feat = net(x_pure)   # obtain the regression score and classification score
        loss = response_level_loss(delta, score, gt, target_pos, scale_z, p, gpu_str)

        # statistic the convergence of the loss 9.30
        losses.append(loss.cpu().detach().numpy())
        # statistic the convergence of the loss end

        # calculate the derivative
        net.zero_grad()  # clear the gradient of network parameter
        if x_pure.grad is not None:
            x_pure.grad.data.fill_(0)
        loss.backward(retain_graph=True)

        adv_grad = where((x_pure.grad > 0) | (x_pure.grad < 0), x_pure.grad, 0)
        adv_grad = torch.sign(adv_grad)
        x_adv = x_adv - alpha * adv_grad  # 因为loss的目标是最小化，所以是减号

        x_adv = where(x_adv > x + eps, x + eps, x_adv)  # clip the value of perturbation that is over the epsilon ball to the boundary of ball
        x_adv = where(x_adv < x - eps, x - eps, x_adv)  # clip the value of perturbation that is over the epsilon ball to the boundary of ball
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        x_adv = Variable(x_adv.data, requires_grad=True)

    return x_adv, losses

def bdpa_ADA(net, x_init, x, gt, target_pos, target_sz, scale_z, p, gpu_str, netG, eps=10, alpha=1, iteration=10, x_val_min=0, x_val_max=255):
    x = Variable(x.data)
    x_adv = Variable(x_init.data, requires_grad=True)

    alpha = eps * 1.0 / iteration  # the max perturbation of pixel value is 10 in RTAA
    losses = []

    for i in range(iteration):
        h_this = x_adv.shape[2]
        w_this = x_adv.shape[3]
        x_crop_adv = x_adv[:, [2, 1, 0], :, :]  # to RGB
        x = x_crop_adv * 1.0 / 255.0
        x = 2 * x - 1
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        x = netG.forward(x)
        x = F.interpolate(x, size=(h_this, w_this), mode='bilinear', align_corners=True)
        x = (x + 1) * 0.5
        x = x * 255.0
        x_pure = x[:, [2, 1, 0], :, :]  # to BGR
        x_pure = Variable(x_pure.data, requires_grad=True)

        delta, score, adv_feat = net(x_pure)  # obtain the regression score and classification score
        loss = response_level_loss(delta, score, gt, target_pos, scale_z, p, gpu_str)

        # statistic the convergence of the loss 9.30
        losses.append(loss.cpu().detach().numpy())
        # statistic the convergence of the loss end

        # calculate the derivative
        net.zero_grad()  # clear the gradient of network parameter
        if x_pure.grad is not None:
            x_pure.grad.data.fill_(0)
        loss.backward(retain_graph=True)

        adv_grad = where((x_pure.grad > 0) | (x_pure.grad < 0), x_pure.grad, 0)
        adv_grad = torch.sign(adv_grad)
        x_adv = x_adv - alpha * adv_grad  # 因为loss的目标是最小化，所以是减号

        x_adv = where(x_adv > x + eps, x + eps,
                      x_adv)  # clip the value of perturbation that is over the epsilon ball to the boundary of ball
        x_adv = where(x_adv < x - eps, x - eps,
                      x_adv)  # clip the value of perturbation that is over the epsilon ball to the boundary of ball
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        x_adv = Variable(x_adv.data, requires_grad=True)

    return x_adv, losses

def bdpa_DUNET(net, x_init, x, gt, target_pos, target_sz, scale_z, p, gpu_str, net1, eps=10, alpha=1, iteration=10, x_val_min=0, x_val_max=255):
    x = Variable(x.data)
    x_adv = Variable(x_init.data, requires_grad=True)

    alpha = eps * 1.0 / iteration  # the max perturbation of pixel value is 10 in RTAA
    losses = []
    mean_torch = autograd.Variable(
        torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).astype('float32')).to(gpu_str),
        volatile=True)
    std_torch = autograd.Variable(
        torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).astype('float32')).to(gpu_str),
        volatile=True)

    for i in range(iteration):
        h_this = x_adv.shape[2]
        w_this = x_adv.shape[3]
        x_crop_adv = x_adv[:, [2, 1, 0], :, :]  # to RGB
        x = x_crop_adv * 1.0 / 255.0
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        x = (x - mean_torch) / std_torch
        x = net1.forward(x, defense=True, pure_result=True)
        x = x * std_torch + mean_torch
        x = F.interpolate(x, size=(h_this, w_this), mode='bilinear', align_corners=True)
        x = x * 255.0
        x_pure = x[:, [2, 1, 0], :, :]  # to BGR
        x_pure = Variable(x_pure.data, requires_grad=True)

        delta, score, adv_feat = net(x_pure)  # obtain the regression score and classification score
        loss = response_level_loss(delta, score, gt, target_pos, scale_z, p, gpu_str)

        # statistic the convergence of the loss 9.30
        losses.append(loss.cpu().detach().numpy())
        # statistic the convergence of the loss end

        # calculate the derivative
        net.zero_grad()  # clear the gradient of network parameter
        if x_pure.grad is not None:
            x_pure.grad.data.fill_(0)
        loss.backward(retain_graph=True)

        adv_grad = where((x_pure.grad > 0) | (x_pure.grad < 0), x_pure.grad, 0)
        adv_grad = torch.sign(adv_grad)
        x_adv = x_adv - alpha * adv_grad  # 因为loss的目标是最小化，所以是减号

        x_adv = where(x_adv > x + eps, x + eps,
                      x_adv)  # clip the value of perturbation that is over the epsilon ball to the boundary of ball
        x_adv = where(x_adv < x - eps, x - eps,
                      x_adv)  # clip the value of perturbation that is over the epsilon ball to the boundary of ball
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        x_adv = Variable(x_adv.data, requires_grad=True)

    return x_adv, losses

def bdpa_heuristic(net, x_init, x, gt, target_pos, target_sz, scale_z, p, gpu_str, method, eps=10, alpha=1, iteration=10, x_val_min=0, x_val_max=255):
    x = Variable(x.data)
    x_adv = Variable(x_init.data, requires_grad=True)

    alpha = eps * 1.0 / iteration  # the max perturbation of pixel value is 10 in RTAA
    losses = []

    for i in range(iteration):
        x_adv_np = x_adv.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        if method == 'sq':
            interval = 6
            x_pure_np = 1 * x_adv_np
            x_pure_np //= interval
            x_pure_np *= interval
        elif method == 'mf':
            window_size = 2
            x_pure_np = ndimage.filters.median_filter(x_adv_np, size=(1, window_size, window_size), mode='reflect')
        elif method == 'wd':
            sigma = [0.0, 0.01, 0.02, 0.03, 0.04]  # chose a smaller sigma for small perturbation sizes
            x_pure_np = denoise_wavelet(x_adv_np / 255, multichannel=True, convert2ycbcr=True,
                                        method='BayesShrink', mode='soft', sigma=sigma[1])
            x_pure_np *= 255.0

        x_crop_pure = im_to_torch(x_pure_np)
        x_crop_pure = x_crop_pure.unsqueeze(0).to(gpu_str)
        x_pure = Variable(x_crop_pure, requires_grad=True)

        delta, score, adv_feat = net(x_pure)  # obtain the regression score and classification score
        loss = response_level_loss(delta, score, gt, target_pos, scale_z, p, gpu_str)

        # statistic the convergence of the loss 9.30
        losses.append(loss.cpu().detach().numpy())
        # statistic the convergence of the loss end

        # calculate the derivative
        net.zero_grad()  # clear the gradient of network parameter
        if x_pure.grad is not None:
            x_pure.grad.data.fill_(0)
        loss.backward(retain_graph=True)

        adv_grad = where((x_pure.grad > 0) | (x_pure.grad < 0), x_pure.grad, 0)
        adv_grad = torch.sign(adv_grad)
        x_adv = x_adv - alpha * adv_grad  # 因为loss的目标是最小化，所以是减号

        x_adv = where(x_adv > x + eps, x + eps,
                      x_adv)  # clip the value of perturbation that is over the epsilon ball to the boundary of ball
        x_adv = where(x_adv < x - eps, x - eps,
                      x_adv)  # clip the value of perturbation that is over the epsilon ball to the boundary of ball
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        x_adv = Variable(x_adv.data, requires_grad=True)

    return x_adv, losses

# new added 8.11
def afterbefore(adv_feat, ori_feat):
    loss = -F.mse_loss(adv_feat.squeeze(), ori_feat.squeeze())
    return loss

def var(adv_feat, ori_feat):
    adv_activation = F.adaptive_avg_pool2d(adv_feat, (1, 1)).squeeze()
    ori_activation = F.adaptive_avg_pool2d(ori_feat, (1, 1)).squeeze()
    adv_var = torch.var(adv_activation)
    ori_var = torch.var(ori_activation)
    loss = ori_var / adv_var
    return loss

def mean(adv_feat, ori_feat):
    adv_activation = F.adaptive_avg_pool2d(adv_feat, (1, 1)).squeeze()
    ori_activation = F.adaptive_avg_pool2d(ori_feat, (1, 1)).squeeze()
    adv_mean = torch.mean(adv_activation)
    ori_mean = torch.mean(ori_activation)
    loss = ori_mean - adv_mean
    return loss

def active_channel(adv_feat, ori_feat):
    adv_activation = F.adaptive_avg_pool2d(adv_feat, (1, 1)).squeeze()
    ori_activation = F.adaptive_avg_pool2d(ori_feat, (1, 1)).squeeze()
    adv_mean = torch.mean(adv_activation)
    ori_mean = torch.mean(ori_activation)

    num = 0
    loss_iter = None
    for j in range(adv_activation.shape[0]):
        if ((adv_activation[j] > adv_mean) and (ori_activation[j] < ori_mean)) or (
                (adv_activation[j] < adv_mean) and (ori_activation[j] > ori_mean)):
            if loss_iter is None:
                loss_iter = torch.abs(adv_activation[j] - ori_activation[j])
            else:
                loss_iter = loss_iter + torch.abs(adv_activation[j] - ori_activation[j])
            num = num + 1
    if num == 0:
        loss = -(torch.min(adv_activation) / torch.max(adv_activation))
    else:
        loss = -(loss_iter / num)
    return loss

def feat_attack(net, x_crop_init, x_crop, iteration, eps=10, x_val_min=0, x_val_max=255):
    x = Variable(x_crop.data)
    x_adv = Variable(x_crop_init.data, requires_grad=True)
    losses = []

    if iteration != 0:
        alpha = eps * 1.0 / iteration  # the max perturbation of pixel value is 10 in RTAA
    else:
        return x_crop, losses

    _, _, ori_feat = net(x)
    ori_feat_relu = net.after_relu(ori_feat)
    # layer-wise forward 10.20
    # ori_feat = net.layer_forward(x)
    # ori_feat_relu = net.after_relu(ori_feat)
    # layer-wise forward end

    for i in range(iteration):
        _, _, adv_feat = net(x_adv)
        adv_feat_relu = net.after_relu(adv_feat)
        # layer-wise forward 10.20
        # adv_feat = net.layer_forward(x_adv)
        # adv_feat_relu = net.after_relu(adv_feat)
        # layer-wise forward

        # inconsistent activation after and before attack
        # loss = afterbefore(adv_feat, ori_feat)

        # variance attack
        loss = var(adv_feat_relu, ori_feat_relu)

        # mean attack
        # loss = mean(adv_feat, ori_feat)

        # actived channel iou attck
        # loss = active_channel(adv_feat_relu, ori_feat_relu)

        # statistic the convergence of the loss 9.30
        losses.append(loss.cpu().detach().numpy())
        # statistic the convergence of the loss end

        # calculate the derivative
        net.zero_grad()  # clear the gradient of network parameter
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        loss.backward(retain_graph=True)

        adv_grad = where((x_adv.grad > 0) | (x_adv.grad < 0), x_adv.grad, 0)
        adv_grad = torch.sign(adv_grad)
        x_adv = x_adv - alpha * adv_grad

        x_adv = where(x_adv > x + eps, x + eps,
                      x_adv)  # clip the value of perturbation that is over the epsilon ball to the boundary of ball
        x_adv = where(x_adv < x - eps, x - eps,
                      x_adv)  # clip the value of perturbation that is over the epsilon ball to the boundary of ball
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        x_adv = Variable(x_adv.data, requires_grad=True)

    # statistic the convergence of the loss 9.30
    return x_adv, losses
    # statistic the convergence of the loss end

# ensemble response-level and feature-level attack 10.15
def ensmeble_attack(net, x_init, x, gt, target_pos, target_sz, scale_z, p, eps=10, alpha=1, iteration=10, x_val_min=0, x_val_max=255):
    x = Variable(x.data)
    x_adv = Variable(x_init.data, requires_grad=True)
    losses = []

    if iteration != 0:
        alpha = eps * 1.0 / iteration  # the max perturbation of pixel value is 10 in RTAA
    else:
        return x_adv, losses

    _, _, ori_feat = net(x)
    ori_feat_relu = net.after_relu(ori_feat)

    for i in range(iteration):
        delta, score, adv_feat = net(x_adv)  # obtain the regression score and classification score
        adv_feat_relu = net.after_relu(adv_feat)

        loss_response = response_level_loss(delta, score, gt, target_pos, scale_z, p)
        loss_afterbefore = afterbefore(adv_feat, ori_feat)
        loss_var = var(adv_feat_relu, ori_feat_relu)
        loss_mean = mean(adv_feat, ori_feat)
        loss_act_channel = active_channel(adv_feat_relu, ori_feat_relu)

        loss = loss_act_channel*(100/4.5) + loss_afterbefore*(100/2.25) + loss_mean*(100/4.5) + loss_var*(10/4.7) + loss_response*(0.02)
        # loss = loss_response*(0.02) + loss_act_channel*(100/4.5)
        # loss = loss_response * (0.02) + loss_afterbefore*(100/2.25)
        # loss = loss_response * (0.02) + loss_mean*(100/4.5)
        # loss = loss_response * (0.02) + loss_var*(10/4.7)
        # loss = loss_act_channel*(100/4.5) + loss_afterbefore*(100/2.25) + loss_mean*(100/4.5) + loss_var*(10/4.7)

        # statistic the convergence of the loss 9.30
        losses.append(loss.cpu().detach().numpy())
        # statistic the convergence of the loss end

        # calculate the derivative
        net.zero_grad()  # clear the gradient of network parameter
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        loss.backward(retain_graph=True)

        adv_grad = where((x_adv.grad > 0) | (x_adv.grad < 0), x_adv.grad, 0)
        adv_grad = torch.sign(adv_grad)
        x_adv = x_adv - alpha * adv_grad

        x_adv = where(x_adv > x + eps, x + eps,
                      x_adv)  # clip the value of perturbation that is over the epsilon ball to the boundary of ball
        x_adv = where(x_adv < x - eps, x - eps,
                      x_adv)  # clip the value of perturbation that is over the epsilon ball to the boundary of ball
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        x_adv = Variable(x_adv.data, requires_grad=True)

    return x_adv, losses
# ensemble response-level and feature-level attack end
# new added end

def tracker_eval_rtaa(net, x_crop, target_pos, target_sz, window, scale_z, p, f, gt, state):
    # new added 6.3 multi-step one-shot ensemble
    if x_crop.shape[0]>1:
        delta_batch, score_batch, _ = net(x_crop)

        delta = torch.sum(delta_batch, dim=0, keepdim=True) / x_crop.shape[0]
        score = torch.sum(score_batch, dim=0, keepdim=True) / x_crop.shape[0]

        # delta = torch.unsqueeze(delta_batch[0], 0)
        # score = torch.unsqueeze(score_batch[-1], 0)
    else:
        delta, score, _ = net(x_crop)
    # new added end

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

    # visualization 9.19
    best_score_id = np.argmax(score)
    best_anchor = best_score_id // 361
    response = score[best_anchor*361: best_anchor*361+361]
    response_vis = response.reshape(19, 19)
    # visualization end

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]



    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore = penalty * score

    # window float
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    target = delta[:, best_pscore_id] / scale_z
    target_sz = target_sz / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])


    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

    return target_pos, target_sz, score[best_pscore_id], response

# new added 8.7
def tracker_eval_feat_attack(net, z_feat, x_adv, x_ori,target_pos, target_sz, window, scale_z, p, f, gt, state):
    # new added 8.7 for vis
    delta, score, adv_feat = net(x_adv)
    # delta_ori, score_ori, ori_feat = net(x_ori)

    # new added 8.8 network architecture modification
    # adv_feat = net.after_relu(x_adv)
    # ori_feat = net.after_relu(x_ori)
    # new added end

    # distribution of spatial activation
    # spatial_adv_feat = torch.mean(adv_feat, dim=1).squeeze().cpu().detach().numpy()
    # spatial_ori_feat = torch.mean(ori_feat, dim=1).squeeze().cpu().detach().numpy()
    # spatial_z_feat = torch.mean(z_feat, dim=1).squeeze().cpu().detach().numpy()
    # spatial_adv_feat_var = np.var(spatial_adv_feat)
    # spatial_ori_feat_var = np.var(spatial_ori_feat)
    #
    # plt.ion()
    # f = plt.figure()
    # for i in range(3):
    #     f.add_subplot(1,3,i+1)
    #     if i==0:
    #         plt.imshow(spatial_z_feat)
    #     elif i==1:
    #         plt.imshow(spatial_ori_feat)
    #     elif i==2:
    #         plt.imshow(spatial_adv_feat)
    # plt.show()

    # distribution of channel activation
    # adv_activation = F.adaptive_avg_pool2d(adv_feat, (1, 1)).squeeze().cpu().detach().numpy()
    # ori_activation = F.adaptive_avg_pool2d(ori_feat, (1, 1)).squeeze().cpu().detach().numpy()
    # # z_activation = F.adaptive_avg_pool2d(z_feat, (1, 1)).squeeze().cpu().detach().numpy()
    # adv_mean = np.mean(adv_activation)
    # adv_var = np.var(adv_activation)
    # ori_mean = np.mean(ori_activation)
    # ori_var = np.var(ori_activation)


    # x = np.arange(256)
    # plt.plot(x, adv_activation, x, ori_activation)
    # plt.plot(x, ori_activation, x, z_activation)
    # plt.plot(x, adv_activation, x, z_activation)
    # plt.show()

    # consistence comparison of channel activation between template and search
    # adv_act = np.where(adv_activation > adv_mean)[0]
    # ori_act = np.where(ori_activation > ori_mean)[0]
    # inter = np.intersect1d(adv_act, ori_act)
    # union = np.union1d(adv_act,ori_act)
    # iou_adv = inter.size /union.size
    # print('666')
    # new added end

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

    # visualization 9.19
    best_score_id = np.argmax(score)
    best_anchor = best_score_id // 361
    response = score[best_anchor * 361: best_anchor * 361 + 361]
    response_vis = response.reshape(19, 19)
    # visualization end

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]



    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore = penalty * score

    # window float
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    best_pscore_id = np.argmax(pscore)

    target = delta[:, best_pscore_id] / scale_z
    target_sz = target_sz / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])


    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

    return target_pos, target_sz, score[best_pscore_id], response
# new added end

def SiamRPN_init(im, target_pos, target_sz, net, gpu_id=None):
    state = dict()
    p = TrackerConfig()
    p.update(net.cfg)  # put cfg into 'self.' container
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    if p.adaptive:
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = 287  # small object big search region
        else:
            p.instance_size = 271

        p.score_size = (p.instance_size - p.exemplar_size) / p.total_stride + 1

    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, int(p.score_size))

    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    _, z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    z = Variable(z_crop.unsqueeze(0))
    # net.temple(z.cuda())
    # new added 8.7 for vis
    if gpu_id is None:
        z_feat = net.temple(z)
    else:
        z_feat = net.temple(z.to(gpu_id))
    # new added end

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    return state, z_feat

# new added 9.20 the similarity comparison between feature-level and pixel-level
def feat_similarity(net, x_crop, x_adv1, x_crop_pure):
    delta, score, clean_feat = net(x_crop)
    delta, score, adv_feat = net(x_adv1)
    delta, score, pure_feat = net(x_crop_pure)

    metric = torch.nn.MSELoss()
    clean_adv = metric(clean_feat, adv_feat).item()
    clean_pure = metric(clean_feat, pure_feat).item()
    compare = [clean_adv, clean_pure]
    return compare[1]

def IQA(x_crop, x_adv1, x_crop_pure):
    clean_img = x_crop.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    adv_img = x_adv1.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    pure_img = x_crop_pure.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)

    psnr_val = psnr(clean_img, pure_img)
    ssim_val = ssim(clean_img, pure_img, multichannel=True)
    compare = [psnr_val, ssim_val]
    return compare[1]
# new added end

# new added 5.8
def SiamRPN_defense(state, z_feat, im, f, last_result, att_per, def_per, image_save, denoiser, new_config, iter=10):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    # x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))

    # new added 6.20  select timestep according to image complexity
    x_crop_arr, x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans)

    x_crop = Variable(x_crop).unsqueeze(0)
    x_crop = x_crop.to(new_config.device.clf_device)

    ic = sobel(x_crop_arr)
    # new added end

    # response(label)-based adversarial attack
    if type(att_per) != type(0):
        att_per = att_per.cpu().detach().numpy()
        att_per = np.resize(att_per, (1, x_crop.shape[1], x_crop.shape[2], x_crop.shape[3]))
        att_per = torch.from_numpy(att_per).to(new_config.device.clf_device)
    x_crop_init = x_crop + att_per * 1
    x_crop_init = torch.clamp(x_crop_init, 0, 255)
    x_adv1, losses = rtaa_attack(net, x_crop_init, x_crop, last_result, target_pos, target_sz, scale_z, p, gpu_str=new_config.device.clf_device,
                                 iteration=iter)
    att_per = x_adv1 - x_crop

    # purification
    if new_config.purification.ic_select:
        x_crop_pure_dict = denoiser.denoise(x_adv1, x_crop, ic=ic)
    else:
        x_crop_pure_dict = denoiser.denoise(x_adv1, x_crop)
    x_crop_pure = (x_crop_pure_dict[0][0] * 255.0).to(new_config.device.clf_device)

    target_pos, target_sz, score, response = tracker_eval_rtaa(net, x_crop_pure, target_pos, target_sz * scale_z, window, scale_z, p,
                                                     f,
                                                     last_result, state)

    # new added 9.20 the similarity comparison between feature-level and pixel-level
    # if f == 1:
    #     # def_per = feat_similarity(net, x_crop, x_adv1, x_crop_pure)  # feature-level comparison
    #     def_per = IQA(x_crop, x_adv1, x_crop_pure)  # # pixel-level comparison
    # new added end

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state, att_per, def_per, losses
# new added end

# new added 12.2
def defense_adaptive_attack(state, z_feat, im, f, last_result, att_per, def_per, image_save, denoiser, new_config, iter=10):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    # x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))

    # new added 6.20  select timestep according to image complexity
    x_crop_arr, x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans)

    x_crop = Variable(x_crop).unsqueeze(0)
    x_crop = x_crop.to(new_config.device.clf_device)

    ic = sobel(x_crop_arr)
    # new added end

    # response(label)-based adversarial attack
    if type(att_per) != type(0):
        att_per = att_per.cpu().detach().numpy()
        att_per = np.resize(att_per, (1, x_crop.shape[1], x_crop.shape[2], x_crop.shape[3]))
        att_per = torch.from_numpy(att_per).to(new_config.device.clf_device)
    x_crop_init = x_crop + att_per * 1
    x_crop_init = torch.clamp(x_crop_init, 0, 255)
    x_adv1, losses = rtaa_bpda_attack(net, x_crop_init, x_crop, last_result, target_pos, target_sz, scale_z, p, denoiser, new_config, ic, gpu_str=new_config.device.clf_device,
                                 iteration=iter)
    att_per = x_adv1 - x_crop

    # purification
    if new_config.purification.ic_select:
        x_crop_pure_dict = denoiser.denoise(x_adv1, x_crop, ic=ic)
    else:
        x_crop_pure_dict = denoiser.denoise(x_adv1, x_crop)
    x_crop_pure = (x_crop_pure_dict[0][0] * 255.0).to(new_config.device.clf_device)

    target_pos, target_sz, score, response = tracker_eval_rtaa(net, x_crop_pure, target_pos, target_sz * scale_z, window, scale_z, p,
                                                     f,
                                                     last_result, state)

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state, att_per, def_per, losses
# new added end

# new added 5.19
def SiamRPN_purify_clean(state, im, f, last_result, denoiser, new_config):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    # x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))

    # new added 6.20  select timestep according to image complexity
    x_crop_arr, x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans)
    x_crop = Variable(x_crop).unsqueeze(0)
    x_crop = x_crop.to(new_config.device.clf_device)

    ic = sobel(x_crop_arr)
    # new added end

    # purification
    if new_config.purification.ic_select:
        x_crop_pure_dict = denoiser.denoise(x_crop, x_crop, ic=ic)
    else:
        x_crop_pure_dict = denoiser.denoise(x_crop, x_crop)
    x_crop_pure = (x_crop_pure_dict[0][0] * 255.0).to(new_config.device.clf_device)

    target_pos, target_sz, score, response = tracker_eval_rtaa(net, x_crop_pure, target_pos, target_sz * scale_z, window, scale_z, p,
                                                     f,
                                                     last_result, state)

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state
# new added end

# new added 9.6
def ADA_pure_adv(state, z_feat, im, f, last_result, att_per, def_per, image_save, netG, gpu_str, iter=10):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x_crop_arr, x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans)
    x_crop = Variable(x_crop).unsqueeze(0)
    x_crop = x_crop.to(gpu_str)

    # response(label)-based adversarial attack
    if type(att_per) != type(0):
        att_per = att_per.cpu().detach().numpy()
        att_per = np.resize(att_per, (1, x_crop.shape[1], x_crop.shape[2], x_crop.shape[3]))
        att_per = torch.from_numpy(att_per).to(gpu_str)
    x_crop_init = x_crop + att_per * 1
    x_crop_init = torch.clamp(x_crop_init, 0, 255)
    x_adv1, losses = rtaa_attack(net, x_crop_init, x_crop, last_result, target_pos, target_sz, scale_z, p, gpu_str,
                                 iteration=iter)
    att_per = x_adv1 - x_crop

    # purification
    h_this = x_adv1.shape[2]
    w_this = x_adv1.shape[3]
    x_crop_adv = x_adv1[:, [2, 1, 0], :, :]  # to RGB
    x = x_crop_adv * 1.0 / 255.0
    x = 2 * x - 1
    x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
    x = netG.forward(x)
    x = F.interpolate(x, size=(h_this, w_this), mode='bilinear', align_corners=True)
    x = (x + 1) * 0.5
    x = x * 255.0
    x_crop_pure = x[:, [2, 1, 0], :, :]  # to BGR


    target_pos, target_sz, score, response = tracker_eval_rtaa(net, x_crop_pure, target_pos, target_sz * scale_z, window, scale_z, p,
                                                     f,
                                                     last_result, state)

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state, att_per, def_per, losses

def DAA_ADA(state, z_feat, im, f, last_result, att_per, def_per, image_save, netG, gpu_str, iter=10):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x_crop_arr, x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans)
    x_crop = Variable(x_crop).unsqueeze(0)
    x_crop = x_crop.to(gpu_str)

    # response(label)-based adversarial attack
    if type(att_per) != type(0):
        att_per = att_per.cpu().detach().numpy()
        att_per = np.resize(att_per, (1, x_crop.shape[1], x_crop.shape[2], x_crop.shape[3]))
        att_per = torch.from_numpy(att_per).to(gpu_str)
    x_crop_init = x_crop + att_per * 1
    x_crop_init = torch.clamp(x_crop_init, 0, 255)
    x_adv1, losses = bdpa_ADA(net, x_crop_init, x_crop, last_result, target_pos, target_sz, scale_z, p, gpu_str, netG,
                                 iteration=iter)
    att_per = x_adv1 - x_crop

    # purification
    h_this = x_adv1.shape[2]
    w_this = x_adv1.shape[3]
    x_crop_adv = x_adv1[:, [2, 1, 0], :, :]  # to RGB
    x = x_crop_adv * 1.0 / 255.0
    x = 2 * x - 1
    x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
    x = netG.forward(x)
    x = F.interpolate(x, size=(h_this, w_this), mode='bilinear', align_corners=True)
    x = (x + 1) * 0.5
    x = x * 255.0
    x_crop_pure = x[:, [2, 1, 0], :, :]  # to BGR


    target_pos, target_sz, score, response = tracker_eval_rtaa(net, x_crop_pure, target_pos, target_sz * scale_z, window, scale_z, p,
                                                     f,
                                                     last_result, state)

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state, att_per, def_per, losses

def ADA_pure_clean(state, im, f, last_result, netG, gpu_str):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x_crop_arr, x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans)
    x_crop = Variable(x_crop).unsqueeze(0)
    x_crop = x_crop.to(gpu_str)

    # purification
    h_this = x_crop.shape[2]
    w_this = x_crop.shape[3]
    x_crop_rgb = x_crop[:, [2, 1, 0], :, :]  # to RGB
    x = x_crop_rgb * 1.0 / 255.0
    x = 2 * x - 1
    x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
    x = netG.forward(x)
    x = F.interpolate(x, size=(h_this, w_this), mode='bilinear', align_corners=True)
    x = (x + 1) * 0.5
    x = x * 255.0
    x_crop_pure = x[:, [2, 1, 0], :, :]  # to BGR


    target_pos, target_sz, score, response = tracker_eval_rtaa(net, x_crop_pure, target_pos, target_sz * scale_z, window, scale_z, p,
                                                     f,
                                                     last_result, state)

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state

def DUNET_pure_adv(state, z_feat, im, f, last_result, att_per, def_per, image_save, net1, gpu_str, iter=10):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x_crop_arr, x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans)
    x_crop = Variable(x_crop).unsqueeze(0)
    x_crop = x_crop.to(gpu_str)

    # response(label)-based adversarial attack
    if type(att_per) != type(0):
        att_per = att_per.cpu().detach().numpy()
        att_per = np.resize(att_per, (1, x_crop.shape[1], x_crop.shape[2], x_crop.shape[3]))
        att_per = torch.from_numpy(att_per).to(gpu_str)
    x_crop_init = x_crop + att_per * 1
    x_crop_init = torch.clamp(x_crop_init, 0, 255)
    x_adv1, losses = rtaa_attack(net, x_crop_init, x_crop, last_result, target_pos, target_sz, scale_z, p, gpu_str,
                                 iteration=iter)
    att_per = x_adv1 - x_crop

    # purification
    mean_torch = autograd.Variable(
        torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).astype('float32')).to(gpu_str), volatile=True)
    std_torch = autograd.Variable(
        torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).astype('float32')).to(gpu_str), volatile=True)

    h_this = x_adv1.shape[2]
    w_this = x_adv1.shape[3]
    x_crop_adv = x_adv1[:, [2, 1, 0], :, :]  # to RGB
    x = x_crop_adv * 1.0 / 255.0
    x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
    x = (x - mean_torch) / std_torch
    x = net1.forward(x, defense=True, pure_result=True)
    x = x * std_torch + mean_torch
    x = F.interpolate(x, size=(h_this, w_this), mode='bilinear', align_corners=True)
    x = x * 255.0
    x_crop_pure = x[:, [2, 1, 0], :, :]  # to BGR

    # pure_vis = x_crop_pure.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    # x_vis = x_crop.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    # compare = np.hstack([x_vis, pure_vis])
    # cv2.imshow('perturbation', compare)
    # cv2.waitKey(1)

    target_pos, target_sz, score, response = tracker_eval_rtaa(net, x_crop_pure, target_pos, target_sz * scale_z, window, scale_z, p,
                                                     f,
                                                     last_result, state)

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state, att_per, def_per, losses

def DAA_DUNET(state, z_feat, im, f, last_result, att_per, def_per, image_save, net1, gpu_str, iter=10):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x_crop_arr, x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans)
    x_crop = Variable(x_crop).unsqueeze(0)
    x_crop = x_crop.to(gpu_str)

    # response(label)-based adversarial attack
    if type(att_per) != type(0):
        att_per = att_per.cpu().detach().numpy()
        att_per = np.resize(att_per, (1, x_crop.shape[1], x_crop.shape[2], x_crop.shape[3]))
        att_per = torch.from_numpy(att_per).to(gpu_str)
    x_crop_init = x_crop + att_per * 1
    x_crop_init = torch.clamp(x_crop_init, 0, 255)
    x_adv1, losses = bdpa_DUNET(net, x_crop_init, x_crop, last_result, target_pos, target_sz, scale_z, p, gpu_str, net1,
                                 iteration=iter)
    att_per = x_adv1 - x_crop

    # purification
    mean_torch = autograd.Variable(
        torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).astype('float32')).to(gpu_str), volatile=True)
    std_torch = autograd.Variable(
        torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).astype('float32')).to(gpu_str), volatile=True)

    h_this = x_adv1.shape[2]
    w_this = x_adv1.shape[3]
    x_crop_adv = x_adv1[:, [2, 1, 0], :, :]  # to RGB
    x = x_crop_adv * 1.0 / 255.0
    x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
    x = (x - mean_torch) / std_torch
    x = net1.forward(x, defense=True, pure_result=True)
    x = x * std_torch + mean_torch
    x = F.interpolate(x, size=(h_this, w_this), mode='bilinear', align_corners=True)
    x = x * 255.0
    x_crop_pure = x[:, [2, 1, 0], :, :]  # to BGR

    # pure_vis = x_crop_pure.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    # x_vis = x_crop.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    # compare = np.hstack([x_vis, pure_vis])
    # cv2.imshow('perturbation', compare)
    # cv2.waitKey(1)

    target_pos, target_sz, score, response = tracker_eval_rtaa(net, x_crop_pure, target_pos, target_sz * scale_z, window, scale_z, p,
                                                     f,
                                                     last_result, state)

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state, att_per, def_per, losses

def DUNET_pure_clean(state, im, f, last_result, net1, gpu_str):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x_crop_arr, x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans)
    x_crop = Variable(x_crop).unsqueeze(0)
    x_crop = x_crop.to(gpu_str)

    # purification
    mean_torch = autograd.Variable(
        torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).astype('float32')).to(gpu_str), volatile=True)
    std_torch = autograd.Variable(
        torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).astype('float32')).to(gpu_str), volatile=True)

    h_this = x_crop.shape[2]
    w_this = x_crop.shape[3]
    x_crop_rgb = x_crop[:, [2, 1, 0], :, :]  # to RGB
    x = x_crop_rgb * 1.0 / 255.0
    x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
    x = (x - mean_torch) / std_torch
    x = net1.forward(x, defense=True, pure_result=True)
    x = x * std_torch + mean_torch
    x = F.interpolate(x, size=(h_this, w_this), mode='bilinear', align_corners=True)
    x = x * 255.0
    x_crop_pure = x[:, [2, 1, 0], :, :]  # to BGR

    # pure_vis = x_crop_pure.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    # x_vis = x_crop.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    # compare = np.hstack([x_vis, pure_vis])
    # cv2.imshow('perturbation', compare)
    # cv2.waitKey(1)

    target_pos, target_sz, score, response = tracker_eval_rtaa(net, x_crop_pure, target_pos, target_sz * scale_z, window, scale_z, p,
                                                     f,
                                                     last_result, state)

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state

def heuristic_pure_adv(state, z_feat, im, f, last_result, att_per, def_per, image_save, method, gpu_str, iter=10):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x_crop_arr, x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans)
    x_crop = Variable(x_crop).unsqueeze(0)
    x_crop = x_crop.to(gpu_str)

    # response(label)-based adversarial attack
    if type(att_per) != type(0):
        att_per = att_per.cpu().detach().numpy()
        att_per = np.resize(att_per, (1, x_crop.shape[1], x_crop.shape[2], x_crop.shape[3]))
        att_per = torch.from_numpy(att_per).to(gpu_str)
    x_crop_init = x_crop + att_per * 1
    x_crop_init = torch.clamp(x_crop_init, 0, 255)
    x_adv1, losses = rtaa_attack(net, x_crop_init, x_crop, last_result, target_pos, target_sz, scale_z, p, gpu_str,
                                 iteration=iter)
    att_per = x_adv1 - x_crop

    # purification
    x_adv_np = x_adv1.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    if method == 'sq':
        interval = 6
        x_pure_np = 1*x_adv_np
        x_pure_np //= interval
        x_pure_np *= interval
    elif method == 'mf':
        window_size = 2
        x_pure_np = ndimage.filters.median_filter(x_adv_np, size=(1, window_size, window_size), mode='reflect')
    elif method == 'wd':
        sigma = [0.0, 0.01, 0.02, 0.03, 0.04]  # chose a smaller sigma for small perturbation sizes
        x_pure_np = denoise_wavelet(x_adv_np / 255, multichannel=True, convert2ycbcr=True,
                                    method='BayesShrink', mode='soft', sigma=sigma[1])
        x_pure_np *= 255.0

    x_crop_pure = im_to_torch(x_pure_np)
    x_crop_pure = Variable(x_crop_pure).unsqueeze(0)
    x_crop_pure = x_crop_pure.to(gpu_str)

    # pure_vis = x_crop_pure.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    # x_vis = x_crop.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    # compare = np.hstack([x_vis, pure_vis])
    # cv2.imshow('perturbation', compare)
    # cv2.waitKey(1)

    target_pos, target_sz, score, response = tracker_eval_rtaa(net, x_crop_pure, target_pos, target_sz * scale_z, window, scale_z, p,
                                                     f,
                                                     last_result, state)

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state, att_per, def_per, losses

def DAA_heuristic(state, z_feat, im, f, last_result, att_per, def_per, image_save, method, gpu_str, iter=10):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x_crop_arr, x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans)
    x_crop = Variable(x_crop).unsqueeze(0)
    x_crop = x_crop.to(gpu_str)

    # response(label)-based adversarial attack
    if type(att_per) != type(0):
        att_per = att_per.cpu().detach().numpy()
        att_per = np.resize(att_per, (1, x_crop.shape[1], x_crop.shape[2], x_crop.shape[3]))
        att_per = torch.from_numpy(att_per).to(gpu_str)
    x_crop_init = x_crop + att_per * 1
    x_crop_init = torch.clamp(x_crop_init, 0, 255)
    x_adv1, losses = bdpa_heuristic(net, x_crop_init, x_crop, last_result, target_pos, target_sz, scale_z, p, gpu_str, method,
                                 iteration=iter)
    att_per = x_adv1 - x_crop

    # purification
    x_adv_np = x_adv1.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    if method == 'sq':
        interval = 6
        x_pure_np = 1*x_adv_np
        x_pure_np //= interval
        x_pure_np *= interval
    elif method == 'mf':
        window_size = 2
        x_pure_np = ndimage.filters.median_filter(x_adv_np, size=(1, window_size, window_size), mode='reflect')
    elif method == 'wd':
        sigma = [0.0, 0.01, 0.02, 0.03, 0.04]  # chose a smaller sigma for small perturbation sizes
        x_pure_np = denoise_wavelet(x_adv_np / 255, multichannel=True, convert2ycbcr=True,
                                    method='BayesShrink', mode='soft', sigma=sigma[1])
        x_pure_np *= 255.0

    x_crop_pure = im_to_torch(x_pure_np)
    x_crop_pure = Variable(x_crop_pure).unsqueeze(0)
    x_crop_pure = x_crop_pure.to(gpu_str)

    target_pos, target_sz, score, response = tracker_eval_rtaa(net, x_crop_pure, target_pos, target_sz * scale_z, window, scale_z, p,
                                                     f,
                                                     last_result, state)

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state, att_per, def_per, losses

def heuristic_pure_clean(state, im, f, last_result, method, gpu_str):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x_crop_arr, x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans)
    # x_crop = Variable(x_crop).unsqueeze(0)
    # x_crop = x_crop.to(gpu_str)

    # purification
    if method == 'sq':
        interval = 6
        x_pure_np = 1*x_crop_arr
        x_pure_np //= interval
        x_pure_np *= interval
    elif method == 'mf':
        window_size = 2
        x_pure_np = ndimage.filters.median_filter(x_crop_arr, size=(1, window_size, window_size), mode='reflect')
    elif method == 'wd':
        sigma = [0.0, 0.01, 0.02, 0.03, 0.04]  # chose a smaller sigma for small perturbation sizes
        x_pure_np = denoise_wavelet(x_crop_arr / 255, multichannel=True, convert2ycbcr=True,
                                    method='BayesShrink', mode='soft', sigma=sigma[1])
        x_pure_np *= 255.0

    x_crop_pure = im_to_torch(x_pure_np)
    x_crop_pure = Variable(x_crop_pure).unsqueeze(0)
    x_crop_pure = x_crop_pure.to(gpu_str)

    # pure_vis = x_crop_pure.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    # x_vis = x_crop.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    # compare = np.hstack([x_vis, pure_vis])
    # cv2.imshow('perturbation', compare)
    # cv2.waitKey(1)

    target_pos, target_sz, score, response = tracker_eval_rtaa(net, x_crop_pure, target_pos, target_sz * scale_z, window, scale_z, p,
                                                     f,
                                                     last_result, state)

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state
# new added end

def SiamRPN_track(state, z_feat, im, f, last_result, att_per, def_per, image_save, iter=10):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    # x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
    _, x_crop = get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans)
    x_crop = Variable(x_crop).unsqueeze(0)
    x_crop = x_crop.cuda()

    # original tracking
    # losses = 0
    # target_pos, target_sz, score, response = tracker_eval_rtaa(net, x_crop, target_pos, target_sz * scale_z, window,
    #                                                            scale_z, p, f, last_result, state)

    # response-level adversarial attack
    if type(att_per) != type(0):
        att_per = att_per.cpu().detach().numpy()
        att_per = np.resize(att_per, (1, x_crop.shape[1], x_crop.shape[2], x_crop.shape[3]))
        att_per = torch.from_numpy(att_per).cuda()
    x_crop_init = x_crop + att_per * 1
    x_crop_init = torch.clamp(x_crop_init, 0, 255)
    x_adv1, losses = rtaa_attack(net, x_crop_init, x_crop, last_result, target_pos, target_sz, scale_z, p, iteration=iter)
    att_per = x_adv1 - x_crop

    target_pos, target_sz, score, response = tracker_eval_rtaa(net, x_adv1, target_pos, target_sz * scale_z, window, scale_z, p, f,
                                                last_result, state)

    # new added 8.11  intermediate-level attack
    # if type(att_per) != type(0):
    #     att_per = att_per.cpu().detach().numpy()
    #     att_per = np.resize(att_per, (1, x_crop.shape[1], x_crop.shape[2], x_crop.shape[3]))
    #     att_per = torch.from_numpy(att_per).cuda()
    # # random init the perturbation if att_per=0
    # else:
    #     att_per = np.random.randint(-10, 11, (1, 3, p.instance_size, p.instance_size))
    #     att_per = torch.from_numpy(att_per).cuda()
    # x_crop_init = x_crop + att_per * 1  # init from last frame of attacked perturbation
    # x_crop_init = torch.clamp(x_crop_init, 0, 255)
    #
    # x_adv1, losses = feat_attack(net, x_crop_init, x_crop, iter)

    # ensemble response-level and feature-level attack 10.15
    # x_adv1, losses = ensmeble_attack(net, x_crop_init, x_crop, last_result, target_pos, target_sz, scale_z, p,
    #                              iteration=iter)
    # ensemble response-level and feature-level attack end

    # att_per = x_adv1 - x_crop
    #
    # target_pos, target_sz, score, response = tracker_eval_feat_attack(net, z_feat, x_adv1, x_crop, target_pos, target_sz * scale_z, window, scale_z, p, f,
    #                                             last_result, state)

    # visualization 9.19
    # if f == 1:
    #     mode = 'random'  # afterbefore, var, mean, active_channel, label_attack
    #     x_adv1_vis = x_adv1.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    #     x_crop_vis = x_crop.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    #
    #     att_per_arr = att_per.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    #     att_per_vis = ((att_per_arr + 10) * 12.75).astype(np.uint8)
    #
    #     compare = np.hstack([x_crop_vis, x_adv1_vis, att_per_vis])
    #
    #     cv2.imshow('perturbation', compare)
    #     cv2.waitKey(1)
    #     main_path = '/data/Disk_A/shaochuan/Projects/RTAA-main/DaSiamRPN/label_VS_feat_compare/Biker/'
    #     save_path = main_path + mode + '.jpg'
    #     response_path = main_path + mode + '_response.jpg'
    #     plt.title(str(score))
    #     plt.imshow(response)
    #     plt.savefig(response_path)
    #     cv2.imwrite(save_path, compare)
    #     print('666')
    # visualization end

    # statistic the convergence of the loss 9.30
    # if f < 11:
    #     mode = 'response_attack'  # afterbefore, var, mean, active_channel, response_attack
    #     main_path = '/data/Disk_A/shaochuan/Projects/RTAA-main/DaSiamRPN/label_VS_feat_compare/ClifBar/'
    #     save_path = join(main_path, mode+'_loss')
    #     if not exists(save_path):
    #         os.mkdir(save_path)
    #     x_axis = np.arange(0, iter)
    #     plt.plot(x_axis, losses)
    #     plt.savefig(join(save_path, str(f)+'.jpg'))
    #     # plt.close()
    # else:
    #     print('444')
    # statistic the convergence of the loss end
    # new added end

    # new added 23.8.5  The mean and variance of the response
    response_mean = np.mean(response)
    response_var = np.var(response)
    # print(response_mean, response_var)
    # new added end

    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    return state, att_per, def_per, np.array(losses), response_mean, response_var

def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)

def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''
    rect1 = np.transpose(rect1)

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

def rpn_smoothL1(input, target, label, gpu_str):
    r"""
    :param input: torch.Size([1, 1125, 4])
    :param target: torch.Size([1, 1125, 4])
            label: (torch.Size([1, 1125]) pos neg or ignore
    :return:
    """
    input = torch.transpose(input, 0, 1)
    pos_index = np.where(label.cpu() == 1)#changed
    target = torch.from_numpy(target).to(gpu_str).float()
    loss = F.smooth_l1_loss(input[pos_index], target[pos_index], reduction='sum')


    return loss

def rect_2_cxy_wh(rect):
    return np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2, rect[2], rect[3]])




