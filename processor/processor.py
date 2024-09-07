import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torchvision
import torch.nn.functional as F
import numpy as np
import math
import random
from loss.contrastive_loss import ContrastiveLoss, WeightedContrastiveLoss1, WeightedContrastiveLoss2, \
    WeightedContrastiveLoss3, WeightedContrastiveLoss4
from loss.triplet_loss import TripletLoss
from PIL import Image
from BFG.micro import micro
from loss.KL_loss import KL_Loss1, KL_Loss2, KL_Loss3

from matplotlib import cm
import matplotlib.pyplot as plt


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    colors = cm.rainbow(np.linspace(0, 21, len(label)))
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=colors[label[i]],
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


# L1
def l1_regularizer(weight, lambda_l1):
    return lambda_l1 * torch.norm(weight, 1)


# L2
def l2_regularizer(weight, lambda_l2):
    return lambda_l2 * torch.norm(weight, 2)


def do_train(cfg, model, model_, models, center_criterion, train_loader, val_loader, loss_fn, num_query,
             local_rank, bfg_model, optimizer_encoder, scheduler_encoder):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None

    micro_cou = micro()

    Re_loss = nn.MSELoss()
    KL_loss = KL_Loss2()
    Tan = nn.Tanh()
    contra = WeightedContrastiveLoss4(cfg.SOLVER.IMS_PER_BATCH)

    if device:
        model.to(local_rank)
        model_.to(local_rank)
        bfg_model.to(local_rank)
        micro_cou.to(local_rank)
        models.to(local_rank)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    en_meter1 = AverageMeter()
    en_meter2 = AverageMeter()
    rec_meter = AverageMeter()
    loss_kl = AverageMeter()
    loss_re = AverageMeter()
    loss_vae = AverageMeter()
    contra_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        en_meter1.reset()
        en_meter2.reset()
        rec_meter.reset()
        loss_kl.reset()
        loss_re.reset()
        loss_vae.reset()
        acc_meter.reset()
        evaluator.reset()
        contra_meter.reset()
        if cfg.SOLVER.OPTIMIZER_NAME == 'SGD' or epoch <= cfg.SOLVER.WARMUP_EPOCHS:
            print("scheduler lr...")
            scheduler_encoder.step(epoch)

        models.train()
        for q in model.parameters():
            q.requires_grad = True
        for i in model_.parameters():
            i.requires_grad = False
        for j in bfg_model.parameters():
            j.requires_grad = True

        for n_iter, (ori_img, vid, target_cam, target_view, corrupt_id, corrupt_img, img) in enumerate(train_loader):

            img = img.to(device)
            ori_img = ori_img.to(device)
            corrupt_img = corrupt_img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            with amp.autocast(enabled=True):
                score, feat, feat_ori, feat_dec = model(corrupt_img,
                                                        target,
                                                        cam_label=target_cam,
                                                        view_label=target_view)
                score_, feat_, feat_ori_, feat_dec_ = model_(img,
                                                             target,
                                                             cam_label=target_cam,
                                                             view_label=target_view)

                cl_feat, cl_mu, cl_sigma = bfg_model(feat_)
                t = random.random()
                bbl_feats = micro_cou(feat_, feat, t)
                bbl_feat, bbl_mu, bbl_sigma = bfg_model(bbl_feats)
                re_feat, re_mu, re_sigma = bfg_model(feat)

                clear_feat = torch.mean(feat_ori, dim=1)

                image = ori_img.to(torch.float32)

                kl_loss = KL_loss(re_mu, re_sigma, bbl_mu, bbl_sigma)
                re_loss = Re_loss(feat_, cl_feat)
                vae_loss = kl_loss + re_loss

                contrast_loss = contra(feat_, feat)
                
                loss_en1 = loss_fn(score, feat, target, target_cam)
                loss_en2 = loss_fn(score, re_feat, target, target_cam)
                loss = loss_en1 + loss_en2 + vae_loss + contrast_loss
                # loss1 = loss_en1 + loss_en2 + contrast_loss
                # loss2 = loss_en1 + loss_en2 + vae_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer_encoder)
            scaler.update()
            optimizer_encoder.zero_grad()
            '''
            scaler.scale(loss1).backward(retain_graph=True)
            scaler.step(optimizer_encoder)
            scaler.update()
            optimizer_encoder.zero_grad()
            scaler.scale(loss2).backward()
            scaler.step(optimizer_encoder)
            scaler.update()
            optimizer_encoder.zero_grad()
            '''

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)
            en_meter1.update(loss_en1, img.shape[0])
            en_meter2.update(loss_en2, img.shape[0])
            loss_kl.update(kl_loss, img.shape[0])
            loss_re.update(re_loss, img.shape[0])
            loss_vae.update(vae_loss, img.shape[0])
            contra_meter.update(contrast_loss, img.shape[0])

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, "
                    "Contrast_Loss: {:.3f}, EN_Loss: {:.3f}, REN_Loss: {:.3f}, "
                    "KL_Loss: {:.3f}, Re_Loss: {:.3f}, VAE_Loss: {:.3f}, Encoder Base Lr: {:.2e},"
                    .format(epoch, (n_iter + 1), len(train_loader),
                            loss_meter.avg, acc_meter.avg, contra_meter.avg, en_meter1.avg, en_meter2.avg,
                            loss_kl.avg, loss_re.avg, loss_vae.avg,
                            optimizer_encoder.state_dict()['param_groups'][0]['lr']))


        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        pix_std = np.array(cfg.INPUT.PIXEL_STD)
        pix_mean = np.array(cfg.INPUT.PIXEL_MEAN)

        if epoch % 10 == 0 or epoch == 1 or epoch == 118:
            torch.save(model.state_dict(),
                       os.path.join(cfg.OUTPUT_DIR + '/' + 'model_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()
            for q in model.parameters():
                q.requires_grad = False
                
            name = [
                "Clean eval", "Corrupted eval", "Corrupted query",
                "Corrupted gallery"
            ]
            for loader_i in range(4):
                print("Evaluating on ", name[loader_i])
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader[loader_i]):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        image = img.to(torch.float32)
                        feat, feat_ori = model(image,
                                               cam_label=camids,
                                               view_label=target_view)
                        clear_feat = torch.mean(feat_ori, dim=1)
                        clear_feat = clear_feat.to(torch.float32)
                        evaluator.update((clear_feat, vid, camid))
                cmc, mAP, mINP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mINP: {:.2%}".format(mINP))
                logger.info("mAP: {:.2%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_inference(cfg, model, val_loader, num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        model.to(device)

    model.eval()
    img_path_list = []
    pid_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            image = img.to(torch.float32)
            feat, feat_ori = model(image,
                                   cam_label=camids,
                                   view_label=target_view)
            clear_feat = torch.mean(feat_ori, dim=1)
            clear_feat = clear_feat.to(torch.float32)
            evaluator.update((clear_feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, mINP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mINP: {:.2%}".format(mINP))
    logger.info("mAP: {:.2%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
    return mINP, mAP, cmc[0], cmc[4], cmc[9]

