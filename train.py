from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
import torch.nn as nn
from BFG.BFG_Net import BFG


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file",
                        default="",
                        help="path to config file",
                        type=str)

    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, corrupted_val_loader, \
    corrupted_query_loader, corrupted_gallery_loader, num_query, num_classes, \
    camera_num, view_num, dataset = make_dataloader(cfg)

    loader_list = [
        val_loader, corrupted_val_loader, corrupted_query_loader,
        corrupted_gallery_loader
    ]

    model = make_model(cfg,
                       num_class=num_classes,
                       camera_num=camera_num,
                       view_num=view_num)
    model_ = make_model(cfg,
                        num_class=num_classes,
                        camera_num=camera_num,
                        view_num=view_num)
    bfg_model = BFG(input_dim=768, output_dim=768, hidden_dim=128, latent_dim=16)

    model.load_param("./pth/Market1501/pertrain/pretrain_cor.pth")
    model_.load_param("./pth/Market1501/pertrain/pretrain.pth")
    bfg_model.load_param("./pth/BrowBir/BB_120.pth")

    models = nn.Sequential(model, bfg_model)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    optimizer_encoder = torch.optim.SGD(models.parameters(), lr=0.05, weight_decay=0.001)
    scheduler_encoder = create_scheduler(cfg, optimizer_encoder)

    do_train(cfg, model, model_, models, center_criterion, train_loader, loader_list,
             loss_func, num_query, args.local_rank, bfg_model,
             optimizer_encoder, scheduler_encoder)
