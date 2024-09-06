import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset, CorruptImageDataset
from .bases import rain
from .bases import mixing_erasing
from .bases import corruption_transform
from .sampler import RandomIdentitySampler
from .market1501 import Market1501
from .cuhk03 import CUHK03
from .msmt17 import MSMT17

from PIL import Image
import random
import numpy as np
import albumentations as abm
from imagecorruptions.corruptions import gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, \
    motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression, \
    speckle_noise, gaussian_blur, spatter, saturate

from .augmentations.augmix import augmix
import math


# 输入PIL，输出PIL，放置在所有变换的顶部
class augmix_transform(object):
    def __init__(self, level=0, width=3, depth=-1, alpha=1.):
        self.level = level
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def __call__(self, img):
        img = augmix(np.asarray(img) / 255)
        img = np.clip(img * 255., 0, 255).astype(np.uint8)
        img = Image.fromarray(img)
        return img


__factory = {
    'market1501': Market1501,

}


def train_collate_fn(batch):
    """
    collate_fn这个函数的输入是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    对接base中ImageDataset的返回值（应一致）
    """
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids


def train_collate_fn_with_corrupt(batch):
    ori_imgs, pids, camids, viewids, _, corid, corrupt_imgs, imgs = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    corids = torch.tensor(corid, dtype=torch.int64)
    return torch.stack(ori_imgs, dim=0), pids, camids, viewids, corids, \
           torch.stack(corrupt_imgs, dim=0), torch.stack(imgs, dim=0)


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs,
                       dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader(cfg):
    random_erasing = mixing_erasing(probability=cfg.INPUT.RE_PROB,
                                    mean=cfg.INPUT.PIXEL_MEAN,
                                    type=cfg.INPUT.ERASING_TYPE,
                                    mixing_coeff=cfg.INPUT.MIXING_COEFF)
    re_erasing = mixing_erasing(probability=cfg.INPUT.RE_PROB,
                                mean=cfg.INPUT.PIXEL_MEAN,
                                type='self',
                                mixing_coeff=cfg.INPUT.MIXING_COEFF)

    if cfg.INPUT.AUGMIX:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            augmix_transform(),
            T.ToTensor(), random_erasing, re_erasing
        ])
    else:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(), random_erasing, re_erasing
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor()
    ])

    val_with_corruption_transforms = T.Compose([
        corruption_transform(0),
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor()
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = Market1501(root=cfg.DATASETS.ROOT_DIR)

    train_set = CorruptImageDataset(dataset.train, train_transforms, cfg=cfg)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        train_loader = DataLoader(train_set,
                                  batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                  sampler=RandomIdentitySampler(
                                      dataset.train,
                                      cfg.SOLVER.IMS_PER_BATCH,
                                      cfg.DATALOADER.NUM_INSTANCE),
                                  num_workers=num_workers,
                                  collate_fn=train_collate_fn_with_corrupt)
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(train_set,
                                  batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=train_collate_fn_with_corrupt)
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.
              format(cfg.SAMPLER))

    train_loader_normal = DataLoader(train_set_normal,
                                     batch_size=cfg.TEST.IMS_PER_BATCH,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     collate_fn=val_collate_fn)

    query_set = ImageDataset(dataset.query, val_transforms)
    gallery_set = ImageDataset(dataset.gallery, val_transforms)
    corrupted_query_set = ImageDataset(dataset.query, val_with_corruption_transforms)
    corrupted_gallery_set = ImageDataset(dataset.gallery, val_with_corruption_transforms)
    # train_set_corrupt = ImageDataset(dataset.train, train_with_corruption_transforms)

    val_set = torch.utils.data.ConcatDataset([query_set, gallery_set])
    corrupted_val_set = torch.utils.data.ConcatDataset([corrupted_query_set, corrupted_gallery_set])
    corrupted_query_set = torch.utils.data.ConcatDataset([corrupted_query_set, gallery_set])
    corrupted_gallery_set = torch.utils.data.ConcatDataset([query_set, corrupted_gallery_set])

    # train_loader_corrupt = DataLoader(train_set_corrupt,
    #                                   batch_size=cfg.SOLVER.IMS_PER_BATCH,
    #                                   shuffle=True,
    #                                   num_workers=num_workers,
    #                                   collate_fn=train_collate_fn)

    val_loader = DataLoader(val_set,
                            batch_size=cfg.TEST.IMS_PER_BATCH,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=val_collate_fn)
    corrupted_val_loader = DataLoader(corrupted_val_set,
                                      batch_size=cfg.TEST.IMS_PER_BATCH,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      collate_fn=val_collate_fn)
    corrupted_query_loader = DataLoader(corrupted_query_set,
                                        batch_size=cfg.TEST.IMS_PER_BATCH,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        collate_fn=val_collate_fn)
    corrupted_gallery_loader = DataLoader(corrupted_gallery_set,
                                          batch_size=cfg.TEST.IMS_PER_BATCH,
                                          shuffle=False,
                                          num_workers=num_workers,
                                          collate_fn=val_collate_fn)
    return train_loader, train_loader_normal, val_loader, corrupted_val_loader, corrupted_query_loader, \
           corrupted_gallery_loader, len(dataset.query), num_classes, cam_num, view_num, dataset
