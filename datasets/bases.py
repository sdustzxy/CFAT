import cv2
import numpy
from PIL import ImageFile

from torch.utils.data import Dataset
import os.path as osp
import torch

from .augmentations.augmix import augmix
import torchvision.transforms as T
import numpy as np
import math

from PIL import Image
import random
import albumentations as abm
from collections import deque
from config import cfg

from imagecorruptions.corruptions import gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, \
    motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression, \
    speckle_noise, gaussian_blur, spatter, saturate

ImageFile.LOAD_TRUNCATED_IMAGES = True


# 在遇到截断的JPEG时，程序就会跳过去，读取另一张图片


def rain(image, severity=1):
    if severity == 1:
        type = 'drizzle'
    elif severity == 2 or severity == 3:
        type = 'heavy'
    elif severity == 4 or severity == 5:
        type = 'torrential'
    blur_value = 2 + severity
    bright_value = -(0.05 + 0.05 * severity)
    rain = abm.Compose([
        abm.augmentations.transforms.RandomRain(rain_type=type,
                                                blur_value=blur_value,
                                                brightness_coefficient=1,
                                                always_apply=True),
        abm.augmentations.transforms.RandomBrightness(
            limit=[bright_value, bright_value], always_apply=True)
    ])
    width, height = image.size
    if height <= 60:
        scale_factor = 65.0 / height
        new_size = (int(width * scale_factor), 65)
        image = image.resize(new_size)
    return rain(image=np.array(image))['image']


corruption_function = [
    gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur,
    motion_blur, zoom_blur, snow, frost, fog, brightness, contrast,
    elastic_transform, pixelate, jpeg_compression, speckle_noise,
    gaussian_blur, spatter, saturate, rain
]


class corruption_transform(object):
    def __init__(self, level=0, type='all'):
        self.level = level
        self.type = type

    def __call__(self, img):
        if self.level > 0 and self.level < 6:
            level_idx = self.level
        else:
            level_idx = random.choice(range(1, 6))
        if self.type == 'all':
            corrupt_func = random.choice(corruption_function)
        else:
            func_name_list = [f.__name__ for f in corruption_function]
            corrupt_idx = func_name_list.index(self.type)
            corrupt_func = corruption_function[corrupt_idx]
        c_img = corrupt_func(img.copy(), severity=level_idx)
        img = Image.fromarray(np.uint8(c_img))
        return img


def read_image(img_path):
    """一直尝试读图，直到读取成功"""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print(
                "IOError incurred when reading '{}'. Will redo. Don't worry. Just chill."
                .format(img_path))
            pass
    return img


class BaseDataset(object):
    """reid数据集的基类"""

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class Train_BaseDataset(object):
    """reid数据集的基类（带损坏id版，主要给train使用）"""

    def train_get_imagedata_info(self, data):
        pids, cams, tracks, corruptions = [], [], [], []

        for _, pid, camid, trackid, corruption, _ in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
            corruptions.append(corruption)
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views, corruptions

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset, Train_BaseDataset):

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views, corruptions = \
                self.train_get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_query_views = self.get_imagedata_info(
            query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_views = self.get_imagedata_info(
            gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(
            num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(
            num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(
            num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


def _get_pixels(per_pixel,
                rand_color,
                patch_size,
                dtype=torch.float32,
                device='cuda',
                mean=(0.5, 0.5, 0.5)):
    """
    如果出现CUDA非法访问的错误，一般是由normal_路径引起的，问题解决方法是翻转在CPU上运行
    """
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype,
                           device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class mixing_erasing(object):
    """
    随机选择一个矩形区域，对其中像素用不同的混合操作进行擦除，其中擦除的几个类型：
    normal：原始的随机擦除；
    soft：混合原始图像和随机图像的像素；
    self：混合原始图像和自身随机一部分图像像素；
    Args：
        probability：执行随机擦除的概率。
        sl：擦除区域相对于原始图像的最小比例。
        sh：擦除区域相对于原始图像的最大比例。
        r1：擦除区域最小的宽高比。
        mean：擦除值。
    """

    def __init__(self,
                 probability=0.5,
                 sl=0.02,
                 sh=0.4,
                 r1=0.3,
                 mean=(0.4914, 0.4822, 0.4465),
                 mode='pixel',
                 device='cpu',
                 type='normal',
                 mixing_coeff=[1.0, 1.0]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.rand_color = False
        self.per_pixel = False
        self.mode = mode
        if mode == 'rand':
            self.rand_color = True  # 每个区域都随机是正常不擦除的
        elif mode == 'pixel':
            self.per_pixel = True  # 每个像素随机正常
        else:
            assert not mode or mode == 'const'
        self.device = device
        self.type = type
        self.mixing_coeff = mixing_coeff

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if self.type == 'normal':
                    m = 1.0
                else:
                    m = np.float32(
                        np.random.beta(self.mixing_coeff[0],
                                       self.mixing_coeff[1]))
                if self.type == 'self':
                    x2 = random.randint(0, img.size()[1] - h)
                    y2 = random.randint(0, img.size()[2] - w)
                    img[:, x1:x1 + h,
                    y1:y1 + w] = (1 - m) * img[:, x1:x1 + h, y1:y1 +
                                                                w] + m * img[:, x2:x2 + h,
                                                                         y2:y2 + w]
                else:
                    if self.mode == 'const':
                        img[0, x1:x1 + h,
                        y1:y1 + w] = (1 - m) * img[0, x1:x1 + h, y1:y1 +
                                                                    w] + m * self.mean[0]
                        img[1, x1:x1 + h,
                        y1:y1 + w] = (1 - m) * img[1, x1:x1 + h, y1:y1 +
                                                                    w] + m * self.mean[1]
                        img[2, x1:x1 + h,
                        y1:y1 + w] = (1 - m) * img[2, x1:x1 + h, y1:y1 +
                                                                    w] + m * self.mean[2]
                    else:
                        img[:, x1:x1 + h, y1:y1 +
                                             w] = (1 - m) * img[:, x1:x1 + h,
                                                            y1:y1 + w] + m * _get_pixels(
                            self.per_pixel,
                            self.rand_color,
                            (img.size()[0], h, w),
                            dtype=img.dtype,
                            device=self.device)
                return img
        return img


class RandomPatch(object):
    """
    数据增强：随机补丁
    有一个图块池，存储从人物图像中随机提取的路径。对于每个输入图像进行随机补丁操作。两步：
    1）提取一个随机的图像块，并将该图像块存储在图像块池中；
    2）从补丁池中随机选择一个补丁并粘贴在输入(在随机位置)上模拟遮挡。
    """

    def __init__(
            self,
            prob_happen=0.5,
            pool_capacity=50000,
            min_sample_size=100,
            patch_min_area=0.01,
            patch_max_area=0.5,
            patch_min_ratio=0.1,
            prob_flip_leftright=0.5
    ):
        self.prob_happen = prob_happen

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(self.patch_min_area,
                                         self.patch_max_area) * area
            aspect_ratio = random.uniform(self.patch_min_ratio,
                                          1. / self.patch_min_ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = torch.flip(patch, dims=[2])
        return patch

    def __call__(self, img):
        _, H, W = img.size()  # 原始图像尺寸

        # 搜集新的图片块
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img[..., y1:y1 + h, x1:x1 + w]
            self.patchpool.append(new_patch)

        if len(self.patchpool) < self.min_sample_size:
            return img

        # 在随机位置粘贴随机选择的图块
        patch = random.sample(self.patchpool, 1)[0]
        _, patchH, patchW = patch.size()
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img[..., y1:y1 + patchH, x1:x1 + patchW] = patch

        return img


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, cfg=None):
        self.dataset = dataset
        self.transform = transform
        self.cfg = cfg

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.cfg is not None and self.cfg.INPUT.SELF_ID:
            random_erasing = mixing_erasing(
                probability=self.cfg.INPUT.RE_PROB,
                mean=self.cfg.INPUT.PIXEL_MEAN,
                type=self.cfg.INPUT.ERASING_TYPE,
                mixing_coeff=self.cfg.INPUT.MIXING_COEFF)
            re_erasing = mixing_erasing(
                probability=self.cfg.INPUT.RE_PROB,
                mean=self.cfg.INPUT.PIXEL_MEAN,
                type='self',
                mixing_coeff=self.cfg.INPUT.MIXING_COEFF)
            pre_transform = T.Compose([
                T.Resize(self.cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.RandomHorizontalFlip(p=self.cfg.INPUT.PROB),
                T.Pad(self.cfg.INPUT.PADDING),
                T.RandomCrop(self.cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(), random_erasing, re_erasing
            ])

            post_transform = T.Compose([
                T.ToTensor(),
            ])

            img = pre_transform(img)

            img = T.ToPILImage()(img).convert('RGB')
            if self.cfg.INPUT.AUGMIX:
                img = np.asarray(img) / 255.
                img1 = augmix(img)
                img2 = augmix(img)
                img1 = np.clip(img1 * 255., 0, 255).astype(np.uint8)
                img2 = np.clip(img2 * 255., 0, 255).astype(np.uint8)

            img = post_transform(img)
            img1 = post_transform(img1)
            img2 = post_transform(img2)

            img_list = [img, img1, img2]
            img_tuple = torch.cat(img_list, 0).half()
            return img_tuple, pid, camid, trackid, img_path.split('/')[-1]
        else:
            if self.transform is not None:
                img = self.transform(img)
            return img, pid, camid, trackid, img_path.split('/')[-1]


class CorruptImageDataset(Dataset):
    """
     训练重构，重构过程corrupt图和原图经过resize，左右翻转，pad，随机裁剪变换， 先不做擦除。
    """
    def __init__(self, dataset, transform=None, cfg=None):
        self.dataset = dataset
        self.transform = transform
        self.cfg = cfg

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid, corid, corrupt_image = self.dataset[index]
        corrupt_img = corrupt_image
        img = read_image(img_path)
        # [128, 64, 3]

        # resize，左右翻转，pad，随机裁剪变换
        pre_transform = T.Compose([
            T.Resize(self.cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=self.cfg.INPUT.PROB),
            T.Pad(self.cfg.INPUT.PADDING),
            T.RandomCrop(self.cfg.INPUT.SIZE_TRAIN),
            T.ToTensor()
        ])
        pre_ori = T.Compose([
            T.Resize(self.cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.ToTensor()
        ])

        # 用来做重构损失的原始图像
        ori_img = pre_ori(img)

        # 用来产出目标特征的增强图像
        img = pre_transform(img)

        # 随机损坏图像，加入图像增强强化模型泛化性
        corrupt_img = pre_transform(corrupt_img)

        return ori_img, pid, camid, trackid, img_path.split('/')[-1], corid, corrupt_img, img
