import random
import glob
import re
from PIL import Image
import os.path as osp
import numpy as np
import albumentations as abm

from imagecorruptions.corruptions import gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, \
    motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression, \
    speckle_noise, gaussian_blur, spatter, saturate
from .bases import BaseImageDataset
import warnings

warnings.filterwarnings('ignore')


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


def clean(image, severity=1):

    severity = severity
    image = image
    return image


corruption_function = [
    clean, gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur,
    motion_blur, zoom_blur, snow, frost, fog, brightness, contrast,
    elastic_transform, pixelate, jpeg_compression, speckle_noise,
    gaussian_blur, spatter, saturate, rain
]


# MSMT17_V2
class MSMT17(BaseImageDataset):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = 'MSMT17'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(MSMT17, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'mask_train_v2')
        self.test_dir = osp.join(self.dataset_dir, 'mask_test_v2')
        self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')

        self._check_before_run()
        train, num_train_corruptions = self._process_dir_with_corrupt(self.train_dir, self.list_train_path)
        val, num_val_corruptions = self._process_dir_with_corrupt(self.train_dir, self.list_val_path)
        train += val
        self.num_train_corruptions = num_train_corruptions
        query = self._process_dir(self.test_dir, self.list_query_path)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path)
        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids, self.corruptions = \
            self.train_get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2])
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path,  self.pid_begin +pid, camid-1, 1))
            pid_container.add(pid)
            cam_container.add(camid)
        print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset

    def _process_dir_with_corrupt(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        corruption_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2])
            # cor_pro = random.random()
            # if cor_pro > 0.1:
            #     pro = random.random()
            #     if pro > 0.5:
            #         corruption_id = 0
            #     else:
            #         corruption_id = 11
            # else:
            corruption_id = random.choice(range(1, 21))
            img_path = osp.join(dir_path, img_path)
            # level_pro = random.random()
            # if level_pro > 0.9:
            #     level_idx = 1
            # else:
            level_idx = random.choice(range(1, 6))
            image = Image.open(img_path).convert('RGB')
            if corruption_id == 1:
                corrupt_func = gaussian_noise
            elif corruption_id == 2:
                corrupt_func = shot_noise
            elif corruption_id == 3:
                corrupt_func = impulse_noise
            elif corruption_id == 4:
                corrupt_func = defocus_blur
            elif corruption_id == 5:
                corrupt_func = glass_blur
            elif corruption_id == 6:
                corrupt_func = motion_blur
            elif corruption_id == 7:
                corrupt_func = zoom_blur
            elif corruption_id == 8:
                corrupt_func = snow
            elif corruption_id == 9:
                corrupt_func = frost
            elif corruption_id == 10:
                corrupt_func = fog
            elif corruption_id == 11:
                corrupt_func = brightness
            elif corruption_id == 12:
                corrupt_func = contrast
            elif corruption_id == 13:
                corrupt_func = elastic_transform
            elif corruption_id == 14:
                corrupt_func = pixelate
            elif corruption_id == 15:
                corrupt_func = jpeg_compression
            elif corruption_id == 16:
                corrupt_func = speckle_noise
            elif corruption_id == 17:
                corrupt_func = gaussian_blur
            elif corruption_id == 18:
                corrupt_func = spatter
            elif corruption_id == 19:
                corrupt_func = saturate
            elif corruption_id == 20:
                corrupt_func = rain
            else:
                corrupt_func = clean
            c_img = corrupt_func(image.copy(), severity=level_idx)
            img = Image.fromarray(np.uint8(c_img))
            dataset.append((img_path,  self.pid_begin +pid, camid-1, 1, corruption_id, img))
            pid_container.add(pid)
            cam_container.add(camid)
        print(cam_container, 'cam_container')
        num_corruptions = len(corruption_container)
        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset, num_corruptions


# source: AGW
class MSMT17_V1(BaseImageDataset):
    """
    MSMT17
    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.
    URL: http://www.pkuvmc.com/publications/msmt17.html
    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = 'msmt17'

    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(MSMT17, self).__init__()
        self.dataset_dir = '/data/wzq/msmt17'
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            # if pid == -1: continue  # junk images are just ignored
            # assert 0 <= pid <= 1501  # pid == 0 means background
            # assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid,1))

        return dataset