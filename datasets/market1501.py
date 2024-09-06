import random
import re
import glob
from PIL import Image
import os.path as osp
import numpy as np
import albumentations as abm
import torch
import torch.utils.data
import torchvision.transforms
from imagecorruptions.corruptions import gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, \
    motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression, \
    speckle_noise, gaussian_blur, spatter, saturate
from datasets.bases import BaseImageDataset, ImageDataset
import warnings
from config import cfg

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


class Market1501(BaseImageDataset):
    dataset_dir = 'market1501'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin

        train, num_train_corruptions = self._process_dir_with_corrupt(self.train_dir, relabel=True)
        self.num_train_corruptions = num_train_corruptions
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids, self.corruptions = \
            self.train_get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = \
            self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = \
            self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue 
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            assert 0 <= pid <= 1501  
            assert 1 <= camid <= 6
            camid -= 1  
            if relabel:
                pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset

    def _process_dir_with_corrupt(self, dir_path, relabel=True):

        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        corruption_container = set()
        for img_path in sorted(img_paths):
            corruptions_id = random.choice(range(21))
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  
            pid_container.add(pid)
            corruption_container.add(corruptions_id)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        corruptions2label = {corruption_id: label for label, corruption_id in enumerate(corruption_container)}

        num_pids = len(pid_container)
        num_corruptions = len(corruption_container)

        dataset = []
        pid2corruptions = np.zeros((num_pids, num_corruptions))
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())

            corruption_id = random.choice(range(21))
            if pid == -1:
                continue
            assert 0 <= pid <= 1501  
            assert 1 <= camid <= 6
            camid -= 1  
            if relabel:
                pid = pid2label[pid]
            corruptiones_id = corruptions2label[corruption_id]
            pid2corruptions[pid, corruptiones_id] = 1

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

            dataset.append((img_path, self.pid_begin + pid, camid, 1, corruption_id, img))
        return dataset, num_corruptions


if __name__ == '__main__':
    dataset = Market1501(root='D:/D/Re-IDdataset')
    # train_set = ImageDataset(dataset.train, transform=torchvision.transforms.ToTensor())
    # trainloader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
    # for i, data in enumerate(trainloader, 0):
    #     print(data)
