from torchvision.transforms import transforms
import torchvision.datasets as datasets

import matplotlib
import os

matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.ion()

from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

import torchvision.transforms as transforms
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import math



class CustomAugmentations:
    def __init__(self, output_size=256, steps=None, params=None):
        params = dict() if params is None else params
        self.params = params

        params_rotate = params.get('rotate', dict())
        max_rotation_deg = params_rotate.get('max_rotation_deg', 0)
        theta = math.radians(max_rotation_deg)
        scale_factor = abs(math.cos(theta) - math.sin(theta))
        self.resize_before_rotate = math.ceil(output_size / scale_factor / 10) * 10

        self.output_size = output_size

        # Define default augmentation sequence
        self.steps = steps or ["rotate", "crop", "jitter", "flip", "noise"]
        self.debug = {}

    def rotate(self, img_tensor, angle_rad):
        # Step A: Resize before rotation
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(self.resize_before_rotate, self.resize_before_rotate),
            mode='bilinear', align_corners=False
        ).squeeze(0)

        theta = torch.tensor([
            [math.cos(angle_rad), -math.sin(angle_rad), 0],
            [math.sin(angle_rad), math.cos(angle_rad), 0]
        ], dtype=torch.float32, device=img_tensor.device).unsqueeze(0)

        grid = F.affine_grid(
            theta, size=(1, 1, self.resize_before_rotate, self.resize_before_rotate),
            align_corners=False
        )
        img_tensor = F.grid_sample(
            img_tensor.unsqueeze(0), grid, mode='bilinear',
            padding_mode='zeros', align_corners=False
        ).squeeze(0)

        _, h, w = img_tensor.shape
        top = (h - self.output_size) // 2
        left = (w - self.output_size) // 2
        self.debug['center_crop_top_left'] = (top, left)
        img_tensor = img_tensor[:, top:top + self.output_size, left:left + self.output_size]
        return img_tensor

    def scale_and_crop(self, img_tensor, scale_factor, output_size, prob=0.5,max_shift=None, shift=None):
        if random.uniform(0, 1) > prob:
            return img_tensor

        # Step 1: scale the full image
        c, h, w = img_tensor.shape
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)

        scaled = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        _, h, w = scaled.shape
        if shift is None:
            top = random.randint(0, max_shift)
            left = random.randint(0, max_shift)
        else:
            top, left = shift, shift

        # Step 2: validate and adjust crop coordinates
        crop_h = min(output_size, h - top)
        crop_w = min(output_size, w - left)

        cropped = scaled[:, top:crop_h, left:crop_w]

        cropped = torch.nn.functional.interpolate(
            cropped.unsqueeze(0),
            size=(output_size, output_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # optional: debug info
        self.debug['scale_factor'] = scale_factor
        self.debug['crop_top'] = top
        self.debug['crop_left'] = left

        return cropped

    def random_jitter(self, img_tensor, brightness_jitter=0.05, contrast_jitter=0.05, prob=0.5):
        if random.uniform(0, 1) > prob:
            return img_tensor

        brightness = random.uniform(1-brightness_jitter, 1+brightness_jitter)
        contrast = random.uniform(1-contrast_jitter, 1+ contrast_jitter)
        self.debug['brightness'] = brightness
        self.debug['contrast'] = contrast

        img_tensor = img_tensor * brightness
        mean = img_tensor.mean()
        img_tensor = (img_tensor - mean) * contrast + mean
        return img_tensor

    def random_erasing(self, img_tensor, sl=0.02, sh=0.1, r1=0.3, prob=0.5):
        if random.uniform(0, 1) > prob:
            return img_tensor

        c, h, w = img_tensor.shape
        area = h * w

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        erase_h = int(round((target_area * aspect_ratio) ** 0.5))
        erase_w = int(round((target_area / aspect_ratio) ** 0.5))

        if erase_h < h and erase_w < w:
            top = random.randint(0, h - erase_h)
            left = random.randint(0, w - erase_w)
            mean = img_tensor.mean(dim=(1, 2), keepdim=True)  # shape [C, 1, 1]
            std = img_tensor.std(dim=(1, 2), keepdim=True)
            erase_value = torch.randn(c, erase_h, erase_w).to(img_tensor.device)  # random noise
            erase_value = erase_value * std + mean  # scale and shift to match image stats
            img_tensor[:, top:top + erase_h, left:left + erase_w] = erase_value
            self.debug['erase_top'] = top
            self.debug['erase_left'] = left
            self.debug['erase_h'] = erase_h
            self.debug['erase_w'] = erase_w

        return img_tensor

    def random_gaussian_noise(self, img_tensor, noise_a=0.01, noise_b=0.05, prob=0.5):
        if random.uniform(0, 1) > prob:
            return img_tensor

        # define standard deviation range
        stddev = random.uniform(noise_a, noise_b)
        self.debug['gaussian_noise_stddev'] = stddev

        noise = torch.randn_like(img_tensor) * stddev
        img_tensor = img_tensor + noise
        return img_tensor

    def random_flip(self, img_tensor):
        if random.random() > 0.5:
            self.debug['hflip'] = True
            img_tensor = TF.hflip(img_tensor)
        else:
            self.debug['hflip'] = False

        if random.random() > 0.5:
            self.debug['vflip'] = True
            img_tensor = TF.vflip(img_tensor)
        else:
            self.debug['vflip'] = False

        return img_tensor


class CustomTrainAugmentations(CustomAugmentations):
    def __init__(self, output_size=256,  steps=None, params=None):
        super().__init__(output_size, steps, params)
        self.debug_mode = False
        self.debug_count = 0

    def __call__(self, img_tensor):
        count = 0

        if self.debug_mode:
            save_tensor_image(img_tensor, os.path.join('debug',f'{self.debug_count:05}_{count}_debug_stretched.png'))

        if "rotate" in self.steps:
            rotate_params = self.params.get('rotate', dict())
            max_rotation_deg = rotate_params.get('max_rotation_deg', 5)
            theta = math.radians(max_rotation_deg)
            angle_rad = random.uniform(-theta, theta)
            self.debug['rotation_deg'] = math.degrees(angle_rad)
            if np.random.rand() > 0.5:
                img_tensor = self.rotate(img_tensor, angle_rad)
            else:
                img_tensor = self.rotate(img_tensor, 0.0)
            if self.debug_mode:
                count += 1
                save_tensor_image(img_tensor, os.path.join('debug',f'{self.debug_count:05}_{count}_debug_rotated.png'))

        if "crop" in self.steps:
            crop_params = self.params.get('crop', dict())
            scale_a = crop_params.get('scale_a', 0.9)
            scale_b = crop_params.get('scale_b', 1.1)
            max_shift= crop_params.get('max_shift', 10)
            scale_factor = random.uniform(scale_a, scale_b)
            img_tensor = self.scale_and_crop(img_tensor, scale_factor = scale_factor, output_size=self.output_size, max_shift=max_shift)

            if self.debug_mode:
                count += 1
                save_tensor_image(img_tensor, os.path.join('debug',f'{self.debug_count:05}_{count}_debug_cropped.png'))

        if "jitter" in self.steps:
            jitter_params = self.params.get('jitter', dict())
            brightness_jitter = jitter_params.get('brightness_jitter', 0.05)
            contrast_jitter = jitter_params.get('contrast_jitter', 0.05)

            img_tensor = self.random_jitter(img_tensor,  brightness_jitter, contrast_jitter)
            if self.debug_mode:
                count += 1
                save_tensor_image(img_tensor, os.path.join('debug',f'{self.debug_count:05}_{count}_debug_jitterred.png'))

        if "noise" in self.steps:
            noise_params = self.params.get('noise', dict())
            noise_a = noise_params.get('noise_a', 0.01)
            noise_b = noise_params.get('noise_b', 0.05)

            img_tensor = self.random_gaussian_noise(img_tensor, noise_a,noise_b)
            if self.debug_mode:
                count += 1
                save_tensor_image(img_tensor, os.path.join('debug',f'{self.debug_count:05}_{count}_debug_noise.png'))

        if "erase" in self.steps:
            erase_params = self.params.get('erase', dict())
            sl = erase_params.get('sl', 0.02)
            sh = erase_params.get('sh', 0.1)
            r1 = erase_params.get('r1', 0.3)
            img_tensor = self.random_erasing(img_tensor, sl=sl, sh=sh, r1=r1)
            if self.debug_mode:
                count += 1
                save_tensor_image(img_tensor, os.path.join('debug',f'{self.debug_count:05}_{count}_debug_erased.png'))
        if "flip" in self.steps:
            img_tensor = self.random_flip(img_tensor)
            if self.debug_mode:
                count += 1
                save_tensor_image(img_tensor, os.path.join('debug',f'{self.debug_count:05}_{count}_debug_flipped.png'))

        self.debug_count += 1
        return img_tensor


class CustomValAugmentations(CustomAugmentations):
    def __init__(self, output_size=256,  steps=None, params=None):
        super().__init__(output_size,  steps, params)

    def __call__(self, img_tensor):
        if "rotate" in self.steps:
            img_tensor = self.rotate(img_tensor, 0)

        return img_tensor

class FinalAligning:
    def __init__(self, output_size=256, interpolation_mode='bilinear',depth_duplication=1):
        self.output_size = output_size
        self.interpolation_mode = interpolation_mode
        self.depth_duplication = depth_duplication

    def __call__(self, img_tensor):
        img_tensor = img_tensor.unsqueeze(0)  # [1, 1, 224, 370] → add batch dim
        img_tensor = F.interpolate(img_tensor, size=(self.output_size, self.output_size), mode=self.interpolation_mode, align_corners=False)
        img_tensor = img_tensor.repeat(1, self.depth_duplication, 1, 1)
        img_tensor = img_tensor.squeeze(0)  # back to [1, 256, 256] if needed
        return img_tensor


class StretchLevels:
    def __init__(self, stretch_type = 'min_max', args=None):
        self.stretch_type = stretch_type
        self.args = {} if args is None else args

    def min_max(self,np_img):
        min_val = np.percentile(np_img, self.args.get('min_percentile', 1))
        max_val = np.percentile(np_img, self.args.get('max_percentile', 99))
        np_img = np.clip(np_img, min_val, max_val)
        if max_val > min_val:
            np_img = (np_img - min_val) / (max_val - min_val)
        else:
            np_img = np.zeros_like(np_img)
        return np_img

    def norm(self,np_img):
        mean = self.args.get('mean', 0.5)
        std = self.args.get('std', 0.25)
        np_img = (np_img - mean) / std
        np_img = np.array(np_img).astype(np.float32)
        return np_img

    def __call__(self, img):
        np_img = np.array(img).astype(np.float32)
        if self.stretch_type == 'min_max':
            np_img = self.min_max(np_img)
        elif self.stretch_type == 'norm':
            np_img = np_img / self.args.get('max_value', 2**16)
            np_img = self.norm(np_img)
        elif self.stretch_type == 'exp_norm':
            np_img = self.min_max(np_img)
            np_img = np.exp(np_img)
            np_img = (np_img - np_img.min()) / (np_img.max() - np_img.min())
            np_img = self.norm(np_img)

        if len(np_img.shape) == 2:
            tensor_img = torch.from_numpy(np_img).unsqueeze(0)  # [1, H, W]
        else:
            tensor_img =torch.from_numpy(np_img)

        return tensor_img

def save_tensor_image(tensor_img, file_path="./debug/debug_image.png"):
    if isinstance(tensor_img, np.ndarray):
        tensor_img = torch.from_numpy(tensor_img)

    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    # img_tensor: [1, H, W] or [C, H, W], float32, [0, 1] range
    img_np = tensor_img.squeeze(0).cpu().numpy()  # shape: [H, W] for grayscale
    plt.imsave(file_path, img_np, cmap='gray')