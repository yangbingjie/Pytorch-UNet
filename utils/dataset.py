from os.path import splitext
from os import listdir
import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import random


class BasicDataset(Dataset):
    def __init__(self, lesion, imgs_dir, masks_dir, scale=1, mask_suffix='', transform=None):
        self.lesion = lesion
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.transform = transform
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)

        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]        
        
        img_file = glob(self.imgs_dir + idx + '.*')
        # print('\n\n\n')
        # print(mask_file)
        # print('\n\n\n')
        # assert len(mask_file) == 1, \
        #     f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
               
        img = Image.open(img_file[0])
        img = self.preprocess(img, self.scale)

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.transform is not None:
            img = self.transform(img)

        mask_list = []

        for i, str in enumerate(self.lesion):
            mask = Image.open(os.path.join(self.masks_dir, str, os.path.basename(img_file[0]).split('.')[0] + '.tif'))
            mask = self.preprocess(mask, self.scale)

            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            if self.transform is not None:
                mask = self.transform(mask)

            mask_list.append(mask)
        
        mask_arr = np.array(mask_list).squeeze()
        # print('\n\n\n') 
        # print(mask_arr.shape)
        # print('\n\n\n')
        # exit(1)
        assert img.shape[1:] == mask.shape[1:], \
            f'Image and mask {idx} should be the same size, but are {img.shape} and {mask.shape}'

        
        
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask_arr).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
