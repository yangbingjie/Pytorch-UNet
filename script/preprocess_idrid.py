# import ptvsd 
# ptvsd.enable_attach(address =('0.0.0.0',8848))
# ptvsd.wait_for_attach()

import sys
import os

sys.path.append(os.getcwd())

import numpy as np
from tqdm import tqdm
from PIL import Image

import cv2


def scaleRadius(resize_img, resize_lesion_imgs, scale):
    x = resize_img[resize_img.shape[0] // 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    # print(x, r)
    s = scale * 1 / r
    lesion_img_list = []
    for lesion_img in resize_lesion_imgs:
        if lesion_img is not None:
            lesion_img_list.append(cv2.resize(lesion_img, (0, 0), fx=s, fy=s))
        else:
            lesion_img_list.append(None)
    return cv2.resize(resize_img, (0, 0), fx=s, fy=s), lesion_img_list


def preprocessing_imgs_kaggle(file, resize_lesion_imgs, scale=348):
    # print(resize_lesion_imgs[0].dtype)

    resize_img, resize_lesion_imgs = scaleRadius(file, resize_lesion_imgs, scale)
       
    # resize_img = cv2.addWeighted(resize_img, 4,
    #                       cv2.GaussianBlur(resize_img, (0, 0), scale / 30), -4,
    #                       128)
    mask = np.zeros(resize_img.shape)
    cv2.circle(mask, (resize_img.shape[1] // 2, resize_img.shape[0] // 2),
               int(scale * 0.92), (1, 1, 1), -1, 8, 0)
    masked_img = resize_img * mask + 0 * (1 - mask)

    label_mask = np.asarray(mask[:,:,0], dtype=np.uint8)
    for i, resize_lesion in enumerate(resize_lesion_imgs):
        if resize_lesion is not None:
            if resize_lesion.shape == (550,829,4):
                resize_lesion = np.squeeze(resize_lesion[:,:,0])
            resize_lesion_imgs[i] = np.asarray(resize_lesion * label_mask, dtype=np.uint8)
        else:
            resize_lesion_imgs[i] = None
    if masked_img.shape[0] % 2 != 0:
        masked_img = np.pad(masked_img, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=(0))

    box_radius = np.min(masked_img.shape[:-1]) // 2

    clipped_img = masked_img[:,
                  masked_img.shape[1] // 2 - int(scale * 0.92):masked_img.shape[1] // 2 + int(scale * 0.92)]
    # cv2.imwrite('/home/bella/Projects/DR/g.jpg', clipped_img)

    clipped_lesion_list = []
    for resize_lesion in resize_lesion_imgs:
        if resize_lesion is not None:
            clipped_lesion = resize_lesion[:,
                    masked_img.shape[1] // 2 - int(scale * 0.92):masked_img.shape[1] // 2 + int(scale * 0.92)]
            # cv2.imwrite('/home/bella/Projects/DR/c.tif', clipped_lesion)
            clipped_lesion_list.append(clipped_lesion)
        else:
            clipped_lesion_list.append(None)


    padding = (clipped_img.shape[1] - clipped_img.shape[0]) // 2

    round_img = np.pad(clipped_img, ((padding, padding), (0, 0), (0, 0)), mode='constant', constant_values=(0))

    # round_img[0, :] = 0
    # round_img[-1, :] = 0
    # round_img[:, 0] = 0
    # round_img[:, -1] = 0

    round_lesion_list = []
    for clipped_lesion in clipped_lesion_list:
        if clipped_lesion is not None:
            padding = (clipped_lesion_list[i].shape[1] - clipped_lesion_list[i].shape[0]) // 2            
            round_lesion = np.pad(clipped_lesion, ((padding, padding), (0, 0)), mode='constant', constant_values=(0))
            if round_lesion.shape[0] != round_lesion.shape[1]:
                round_lesion = np.pad(round_lesion, ((1, 0), (0, 0)), mode='constant', constant_values=(0))

            # round_lesion[0, :] = 0
            # round_lesion[-1, :] = 0
            # round_lesion[:, 0] = 0
            # round_lesion[:, -1] = 0
            round_lesion_list.append(round_lesion)
        else:
            round_lesion_list.append(None)
    
    # cv2.imwrite('/home/bella/Projects/DR/e.tif', round_lesion_list[0].astype(np.uint8))
    # cv2.imwrite('/home/bella/Projects/DR/d.jpg', round_img)    
    # exit(1)
    return round_img, round_lesion_list


root_path = '/home/archive/Files/Lab407/Datasets/IDRiD2/'
out_path = '/home/archive/Files/Lab407/Datasets/IDRiD3/'
sub_folder = ['test', 'train']
lesion_types = ['MA', 'HE', 'EX', 'SE', 'OD']
size = (640, 640)
if not os.path.exists(out_path):
    os.mkdir(out_path)
for folder in sub_folder:
    data_path = os.path.join(root_path, folder, 'images')
    data_list = os.listdir(data_path)
    if not os.path.exists(os.path.join(out_path, folder)):
        os.mkdir(os.path.join(out_path, folder))
    for data in tqdm(data_list):
        img_path = os.path.join(data_path, data)
        lesion_imgs = []
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        if not os.path.exists(os.path.join(out_path, folder, 'images')):
            os.mkdir(os.path.join(out_path, folder, 'images'))
        if not os.path.exists(os.path.join(out_path, folder, 'label')):
            os.mkdir(os.path.join(out_path, folder, 'label'))
        for lesion in lesion_types:
            if not os.path.exists(os.path.join(out_path, folder, 'label', lesion)):
                os.mkdir(os.path.join(out_path, folder, 'label', lesion))
            path = os.path.join(root_path, folder, 'label', lesion)
            # print(os.path.join(path, data.split('.')[0] + '.tif'))
            label_img = cv2.imread(os.path.join(path, data.split('.')[0] + '.tif'), 0)
            ret,label_img = cv2.threshold(label_img,1,255,cv2.THRESH_BINARY)
            # print(os.path.join(path, data.split('.')[0] + '.tif'))
            # cv2.imwrite('/home/bella/Projects/DR/e.tif', label_img.astype(np.uint16))            
            # print(label_img.dtype)
            if label_img is not None:                
                lesion_imgs.append(label_img)
            else:                
                zero_img = np.zeros((h, w), dtype=np.uint8)   
                im = Image.fromarray(zero_img)                             
                im.save(os.path.join(path, data.split('.')[0] + '.tif').replace(root_path, out_path))
                im = cv2.imread(os.path.join(path, data.split('.')[0] + '.tif').replace(root_path, out_path), 0)
                lesion_imgs.append(im)

        new_img, new_lesion_imgs = preprocessing_imgs_kaggle(img, lesion_imgs)
        cv2.imwrite(os.path.join(data_path, data).replace(root_path, out_path), new_img)
        width, height, _ = new_img.shape
        if width != 640 or height != 640:
            print('Wrong img',width, height)
        for i in range(len(lesion_types)):
            path = os.path.join(root_path, folder, 'label', lesion_types[i])
            width, height = new_lesion_imgs[i].shape
            if width != 640 or height != 640:
                print('Wrong',width, height)
            cv2.imwrite(os.path.join(path, data.split('.')[0] + '.tif').replace(root_path, out_path), new_lesion_imgs[i].astype(np.uint8))