import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet
from utils.ReDirectSTD import ReDirectSTD, time_str

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

dir_img = '/home/archive/Files/Lab407/Datasets/IDRiD4/train/images/'
dir_mask = '/home/archive/Files/Lab407/Datasets/IDRiD4/train/label/'
test_dir_img = '/home/archive/Files/Lab407/Datasets/IDRiD4/test/images/'
test_dir_mask = '/home/archive/Files/Lab407/Datasets/IDRiD4/test/label/'
out_root = './runs/07_ALL_ARGU/'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
lesion = ["MA", "EX", "HE", "SE"]

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation([0, 360]),
    ])
    train = BasicDataset(lesion, dir_img, dir_mask, img_scale, transform)
    n_train = len(train)
    val = BasicDataset(lesion, test_dir_img, test_dir_mask, img_scale)
    n_val = len(val)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False, drop_last=True)
    dir_checkpoint = f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}'
    writer = SummaryWriter(log_dir=out_root, comment=dir_checkpoint)
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45])
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=5)
    # if net.n_classes > 1:
    #     criterion = nn.CrossEntropyLoss()
    # else:
    criterion = nn.BCEWithLogitsLoss()
    max_pr = 0
    for epoch in range(epochs):
        net.train()        
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                # print('\n\n\n')
                # print(masks_pred.shape, true_masks.shape)
                # print('\n\n\n')
                loss = criterion(masks_pred, true_masks.float())
                epoch_loss += loss.item()
                
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()                

                pbar.update(imgs.shape[0])
                global_step += 1

            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch)
                writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch)
            result = eval_net(lesion, net, val_loader, device)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('val_roc_mean', result['val_roc_mean'], epoch)
            writer.add_scalar('val_pr_mean', result['val_pr_mean'], epoch)
            for i, str in enumerate(lesion):
                writer.add_scalar('roc_auc/' + str,
                 result['val_roc_auc_' + str], epoch)
                writer.add_scalar('pr_auc/' + str,
                 result['val_pr_auc_' + str], epoch)
                writer.add_scalar('thres/' + str,
                 result['val_thresholds_' + str], epoch)
            writer.add_scalar('Loss/train', epoch_loss, epoch)

            # if net.n_classes > 1:
            #     logging.info('Validation cross entropy: {}'.format(result['val_loss']))
            #     writer.add_scalar('Loss/test', result['val_loss'], epoch)
            # else:
            logging.info('Validation Dice Coeff: {}'.format(result['val_loss']))
            writer.add_scalar('Dice/test', result['val_loss'], epoch)

            writer.add_images('images', imgs, epoch)
            if net.n_classes == 1:
                writer.add_images('masks/true', true_masks, epoch)
                writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, epoch)
        # scheduler.step()
        if save_cp:
            if max_pr < result['val_pr_mean']:
                max_pr = result['val_pr_mean']
                
                torch.save(net.state_dict(),
                       out_root + f'best.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')
                print(f'Checkpoint {epoch + 1} saved !')
            # try:
            #     os.mkdir(dir_checkpoint)
            #     logging.info('Created checkpoint directory')
            # except OSError:
            #     pass
    if save_cp:
        torch.save(net.state_dict(),
                       out_root + f'last.pth')
        logging.info(f'Checkpoint {epoch + 1} saved !')       

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=140,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=3,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=2e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda')
    logging.info(f'Using device {device}')

    if not os.path.exists(out_root):
        os.makedirs(out_root)
    stdout_file = os.path.join(
        out_root, 'stdout_{}.txt'.format(time_str()))
    stderr_file = os.path.join(
        out_root, 'stderr_{}.txt'.format(time_str()))
    ReDirectSTD(stdout_file, 'stdout', False)
    ReDirectSTD(stderr_file, 'stderr', False)
    

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=4, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
