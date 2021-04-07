import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    result = np.zeros(full_mask.shape, dtype=np.bool)
    for i, thres in enumerate(out_threshold):
        result[i] = full_mask[i] > out_threshold[i]
    return result


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='./runs/05_ALL/best.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input_dir', '-i', metavar='INPUT', nargs='+',
                        default='/home/archive/Files/Lab407/Datasets/IDRiD4/test/images/',
                        help='filenames of input images')
    parser.add_argument('--out_path', default='./OUT/05_ALL/')
    parser.add_argument('--lesion', default=["MA", "EX", "HE", "SE"])
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=[0.2922, 0.4769, 0.6065, 0.4198])
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = os.listdir(args.input_dir)
    for ind, item in enumerate(in_files):
        in_files[ind] = os.path.join(args.input_dir, item)
    
    out_list = []
    if not args.output:
        for f in in_files:
            path = os.path.basename(f)
            pathsplit = path.split('.')
            out_files = []
            for str in args.lesion:
                out_files.append(os.path.join(args.out_path, str, pathsplit[0] + '_out.' + pathsplit[1]))
            out_list.append(out_files)
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output
   
    return out_list


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = os.listdir(args.input_dir)
    for ind, item in enumerate(in_files):
        in_files[ind] = os.path.join(args.input_dir, item)
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    for str in args.lesion:
        if not os.path.exists(os.path.join(args.out_path, str)):
            os.mkdir(os.path.join(args.out_path, str))
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=len(args.lesion))

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]            
            for j, str in enumerate(args.lesion):
                result = mask_to_image(mask[j])
                result.save(out_files[i][j])
                logging.info("Mask saved to {}".format(out_files[i][j]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
