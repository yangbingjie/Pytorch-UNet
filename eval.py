import torch
import torch.nn.functional as F
from tqdm import tqdm
import itertools
import numpy as np
import torch
from sklearn import metrics
import torch.nn.functional as F

from dice_loss import dice_coeff


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    y_preds = []
    y_true = []
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            y_true.append(true_masks.cpu().numpy())
            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                y_preds.append(pred.cpu().numpy())
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    y_preds = np.array(y_preds)
    y_true = np.array(y_true, dtype=np.int32)
    # print(y_preds.dtype, y_preds.flatten()[:10])
    # print(y_true.dtype, y_true.flatten()[:10])
    result = {'val_loss': tot / n_val}
    result['val_roc_auc'] = metrics.roc_auc_score(y_true.flatten(), y_preds.flatten())
    precision, recall, thresholds = metrics.precision_recall_curve(y_true.flatten(), y_preds.flatten())
    result['val_pr_auc'] = metrics.auc(recall, precision)
    
    # val_roc_sum = 0
    # val_pr_sum = 0
    # for i, str in enumerate(lesion):
    #     result['val_roc_auc_' + str] = metrics.roc_auc_score(y_true[i].flatten(), y_preds[i].flatten())
    #     precision, recall, thresholds = metrics.precision_recall_curve(y_true[i].flatten(), y_preds[i].flatten())
    #     result['val_pr_auc_' + str] = metrics.auc(recall, precision)
    #     val_roc_sum += result['val_roc_auc_' + str]
    #     val_pr_sum += result['val_pr_auc_' + str]
    # result['val_roc_mean'] = val_roc_sum / (len(lesion))
    # result['val_pr_mean'] = val_pr_sum / (len(lesion))
    net.train()
    return result
