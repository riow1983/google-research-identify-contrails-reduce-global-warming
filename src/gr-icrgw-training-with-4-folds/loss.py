#########################################
################ loss.py ################
#########################################

import os
os.system("pip install segmentation_models_pytorch")

import segmentation_models_pytorch as smp
import yaml

with open("config.yaml", "r") as file_obj:
    config = yaml.safe_load(file_obj)

DiceLoss = smp.losses.DiceLoss(mode='binary', smooth=config["model"]["loss_smooth"])
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
alpha = 0.5
beta = 1 - alpha
TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False, alpha=alpha, beta=beta)

def criterion(y_pred, y_true):
    y_pred = y_pred.squeeze(1)
    # print(f'y_pred.shape: {y_pred.shape}; y_true.shape: {y_true.shape}')
    return 0.3 * BCELoss(y_pred, y_true)  + 0.4 * DiceLoss(y_pred, y_true) + 0.3 * TverskyLoss(y_pred, y_true)