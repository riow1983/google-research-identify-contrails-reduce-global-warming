#########################################
################ main.py ################
#########################################

comp_name = "google-research-identify-contrails-reduce-global-warming"
proj_name = "gr-icrgw-training-with-4-folds"

import os
os.system("pip install timm")
os.system("pip install pytorch-lightning")
os.system("pip install wandb")
os.system("pip install transformers")

import sys
sys.path.append(f"/content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}/input/pretrained-models-pytorch")
sys.path.append(f"/content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}/input/efficientnet-pytorch")
sys.path.append(f"/content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}/input/smp-github/segmentation_models.pytorch-master")
# sys.path.append(f"/content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}/input/timm-pretrained-resnest/resnest/")
sys.path.append(f"/content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}/input/timm-pretrained-efficientnet/efficientnet/")
import segmentation_models_pytorch as smp

os.system("mkdir -p /root/.cache/torch/hub/checkpoints/")
# os.system(f"cp /content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}/input/timm-pretrained-resnest/resnest/gluon_resnest26-50eb607c.pth /root/.cache/torch/hub/checkpoints/gluon_resnest26-50eb607c.pth")
os.system(f"cp /content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}/input/timm-pretrained-efficientnet/efficientnet/efficientnet_b0_ra-3dd342df.pth /root/.cache/torch/hub/checkpoints/efficientnet_b0_ra-3dd342df.pth")


import warnings

warnings.filterwarnings("ignore")

import gc
import os
import torch
import yaml
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
# from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
import wandb

from dataset import ContrailsDataset
from module import LightningModule
from _wandb import wandb_login
from utils import send_line_notification

torch.set_float32_matmul_precision("medium")

with open("config.yaml", "r") as file_obj:
    config = yaml.safe_load(file_obj)

if config["exp"] == "debug":
    config["trainer"]["max_epochs"] = 2
    config["trainer"]["min_epochs"] = 2
    # print(config)
    # raise ValueError("Stop here")
print(config)

wandb_login(config["wandb_json_path"], kaggle_env=False)

pl.seed_everything(config["seed"])

gc.enable()

contrails = os.path.join(config["data_path"], "contrails/")
train_path = os.path.join(config["data_path"], "train_df.csv")
valid_path = os.path.join(config["data_path"], "valid_df.csv")

if config["exp"] == "debug":
    train_df = pd.read_csv(train_path, nrows=1000)
    valid_df = pd.read_csv(valid_path, nrows=100)
    # train_df = pd.read_csv(train_path)
    # valid_df = pd.read_csv(valid_path)
    # print("train_df.shape: ", train_df.shape) # train_df.shape:  (20529, 2)
    # print("valid_df.shape: ", valid_df.shape) # valid_df.shape:   (1856, 2)
    # raise ValueError("Stop here")
else:
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

train_df["path"] = contrails + train_df["record_id"].astype(str) + ".npy"
valid_df["path"] = contrails + valid_df["record_id"].astype(str) + ".npy"

df = pd.concat([train_df, valid_df]).reset_index()

Fold = KFold(shuffle=True, **config["folds"])
for n, (trn_index, val_index) in enumerate(Fold.split(df)):
    df.loc[val_index, "kfold"] = int(n)
df["kfold"] = df["kfold"].astype(int)


for fold in config["train_folds"]:
    print(f"\n###### Fold {fold}")
    trn_df = df[df.kfold != fold].reset_index(drop=True)
    vld_df = df[df.kfold == fold].reset_index(drop=True)
    
    dataset_train = ContrailsDataset(trn_df, config["model"]["image_size"], train=True)
    dataset_validation = ContrailsDataset(vld_df, config["model"]["image_size"], train=False)

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=config["train_bs"],
        shuffle=True,
        num_workers=config["workers"],
    )
    data_loader_validation = DataLoader(
        dataset_validation,
        batch_size=config["valid_bs"],
        shuffle=False,
        num_workers=config["workers"],
    )

    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        monitor="val_dice",
        dirpath=config["output_dir"],
        mode="max",
        filename=f"model-f{fold}-{{val_dice:.4f}}",
        save_top_k=1,
        verbose=1,
    )

    progress_bar_callback = TQDMProgressBar(
        refresh_rate=config["progress_bar_refresh_rate"]
    )

    early_stop_callback = EarlyStopping(**config["early_stop"])

    wandb_logger = WandbLogger(project=proj_name, 
                               name=f'{config["output_dir"].split("/")[-1].split("_")[0]}_FOLD{fold}-EXP{config["exp"]}',
                               # id=f'FOLD{fold}-EXP{config["exp"]}',
                               config=config,
                               log_model="all",
                               save_dir=config["output_dir"]) # This is equivalent to wandb.init()
    
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback, progress_bar_callback],
        # logger=CSVLogger(save_dir=f'logs_f{fold}/'),
        logger=wandb_logger,
        **config["trainer"],
    )

    model = LightningModule(config["model"])

    trainer.fit(model, data_loader_train, data_loader_validation)
    if fold == len(config["train_folds"]) - 1:
        wandb_url = wandb.run.url
    # wandb_logger.finalize()
    wandb.finish()

    del (
        dataset_train,
        dataset_validation,
        data_loader_train,
        data_loader_validation,
        model,
        trainer,
        checkpoint_callback,
        progress_bar_callback,
        early_stop_callback,
        wandb_logger
    )
    torch.cuda.empty_cache()
    gc.collect()


os.system(f'cp /content/drive/MyDrive/colab_notebooks/kaggle/{comp_name}/src/{proj_name}/config.yaml {config["output_dir"]}/config.yaml')
send_line_notification(f'Training of {proj_name} EXP{config["exp"]} has been done. \nSee {wandb_url}', config["line_json_path"])