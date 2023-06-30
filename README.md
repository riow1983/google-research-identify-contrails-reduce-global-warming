# Competition Name
google-research-identify-contrails-reduce-global-warming

# Overview
https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/overview

# Results

# Ideas

# CV Folds

# 反省点

# Top solutions

# Q&A

# W&B

# Kaggle Discussions

# Kaggle Code
- [Contrails Dataset (Ash Color)](https://www.kaggle.com/code/shashwatraman/contrails-dataset-ash-color/notebook)
- [Simple Unet Baseline (Train) - [LB 0.580]](https://www.kaggle.com/code/shashwatraman/simple-unet-baseline-train-lb-0-580/notebook)
- [Simple Unet Baseline (Infer) - [LB 0.580]](https://www.kaggle.com/code/shashwatraman/simple-unet-baseline-infer-lb-0-580/notebook)
- [[GR-ICRGW] Training with 4 folds](https://www.kaggle.com/code/egortrushin/gr-icrgw-training-with-4-folds/notebook)

# Kaggle Datasets
- [Contrails Images (Ash Color)](https://www.kaggle.com/datasets/shashwatraman/contrails-images-ash-color?select=contrails)
```
train_df.shape:  (20529, 2)
valid_df.shape:  (1856, 2)
```

# Documentations

# GitHub

# Papers
- [OpenContrails: Benchmarking Contrail Detection on GOES-16 ABI](https://arxiv.org/pdf/2304.02122.pdf)

# Tips
- W&Bのrun nameはGUI上でdeleteすると同じnameのrunが出来なくなるので要注意

# Snipets

# Diary
## 2023-06-26
コンペ参加done.

## 2023-06-28
これからやる:<br>
[[GR-ICRGW] Training with 4 folds](https://www.kaggle.com/code/egortrushin/gr-icrgw-training-with-4-folds/notebook)をtrainとinferenceに分解(.pyファイル化)し, trainでUnet以外のモデルを作る.<br>
その後, inferenceで複数モデルのアンサンブルを行う:
```python
MODEL_PATHS = "/kaggle/working/unet_models/"

all_preds = {}
for i, model_path in enumerate(glob.glob(MODEL_PATH + '*.ckpt')):
    print(model_path)
    model = LightningModule(config["model"]).load_from_checkpoint(model_path, config=config["model"])
    model.to(device)
    model.eval()

    model_preds = {}
    
    for _, data in enumerate(test_dl):
        images, image_id = data
    
        images = images.to(device)
        
        with torch.no_grad():
            predicted_mask = model(images[:, :, :, :])
        if config["model"]["image_size"] != 256:
            predicted_mask = torch.nn.functional.interpolate(predicted_mask, size=256, mode='bilinear')
        predicted_mask = torch.sigmoid(predicted_mask).cpu().detach().numpy()
                
        for img_num in range(0, images.shape[0]):
            current_mask = predicted_mask[img_num, :, :, :]
            current_image_id = image_id[img_num].item()
            model_preds[current_image_id] = current_mask
    all_preds[f"f{i}"] = model_preds


for index in submission.index.tolist():
    for i in range(len(glob.glob(MODEL_PATH + '*.ckpt'))):
        if i == 0:
            predicted_mask = all_preds[f"f{i}"][index]
        else:
            predicted_mask += all_preds[f"f{i}"][index]
    predicted_mask = predicted_mask / len(glob.glob(MODEL_PATH + '*.ckpt'))
    predicted_mask_with_threshold = np.zeros((256, 256))
    predicted_mask_with_threshold[predicted_mask[0, :, :] < THR] = 0
    predicted_mask_with_threshold[predicted_mask[0, :, :] > THR] = 1
    submission.loc[int(index), 'encoded_pixels'] = list_to_string(rle_encode(predicted_mask_with_threshold))
```
https://www.kaggle.com/code/egortrushin/gr-icrgw-training-with-4-folds?scriptVersionId=134148499&cellId=19
https://www.kaggle.com/code/egortrushin/gr-icrgw-training-with-4-folds?scriptVersionId=134148499&cellId=20
これを以下のように変更:
```python
# Ensemble
MODEL_PATHS = ["/kaggle/working/unet_models/", "/kaggle/working/hoge_models/", ...]
list_all_preds = []
for MODEL_PATH in MODEL_PATHS:
    all_preds = {}
    for i, model_path in enumerate(glob.glob(MODEL_PATH + '*.ckpt')):
        print(model_path)
        model = LightningModule(config["model"]).load_from_checkpoint(model_path, config=config["model"])
        model.to(device)
        model.eval()

        model_preds = {}
        
        for _, data in enumerate(test_dl):
            images, image_id = data
        
            images = images.to(device)
            
            with torch.no_grad():
                predicted_mask = model(images[:, :, :, :])
            if config["model"]["image_size"] != 256:
                predicted_mask = torch.nn.functional.interpolate(predicted_mask, size=256, mode='bilinear')
            predicted_mask = torch.sigmoid(predicted_mask).cpu().detach().numpy()
                    
            for img_num in range(0, images.shape[0]):
                current_mask = predicted_mask[img_num, :, :, :]
                current_image_id = image_id[img_num].item()
                model_preds[current_image_id] = current_mask
        all_preds[f"f{i}"] = model_preds
        list_all_preds.append(all_preds)


for index in submission.index.tolist():
    for j, MODEL_PATH in enumerate(MODEL_PATHS):
        all_preds = list_all_preds[j]
        for i in range(len(glob.glob(MODEL_PATH + '*.ckpt'))):
            if i == 0:
                predicted_mask_single = all_preds[f"f{i}"][index]
            else:
                predicted_mask_single += all_preds[f"f{i}"][index]
        predicted_mask_single = predicted_mask_single / len(glob.glob(MODEL_PATH + '*.ckpt'))
        if j == 0:
            predicted_mask = predicted_mask_single
        else:
            predicted_mask += predicted_mask_single
    predicted_mask = predicted_mask / len(MODEL_PATHS)

    predicted_mask_with_threshold = np.zeros((256, 256))
    predicted_mask_with_threshold[predicted_mask[0, :, :] < THR] = 0
    predicted_mask_with_threshold[predicted_mask[0, :, :] > THR] = 1
    submission.loc[int(index), 'encoded_pixels'] = list_to_string(rle_encode(predicted_mask_with_threshold))
```