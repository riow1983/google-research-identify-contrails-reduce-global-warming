# Competition Name
google-research-identify-contrails-reduce-global-warming

# Overview
https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/overview

# Results

# Ideas
- Increase image size (384 -> 512)
- Increase folds (4 -> 5 or 10)
- Threshold optimization: `np.arange(0.01, 0.51, 0.01)`

# CV Folds
- [Kfold(train + valid)](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/416395#2295731)
- [Kfold(train) + valid](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/413153#2315071)
- Kfold(valid) + train

# 反省点

# Top solutions

# Q&A

# W&B

# Kaggle Discussions
- [Increasing image size doesn't work for me on LB](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/420079)

# Kaggle Code
- [Contrails Dataset (Ash Color)](https://www.kaggle.com/code/shashwatraman/contrails-dataset-ash-color/notebook)
- [Simple Unet Baseline (Train) - [LB 0.580]](https://www.kaggle.com/code/shashwatraman/simple-unet-baseline-train-lb-0-580/notebook)
- [Simple Unet Baseline (Infer) - [LB 0.580]](https://www.kaggle.com/code/shashwatraman/simple-unet-baseline-infer-lb-0-580/notebook)
- [[GR-ICRGW] Training with 4 folds](https://www.kaggle.com/code/egortrushin/gr-icrgw-training-with-4-folds/notebook)
- [[GR-ICRGW] Training with 4 folds Inference](https://www.kaggle.com/code/riow1983/gr-icrgw-training-with-4-folds-inference)

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
- W&Bのrunはdeleteすると, 同じrun idのrunが実行出来なくなるので要注意 (逆に同一run idのrunが存在している場合はresumeされる)

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

## 2023-07-10
segmentation models pytorchにtimmのencoderを使う方法を説明した[公式ページ](https://smp.readthedocs.io/en/latest/encoders_timm.html)発見. そしてここにどのencoderがdilated modeに対応しているのか書いてあった. 長いことかかってしまった.<br>
ToDo: [inference code](https://www.kaggle.com/code/riow1983/gr-icrgw-training-with-4-folds-inference)のinputにhttps://www.kaggle.com/datasets/ar90ngas/timm-pretrained-efficientnet を加える.

## 2023-07-11
ecoderに`timm-efficientnet-b0`を使い, DeepLabV3PlusのEXP 0を実行.<br><br>
[Discussion](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/415176)にて, [Simple Unet Baseline (Train) - [LB 0.580]](https://www.kaggle.com/code/shashwatraman/simple-unet-baseline-train-lb-0-580/notebook)が改良された旨認知した.
> Using EfficientNet B3, adding some augmentations and training for 30 epochs, the new version of the baseline gets a lb score of 0.628.

また[Discussion](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/413767)にて, UnetにBatchNormを加えると良いというtipsが公開されていた.
> I don't remember since i use smp implementation now which has a parameter for enabling BatchNorm. I recall it did improve the CV but also sped up training by lowering the number of epochs needed.

しかしこれは`smp.Unet()`ではデフォルトでBatchNormが挿入されている.

## 2023-07-14
`efficientnetB0 + deeplabv3plus`は最後の4fold目で意味不明なエラー. これはやめて`efficientnetB4 + FPN`を試すことにした. こちらepoch 0はtrain+validで2時間近くかかったが, epoch 1以降は5分で終わる.<br><br>
[Discussion](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/409624) によると, マスクがブランクのものが57のrecord_idで見つかったとのこと. ただし、これらが学習データから除外されるべきかというとそうでは無いと思う. 飛行機雲が存在しない画像もあるからだ.<br><br>
WCEとDICEの組み合わせは上手くいかないとのこと: https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/412554#2279642<br><br>
メタデータ (record_idごとの時空間情報など)はtestデータでは利用できないとのこと: https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/419562#2318687<br>
とするとCV作成時に利用するくらいか. なお[メタデータ(jsonファイル)を読み込むときはrecord_idのデータ型を`str`型に指定してやる必要がある](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/418016#2311814)とのこと.<br><br>
ToDo: train.metadata.geolocationとvalid.metadata.geolocationの分布の違いを確認

## 2023-07-16
`efficientnetB4 + FPN`は遅すぎるため, `efficientnetB4 + deeplabv3plus`に挑戦. しかし`efficientnetB0 + deeplabv3plus`では最後の4fold目で意味不明なエラーだった. これが再発しないことを期待.

## 2023-07-19
`efficientnetB4 + deeplabv3plus`は0 fold目で原因不明の停止が発生. もしかするとtimm系のencoderが原因かも分からないので, encoderにtimmを使う方針を放棄し, segmentation models pytorch内蔵のencoderに切り替える. `DeepLabV3+ w/ resnet101 (smp内蔵)`で再挑戦.

## 2023-07-21
`DeepLabV3+ w/ resnet101 (smp内蔵)`は3 fold目で中絶していた. 改めて3 fold目だけ実行するか.<br>
Ensembling of `Unet w/ timm-resnest26d` (LB: 0.646) and `DeepLabV3+ w/ resnet101` ended up with LB 0.631. The average CVs of `DeepLabV3+ w/ resnet101` was around 0.60, whereas `Unet w/ timm-resnest26d`'s was 0.66. Have to seek a more accurate model.

## 2023-07-24
Started to run `Unet++ w/ resnet101`. During this, reading discussions in order to decide which direction I should dig next:
- [3D or 2.5D models does not help](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/419816)
Other topics I could deep further:
- Loss modification
- EDA revisit
- CV modification
- Different train datasets

## 2023-07-27
`Unet++ w/ resnet101` has been finished and uploaded to Kaggle. But the result of ensembling with `Unet w/ timm-resnest26d` was not better than the single `Unet w/ timm-resnest26d`.