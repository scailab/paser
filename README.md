# PaSeR: Parsimonious Segmentation with Reinforcement Learning

Code repository for AAAI 2024 paper [Reinforcement Learning as a Parsimonious Alternative to Prediction Cascades: A Case Study on Image Segmentation](https://arxiv.org/abs/2402.11760).

**Note**: We are still in the process of testing this code for public use as well as cleaning up and uploading MNIST model training code. Expect further updates.

## Project Setup

1. Clone this repository
2. Install required packages and anaconda environment:
`conda env create -n paser --file environment.yaml`
3. Activate the environment: `conda activate paser`
3. Download the battery dataset [here](https://stevens0-my.sharepoint.com/:u:/g/personal/bsrikish_stevens_edu/EbnYfLd2cadIpfXWM6uwzmMBdBJ4_eGmfA4aK6iTJR22xw?e=L1Zu8m) and place it in the `paser/data/` directory

## PaSeR Model Training

1. Train the large unet model on the battery dataset:
`python pretrain.py +experiment=pretrain_unet_large_battery epochs={NUM_EPOCHS} checkpoint_save_interval={SAVE_INT} device={DEVICE}`
Note above you need to set the number of epochs to train, how often to save model training checkpoints, and the device id.
2. Train the medium unet model
`python pretrain.py +experiment=pretrain_unet_medium_battery epochs={NUM_EPOCHS} checkpoint_save_interval={SAVE_INT} device={DEVICE}`
3. Train the small unet model with distillation
`python pretrain.py +experiment=pretrain_unet_small_distill_battery epochs={NUM_EPOCHS} checkpoint_save_interval={SAVE_INT} device={DEVICE} local_model.checkpoint={LARGE_UNET_CHECKPOINT_PATH}`
Note above you need to set the path to the model checkpoint of the trained large unet.
4. Train the RL policy
`python pretrain_rl.py +experiment=pretrain_rl_battery epochs={NUM_EPOCHS} checkpoint_save_interval={SAVE_INT} global_model.checkpoint={SMALL_UNET_CHECKPOINT} global_model.device={SMALL_UNET_DEVICE} local_models.unet_small.checkpoint={MEDIUM_UNET_CHECKPOINT} local_models.unet_small.device={MEDIUM_UNET_DEVICE} local_models.unet.checkpoint={LARGE_UNET_CHECKPOINT} local_models.unet.device={LARGE_UNET_DEVICE} rl_model.device={RL_MODEL_DEVICE}`
5. Finetune the RL policy and segmentation models
`python finetune.py +experiment=finetune_battery epochs={NUM_EPOCHS} checkpoint_save_interval={SAVE_INT} global_model.checkpoint={SMALL_UNET_CHECKPOINT} global_model.device={SMALL_UNET_DEVICE} local_models.unet_small.checkpoint={MEDIUM_UNET_CHECKPOINT} local_models.unet_small.device={MEDIUM_UNET_DEVICE} local_models.unet.checkpoint={LARGE_UNET_CHECKPOINT} local_models.unet.device={LARGE_UNET_DEVICE} rl_model.checkpoint={RL_MODEL_CHECKPOINT} rl_model.device={RL_MODEL_DEVICE}`

