import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

import numpy as np
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
from itertools import product

import wandb

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import Optimizer
from torchmetrics.functional.classification import accuracy, multiclass_precision
from torchmetrics.functional import jaccard_index
from torchvision.utils import draw_segmentation_masks, make_grid
from torchvision.transforms.functional import convert_image_dtype

from einops import rearrange

import utils
from data.battery_dataset import BatteryDataset, Material
from models.unet import UNet, UNetSmallChannels, UNetTinyChannels, UNetTinyThreeBlock, UNetTinyTwoBlock

# logger
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="pretrain_seg_config")
def pretrain(cfg: DictConfig) -> None:
    # initiallize wandb run
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    run = wandb.init(project=cfg.wandb.project, config=wandb.config, group=cfg.wandb.group, name=cfg.wandb.name, mode=cfg.wandb.mode)
    log.info(OmegaConf.to_yaml(cfg))
    # get the hydra config output directory
    output_dir = HydraConfig.get().run.dir
    # save code snapshot
    utils.save_code_snapshot(cfg)
    # load datasets
    datasets = load_datasets(cfg)
    log.info(f'Training dataset size {len(datasets["train_dataset"])}')
    log.info(f'Validation dataset size {len(datasets["val_dataset"])}')
    log.info(f'Test dataset size {len(datasets["test_dataset"])}')
    # load noisy data subset and combine with clean data subset when training local models
    if cfg.noisy_dataset == True and cfg.model.name in ['unet', 'unet_small']:
        # load noisy dataset
        noisy_datasets = load_noisy_datasets(cfg.noisy_dataset, 0)
        joined_train_dataset = ConcatDataset([datasets['train_dataset'], noisy_datasets['train_dataset']])
        joined_val_dataset = ConcatDataset([datasets['val_dataset'], noisy_datasets['val_dataset']])
        joined_test_dataset = ConcatDataset([datasets['test_dataset'], noisy_datasets['test_dataset']])
        dataloaders = {}
        train_dataloader = DataLoader(joined_train_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, pin_memory=cfg.pin_memory)
        val_dataloader = DataLoader(joined_val_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, pin_memory=cfg.pin_memory)
        test_dataloader = DataLoader(joined_test_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, pin_memory=cfg.pin_memory)
        dataloaders = {'train_dataloader': train_dataloader, 'val_dataloader': val_dataloader, 'test_dataloader': test_dataloader}
    else:
        # create dataloader
        dataloaders = create_dataloaders(cfg, datasets)
    # create model
    model = create_model(cfg)
    # create optimizer
    optimizer = create_optimizer(cfg, model)
    local_model = None
    if cfg.model_type == 'global_distill':
        # load local model
        local_model = utils.load_seg_model(cfg.local_model)
    # train loop
    epoch = train(cfg, model, dataloaders, optimizer, output_dir, local_model)

def load_noisy_datasets(dataset_cfg: DictConfig, split_num: int):
    datasets = {}
    if dataset_cfg.name in {'battery_2_noisy'}:
        log.info('Loading battery dataset')
        train_dataset = BatteryDataset(Path(dataset_cfg.path, dataset_cfg.train_images_dir), Path(dataset_cfg.path, dataset_cfg.train_labels_dir))
        val_dataset = BatteryDataset(Path(dataset_cfg.path, dataset_cfg.val_images_dir), Path(dataset_cfg.path, dataset_cfg.val_labels_dir))
        test_dataset = BatteryDataset(Path(dataset_cfg.path, dataset_cfg.test_images_dir), Path(dataset_cfg.path, dataset_cfg.test_labels_dir))
        # split train dataset with fixed random seed
        split_dataset = random_split(train_dataset, [0.34, 0.33, 0.33], generator=torch.Generator().manual_seed(42))
    else:
        raise ValueError(f'Dataset {dataset_cfg.dataset.name} is not supported')
    # this returns just the first split of the training data, should return different splits for rl pretraining and finetuning
    return {'train_dataset': split_dataset[split_num], 'val_dataset': val_dataset, 'test_dataset': test_dataset}

def load_datasets(cfg: DictConfig):
    if cfg.dataset.name in {'battery', 'battery_2'}:
        log.info('Loading battery dataset')
        train_dataset = BatteryDataset(Path(cfg.dataset.path, cfg.dataset.train_images_dir), Path(cfg.dataset.path, cfg.dataset.train_labels_dir))
        val_dataset = BatteryDataset(Path(cfg.dataset.path, cfg.dataset.val_images_dir), Path(cfg.dataset.path, cfg.dataset.val_labels_dir))
        test_dataset = BatteryDataset(Path(cfg.dataset.path, cfg.dataset.test_images_dir), Path(cfg.dataset.path, cfg.dataset.test_labels_dir))
        # split train dataset with fixed random seed
        split_dataset = random_split(train_dataset, [0.34, 0.33, 0.33], generator=torch.Generator().manual_seed(42))
        # this returns just the first split of the training data, should return different splits for rl pretraining and finetuning
        return {'train_dataset': split_dataset[0], 'val_dataset': val_dataset, 'test_dataset': test_dataset}
    else:
        raise ValueError(f'Dataset {cfg.dataset.name} is not supported')

def create_dataloaders(cfg: DictConfig, datasets: dict):
    dataloaders = {}
    train_dataloader = DataLoader(datasets['train_dataset'], batch_size=cfg.batch_size, shuffle=cfg.shuffle, pin_memory=cfg.pin_memory)
    val_dataloader = DataLoader(datasets['val_dataset'], batch_size=cfg.batch_size, shuffle=cfg.shuffle, pin_memory=cfg.pin_memory)
    test_dataloader = DataLoader(datasets['test_dataset'], batch_size=cfg.batch_size, shuffle=cfg.shuffle, pin_memory=cfg.pin_memory)
    return {'train_dataloader': train_dataloader, 'val_dataloader': val_dataloader, 'test_dataloader': test_dataloader}

def create_model(cfg: DictConfig):
    log.info(f'Initializing {cfg.model.name} model')
    if cfg.model.name in {'unet'}:
        model = UNet(n_channels=cfg.model.n_channels, n_classes=cfg.model.n_classes, bilinear=cfg.model.bilinear)
    elif cfg.model.name in {'unet_small'}:
        model = UNetSmallChannels(n_channels=cfg.model.n_channels, n_classes=cfg.model.n_classes, bilinear=cfg.model.bilinear)
    elif cfg.model.name == 'unet_tiny':
        model = UNetTinyChannels(n_channels=cfg.model.n_channels, n_classes=cfg.model.n_classes, bilinear=cfg.model.bilinear)
    elif cfg.model.name == 'unet_tiny_three_block':
        model = UNetTinyThreeBlock(n_channels=cfg.model.n_channels, n_classes=cfg.model.n_classes, bilinear=cfg.model.bilinear)
    elif cfg.model.name in {'unet_tiny_two_block'}:
        model = UNetTinyTwoBlock(n_channels=cfg.model.n_channels, n_classes=cfg.model.n_classes, bilinear=cfg.model.bilinear)
    else:
        raise ValueError(f'Model {cfg.model.name} is not supported')
    return model

def create_optimizer(cfg: DictConfig, model: nn.Module):
    if cfg.optimizer.name == 'adam':
        log.info('Initializing Adam optimizer')
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer.name == 'adamw':
        log.info('Initializing AdamW optimizer')
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)
    else:
        raise ValueError(f'Optimizer {cfg.optimizer.name} is not supported')
    return optimizer

def train(cfg: DictConfig, model: nn.Module, dataloaders: dict, optimizer: Optimizer, output_dir: str, local_model: nn.Module = None):
    # get default device
    dev = torch.device(cfg.device)
    # send model to device
    model.to(dev)
    # cross entropy loss function with class weights
    if cfg.model.weights is not None:
        log.info(f'Creating cross entropy loss function with weights {cfg.model.weights}')
        criterion1 = nn.CrossEntropyLoss(weight=torch.tensor(cfg.model.weights, device=dev))
    else:
        log.info('Creating cross entropy loss function without weights')
        criterion1 = nn.CrossEntropyLoss()
    train_dataloader = dataloaders['train_dataloader']
    test_dataloader = dataloaders['test_dataloader']
    val_dataloader = dataloaders['val_dataloader']
    best_val_loss = 1000000.0
    # calculate interval to print/save loss every 10% of the epoch
    log_interval = np.floor(len(train_dataloader.dataset) / (cfg.batch_size * 10))
    log.info("Starting model training")
    for epoch in range(cfg.epochs):
        log.info(f'Starting epoch {epoch}')
        pbar = enumerate(tqdm(train_dataloader))
        with logging_redirect_tqdm():
            for i, batch in pbar:
                # only log every 10% of the epoch
                if i % log_interval == 0:
                    log.info(f'Starting batch {i}')
                images = batch['image'].to(dev)
                one_hot_masks = batch['one_hot_mask'].to(dev)
                true_masks = batch['true_mask'].to(dev)
                ids = batch['id']
                optimizer.zero_grad()
                if cfg.model_type == 'local':
                    # split images and labels into patches for local model
                    image_patches = rearrange(images, 'b c (h1 h) (w1 w) -> (b h1 w1) c h w', h1=cfg.num_vertical_patches, w1=cfg.num_horizontal_patches)
                    logits_patches = model(image_patches)
                    # merge patches back together into full images
                    logits = rearrange(logits_patches, '(b h1 w1) c h w -> b c (h1 h) (w1 w)', h1=cfg.num_vertical_patches, w1=cfg.num_horizontal_patches)
                    # calculate weighted cross entropy loss
                    loss = criterion1(logits, one_hot_masks)
                    loss.backward()
                    optimizer.step()
                elif cfg.model_type == 'global':
                    # shape of preds is torch.Size([32, 3, 224, 256])
                    logits = model(images)
                    # calculate weighted cross entropy loss
                    loss = criterion1(logits, one_hot_masks)
                    loss.backward()
                    optimizer.step()
                elif cfg.model_type == 'global_distill':
                    global_logits = model(images)
                    # get cross entropy loss
                    loss1 = criterion1(global_logits, one_hot_masks)
                    # split images and labels into patches for local model
                    image_patches = rearrange(images, 'b c (h1 h) (w1 w) -> (b h1 w1) c h w', h1=cfg.num_vertical_patches, w1=cfg.num_horizontal_patches)
                    # we assume the models are on the same device
                    logits_patches = local_model(image_patches)
                    # merge patches back together into full images
                    local_logits = rearrange(logits_patches, '(b h1 w1) c h w -> b c (h1 h) (w1 w)', h1=cfg.num_vertical_patches, w1=cfg.num_horizontal_patches)
                    if cfg.kd_loss == 'mse':
                        loss2 = loss_kd_mse(global_logits, local_logits)
                    elif cfg.kd_loss == 'kl':
                        loss2 = loss_kd_kl(global_logits, local_logits, cfg.temperature)
                    total_loss = (1. - cfg.alpha_kd)*loss1 + cfg.alpha_kd*loss2
                    total_loss.backward()
                    optimizer.step()
            # compute metrics for train, val and test sets and log
            if cfg.train_metrics_compute_interval != 0 and epoch % cfg.train_metrics_compute_interval == 0:
                log.info(f'Computing train metrics on epoch {epoch}')
                if cfg.model_type == 'global_distill':
                    train_loss = compute_metrics(cfg, model, train_dataloader, 'train', False, local_model)
                else:
                    train_loss = compute_metrics(cfg, model, train_dataloader, 'train', False)
            if cfg.model_type == 'global_distill':
                log.info(f'Computing validation and test metrics on epoch {epoch}')
                val_loss = compute_metrics(cfg, model, val_dataloader, 'val', False, local_model)
                test_loss = compute_metrics(cfg, model, test_dataloader, 'test', True, local_model)
            else:
                log.info('Computing validation and test metrics')
                val_loss = compute_metrics(cfg, model, val_dataloader, 'val', False)
                test_loss = compute_metrics(cfg, model, test_dataloader, 'test', True)
            # save the best validation loss model we have see so far
            if cfg.save_best_checkpoint_val and val_loss < best_val_loss:
                log.info(f'Saving best val loss model checkpoint at epoch {epoch}')
                utils.save_model_checkpoint(epoch, cfg.device, model, optimizer, output_dir, prefix=f'{cfg.model.name}', suffix='best_val', overwrite=True)
                best_val_loss = val_loss
        # save model checkpoint every checkpoint_save_interval epochs
        if epoch != 0 and cfg.checkpoint_save_interval != 0 and ((epoch + 1) % cfg.checkpoint_save_interval == 0):
            log.info(f"Saving model checkpoint for epoch {epoch}")
            utils.save_model_checkpoint(epoch, cfg.device, model, optimizer, output_dir, prefix=f'{cfg.model.name}')
    log.info("Finished global model training")
    # save final model checkpoint after training is complete
    log.info(f"Saving model checkpoint for final model")
    utils.save_model_checkpoint(epoch, cfg.device, model, optimizer, output_dir, prefix=f'{cfg.model.name}', suffix='final')
    return epoch

def loss_kd_mse(student_logits: torch.Tensor, teacher_logits: torch.Tensor):
    loss = F.mse_loss(student_logits, teacher_logits)
    return loss

def loss_kd_kl(student_logits, teacher_logits, T):
    loss = F.kl_div(F.log_softmax(student_logits/T, dim=1), F.softmax(teacher_logits/T, dim=1), reduction='batchmean') * (T * T)
    return loss

def compute_metrics(cfg: DictConfig, model: nn.Module, dataloader: DataLoader, setname: str, commit: bool, local_model: nn.Module = None):
    dev = cfg.device
    # cross entropy loss function with class weights
    if cfg.model.weights is not None:
        criterion1 = nn.CrossEntropyLoss(weight=torch.tensor(cfg.model.weights, device=dev))
    else:
        criterion1 = nn.CrossEntropyLoss()
    n_classes = cfg.model.n_classes
    with torch.no_grad():
        loss, loss1, loss2, acc, jscore = 0.0, 0.0, 0.0, 0.0, 0.0
        class_acc = torch.zeros(n_classes, device=dev)
        for i, batch in enumerate(dataloader):
            images = batch['image'].to(dev)
            one_hot_masks = batch['one_hot_mask'].to(dev)
            true_masks = batch['true_mask'].to(dev)
            ids = batch['id']
            if cfg.model_type == 'local':
                # split images and labels into patches for local model
                image_patches = rearrange(images, 'b c (h1 h) (w1 w) -> (b h1 w1) c h w', h1=cfg.num_vertical_patches, w1=cfg.num_horizontal_patches)
                logits_patches = model(image_patches)
                # join patch predictions back together into full image
                logits = rearrange(logits_patches, '(b h1 w1) c h w -> b c (h1 h) (w1 w)', h1=cfg.num_vertical_patches, w1=cfg.num_horizontal_patches)
                # compute per batch loss
                loss += criterion1(logits, one_hot_masks)
                preds_patches = logits_patches.argmax(dim=1)
                # merge the patch predictions into the full image prediction
                preds = rearrange(preds_patches, '(b h1 w1) h w -> b (h1 h) (w1 w)', h1=cfg.num_vertical_patches, w1=cfg.num_horizontal_patches)
                # compute per image metrics
                for b in range(preds.shape[0]):
                    acc += accuracy(preds[b], true_masks[b], task='multiclass', num_classes=n_classes)
                    class_acc += accuracy(preds[b], true_masks[b], task='multiclass', average='none', num_classes=n_classes)
                    jscore += jaccard_index(preds[b], true_masks[b], task='multiclass', num_classes=n_classes)
            elif cfg.model_type == 'global':
                logits = model(images)
                # compute per batch loss
                loss += criterion1(logits, one_hot_masks)
                preds = logits.argmax(dim=1)
                for b in range(preds.shape[0]):
                    acc += accuracy(preds[b], true_masks[b], task='multiclass', num_classes=n_classes)
                    class_acc += accuracy(preds[b], true_masks[b], task='multiclass', average='none', num_classes=n_classes)
                    jscore += jaccard_index(preds[b], true_masks[b], task='multiclass', num_classes=n_classes)
            elif cfg.model_type == 'global_distill':
                global_logits = model(images)
                loss1 += criterion1(global_logits, one_hot_masks)
                global_preds = global_logits.argmax(dim=1)
                # split images and labels into patches for local model
                image_patches = rearrange(images, 'b c (h1 h) (w1 w) -> (b h1 w1) c h w', h1=cfg.num_vertical_patches, w1=cfg.num_horizontal_patches)
                # we assume the models are on the same device
                logits_patches = local_model(image_patches)
                # merge patches back together into full images
                local_logits = rearrange(logits_patches, '(b h1 w1) c h w -> b c (h1 h) (w1 w)', h1=cfg.num_vertical_patches, w1=cfg.num_horizontal_patches)
                if cfg.kd_loss == 'mse':
                    loss2 += loss_kd_mse(global_logits, local_logits)
                elif cfg.kd_loss == 'kl':
                    loss2 += loss_kd_kl(global_logits, local_logits, cfg.temperature)
                loss = (1. - cfg.alpha_kd)*loss1 + cfg.alpha_kd*loss2
                for b in range(global_preds.shape[0]):
                    acc += accuracy(global_preds[b], true_masks[b], task='multiclass', num_classes=n_classes)
                    class_acc += accuracy(global_preds[b], true_masks[b], task='multiclass', average='none', num_classes=n_classes)
                    jscore += jaccard_index(global_preds[b], true_masks[b], task='multiclass', num_classes=n_classes)
        loss = loss/len(dataloader)
        acc = acc/len(dataloader.dataset)
        jscore = jscore/len(dataloader.dataset)
        class_acc = class_acc/len(dataloader.dataset)
        if cfg.model_type == 'global_distill':
            loss1 = loss1/len(dataloader)
            loss2 = loss2/len(dataloader)
            wandb.log({f'loss_ce_batch_{setname}': loss1,
                       f'loss_kd_batch_{setname}': loss2}, commit=False)
        wandb.log({f'loss_batch_{setname}': loss,
            f'acc_macro_{setname}': acc,
            f'class_acc_pore_{setname}': class_acc[0],
            f'class_acc_carbon_{setname}': class_acc[1],
            f'class_acc_nickel_{setname}': class_acc[2],
            f'jaccard_{setname}': jscore}, commit=commit)
    return loss

def get_img_region(image, mappings: list, r: int):
    if len(image.shape) == 3:
        img_region = image[:, mappings[r][1]:mappings[r][3], mappings[r][0]:mappings[r][2]]
    elif len(image.shape) == 2:
        img_region = image[mappings[r][1]:mappings[r][3], mappings[r][0]:mappings[r][2]]

    return img_region

def get_img_mappings(cfg: DictConfig, image: torch.Tensor):
    mappings = []
    w = image.shape[2]
    h = image.shape[1]
    # calculate patch width and height
    w_stride = int(w / cfg.num_horizontal_patches)
    h_stride = int(h / cfg.num_vertical_patches)
    # make a cartesian product grid which we will use to crop the image
    grid = product(range(0, h-h%h_stride, h_stride), range(0, w-w%w_stride, w_stride))
    for patch_num, (i, j) in enumerate(list(grid)):
        mappings.append((j, i, j + w_stride, i + h_stride))
    return mappings

if __name__ == "__main__":
    pretrain()

