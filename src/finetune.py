import logging
import pickle
import glob
import tarfile
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

import wandb

import numpy as np
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
from itertools import product

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
#from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from torch.distributions.categorical import Categorical
from torchmetrics.functional.classification import accuracy
from torchmetrics.functional import jaccard_index
from torchvision.utils import draw_segmentation_masks, make_grid
from torchvision.transforms.functional import convert_image_dtype, crop
from einops import rearrange

import utils
from data.battery_dataset import BatteryDataset, Material
from models.unet import UNet, UNetSmallChannels, UNetTinyChannels, UNetTinyThreeBlock, UNetTinyTwoBlock
from models.resnet import ResNet, BasicBlock

# logger
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="finetune_config")
def finetune(cfg: DictConfig) -> None:
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
    # create dataloader
    dataloaders = create_dataloaders(cfg, datasets)
    # load global model
    global_model = utils.load_seg_model(cfg.global_model, dropout=True)
    # load local models
    local_models, local_opts = load_local_seg_models(cfg.local_models, cfg.optimizer)
    # load rl model
    rl_model, rl_optimizer = utils.load_model_train(cfg.rl_model, cfg.optimizer)
    num_params = utils.get_num_params(cfg, global_model, local_models, rl_model, trainable=False)
    log.info(f'Number of parameters per model: {num_params}')
    train_alternating(cfg, global_model, local_models, local_opts, rl_model, rl_optimizer, dataloaders, output_dir, None)

def train_alternating(cfg: DictConfig, global_model: nn.Module, local_models: dict, local_optimizers: dict, rl_model: nn.Module, rl_optimizer: Optimizer, dataloaders: dict, output_dir: str, rl_scheduler=None):
    log.info('Starting finetuning')
    num_classes = cfg.global_model.n_classes
    num_models = len(local_models) + 1
    model_names = ['global', 'local_1', 'local_2']
    # setup which devices we should use
    global_dev = cfg.global_model.device
    # init dataloaders
    train_dataloader = dataloaders['train_dataloader']
    test_dataloader = dataloaders['test_dataloader']
    val_dataloader = dataloaders['val_dataloader']
    # create cross entropy loss functions dict with class weights for each local model
    cross_ent_loss_dict = {}
    for name, model in local_models.items():
        if cfg.local_models[name].weights is not None:
            cross_ent_loss_dict[name] = nn.CrossEntropyLoss(weight=torch.tensor(cfg.local_models[name].weights, device=cfg.local_models[name].device))
        else:
            cross_ent_loss_dict[name] = nn.CrossEntropyLoss()
    image = train_dataloader.dataset[0]['image']
    # get image region mappings which are tuples of the form (w1, h1, w2, h2)
    mappings = get_img_mappings(cfg, image)
    num_patches = cfg.num_horizontal_patches * cfg.num_vertical_patches
    global_cost = (16571.0/(16571+1080595+17275459)) * num_patches
    global_cost_reg = (16571.0/(16571+1080595+17275459))
    # create alpha param linspace/schedule
    alphas = np.linspace(cfg.alpha_start, cfg.alpha_end, cfg.epochs)
    num_explore, num_exploit = 0, 0
    rl_logits_all = []
    global_per = 0.0
    # calculate interval to print/save loss every 10% of the epoch
    log_interval = np.floor(len(train_dataloader.dataset) / (cfg.batch_size * 10))
    with logging_redirect_tqdm():
        for epoch in range(cfg.epochs):
            log.info(f'Starting epoch {epoch}')
            # create dictionaries of lists to store data for histograms
            region_nums_dict, region_ents_dict = {}, {}
            region_pore_dict, region_carbon_dict, region_nickel_dict = {}, {}, {}
            for m in range(num_models):
                region_nums_dict[m] = []
                region_ents_dict[m] = []
                region_pore_dict[m] = []
                region_carbon_dict[m] = []
                region_nickel_dict[m] = []
            for i, batch in enumerate(tqdm(train_dataloader)):
                # put local models into train mode
                for model in local_models.values():
                    model.train()
                rl_model.eval()
                # load images and labels
                images = batch['image'].to(global_dev)
                one_hot_masks = batch['one_hot_mask']
                true_masks = batch['true_mask']
                ids = batch['id']
                # run global model on images and compute monte carlo entropy
                with torch.no_grad():
                    # images is B x C x H x W
                    global_logits = torch.zeros((cfg.num_unc_samples, images.shape[0], cfg.global_model.n_classes, images.shape[2], images.shape[3]), device=global_dev)
                    global_probs = torch.zeros((cfg.num_unc_samples, images.shape[0], cfg.global_model.n_classes, images.shape[2], images.shape[3]), device=global_dev)
                    # compute model predictions num_unc_samples times
                    for j in range(cfg.num_unc_samples):
                        global_logits[j] = global_model(images)
                        global_probs[j] += F.softmax(global_logits[j], dim=1)
                    mean_global_probs = torch.mean(global_probs, 0)
                    global_preds = mean_global_probs.argmax(dim=1)
                    # compute entropy map
                    global_ents = Categorical(probs=torch.permute(mean_global_probs, (0, 2, 3, 1))).entropy()
                # create rl input and move to correct device
                rl_input = torch.stack((global_preds, global_ents), dim=1)
                rl_input = rl_input.to(cfg.rl_model.device)
                # run the rl policy on the segmentation and entropy maps
                rl_logits = rl_model(rl_input)
                # save rl_logits values to list for histogram
                rl_logits_all.extend(rl_logits.detach().flatten().tolist())
                # reshape the rl_logits to be 32 x 16 x 3
                rl_logits = rl_logits.reshape((rl_logits.shape[0], num_patches, num_models))
                # log min and max rl logits for test
                wandb.log({'rl_logits_mean': rl_logits.mean(),
                           'rl_logits_max': rl_logits.max(),
                           'rl_logits_min': rl_logits.min()}, commit=False)
                rl_probs = F.softmax(rl_logits, dim=2)
                # use categorical probs to sample region-model pairs
                rl_dist = Categorical(probs=rl_probs)
                # alpha based explore/exploit
                if np.random.rand() < alphas[epoch]:
                    # exploit using rl distribution
                    rl_sample = rl_dist.sample()
                    num_exploit += 1
                else:
                    # explore with uniform probability for all patches and all models
                    uniform_dist = Categorical(logits=torch.ones((rl_logits.shape[0], num_patches, num_models)))
                    rl_sample = uniform_dist.sample().to(cfg.rl_model.device)
                    num_explore += 1
                # build inputs for each segmentation model and get one hot masks
                image_patches_dict = {}
                one_hot_masks_patches_dict = {}
                image_patches_metadata = {} # map from model number to list of image number, patch number tuples
                for m in range(num_models):
                    image_patches_dict[m] = []
                    one_hot_masks_patches_dict[m] = []
                    image_patches_metadata[m] = []
                for b in range(images.shape[0]):
                    for r, m in enumerate(rl_sample[b]):
                        img_region = get_img_region(images[b], mappings, r).unsqueeze(0)
                        one_hot_mask_region = get_img_region(one_hot_masks[b], mappings, r).unsqueeze(0)
                        image_patches_dict[m.item()].append(img_region)
                        one_hot_masks_patches_dict[m.item()].append(one_hot_mask_region)
                        image_patches_metadata[m.item()].append((b, r))
                        # log the data for histogram of region numbers sent to each model
                        if i == 0:
                            mc = m.item()
                            region_nums_dict[mc].append(r)
                            # log mean entropy for each region
                            region_ents_dict[mc].append(get_img_region(global_ents[b], mappings, r).mean().item())
                            # log number of pore, carbon, and nickel pixels
                            region_pore_dict[mc].append(torch.sum(torch.where(get_img_region(true_masks[b], mappings, r) == 0, 1.0, 0.0)).item())
                            region_carbon_dict[mc].append(torch.sum(torch.where(get_img_region(true_masks[b], mappings, r) == 1, 1.0, 0.0)).item())
                            region_nickel_dict[mc].append(torch.sum(torch.where(get_img_region(true_masks[b], mappings, r) == 2, 1.0, 0.0)).item())
                if i == 0:
                    # only log mean entropy per model if regions are sent to that model
                    for j in range(num_models):
                        if len(region_ents_dict[j]) > 0:
                            wandb.log({f'mean_ent_{model_names[j]}': np.mean(region_ents_dict[j])}, commit=False)
                image_patches, one_hot_masks_patches = [], []
                for m in range(num_models):
                    if len(image_patches_dict[m]) > 0:
                        image_patches.append(torch.vstack(image_patches_dict[m]))
                        one_hot_masks_patches.append(torch.vstack(one_hot_masks_patches_dict[m]))
                    elif len(image_patches_dict[m]) == 0:
                        image_patches.append(torch.empty(0))
                        one_hot_masks_patches.append(torch.empty(0))
                # run local models on their respective patches and update them
                for m in range(num_models):
                    if m > 0 and len(image_patches_dict[m]) > 0:
                        model = list(local_models.values())[m-1]
                        optimizer = list(local_optimizers.values())[m-1]
                        criterion = list(cross_ent_loss_dict.values())[m-1]
                        # move patches to correct device
                        dev = list(OmegaConf.to_container(cfg.local_models).values())[m-1]['device']
                        image_patches[m] = image_patches[m].to(dev)
                        one_hot_masks_patches[m] = one_hot_masks_patches[m].to(dev)
                        logits = model(image_patches[m])
                        loss = criterion(logits, one_hot_masks_patches[m])
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                # put local models in eval mode and rl model in train mode
                for model in local_models.values():
                    model.eval()
                rl_model.train()
                with torch.no_grad():
                    # get global_logits for just one batch sample
                    global_logits = global_logits[0]
                    global_probs = F.softmax(global_logits, dim=1)
                    global_preds = global_probs.argmax(dim=1)
                    # compute jaccard score with global model only on all images and regions in batch
                    global_img_jscores, global_reg_jscores = compute_jaccard(cfg, global_preds, true_masks, mappings)
                    # B x C x H x W
                    rl_img_probs = torch.zeros(global_probs.shape)
                    for m in range(num_models):
                        if m > 0 and len(image_patches_dict[m]) > 0:
                            model = list(local_models.values())[m-1]
                            # move region to correct device
                            dev = list(OmegaConf.to_container(cfg.local_models).values())[m-1]['device']
                            image_patches[m] = image_patches[m].to(dev)
                            logits = model(image_patches[m])
                            for p, (b, r) in enumerate(image_patches_metadata[m]):
                                rl_img_probs[b, :, mappings[r][1]:mappings[r][3], mappings[r][0]:mappings[r][2]] = F.softmax(logits[p], dim=0)
                        elif m == 0 and len(image_patches_dict[m]) > 0:
                            for b, r in image_patches_metadata[m]:
                                rl_img_probs[b, :, mappings[r][1]:mappings[r][3], mappings[r][0]:mappings[r][2]] = global_probs[b, :, mappings[r][1]:mappings[r][3], mappings[r][0]:mappings[r][2]]
                # get hard predictions for all images in batch
                rl_img_preds = rl_img_probs.argmax(dim=1)
                # compute rl policy jaccard score for each region
                rl_jscores, rl_reg_jscores = compute_jaccard(cfg, rl_img_preds, true_masks, mappings)
                # compute reward with jaccard per region
                R = (rl_reg_jscores - global_reg_jscores)
                wandb.log({'reward_no_cost_reg': R.mean(),
                        'reward_no_cost_scaled': (1-cfg.lam) * R.mean()}, commit=False)
                # get total image cost and per region cost
                c, c_reg = compute_cost(cfg, rl_sample)
                # using global model cost of 0
                cost = c_reg
                wandb.log({'cost_reg': cost.mean(), 
                    'cost_scaled': cfg.lam * cost.mean()}, commit=False)
                # get percent usage for each model
                global_per = torch.mean(torch.where(rl_sample == 0, 1.0, 0.0))
                local1_per = torch.mean(torch.where(rl_sample == 1, 1.0, 0.0))
                local2_per = torch.mean(torch.where(rl_sample == 2, 1.0, 0.0))
                # compute per region reward by subtracting cost per region
                R = ((1-cfg.lam)*R) - (cfg.lam*cost)
                # use policy gradients to update rl model parameters with optimizer
                R = R.to(cfg.rl_model.device)
                # compute loss per region
                loss = (-rl_dist.log_prob(rl_sample)) * R
                loss = loss.mean()
                rl_optimizer.zero_grad()
                loss.backward()
                rl_optimizer.step()
                wandb.log({'global_img_jaccard_train': global_img_jscores.mean(),
                    'rl_img_jaccard_train': rl_jscores.mean(),
                    'reward': R.mean(),
                    'percent_patches_global': global_per,
                    'percent_patches_local_1': local1_per,
                    'percent_patches_local_2': local2_per
                    }, commit=False)
            # step the scheduler if it exists
            if rl_scheduler is not None:
                log.info(f'Stepping rl scheduler, new learning rate is {rl_optimizer.param_groups[0]["lr"]}')
                rl_scheduler.step()
            if cfg.compute_metrics_interval != 0 and epoch % cfg.compute_metrics_interval == 0:
                log.info(f'Computing train, val, and test metrics')
                # compute train, validation and test metrics according to configuration
                compute_metrics(cfg, train_dataloader, rl_model, global_model, local_models, 'train', False)
                compute_metrics(cfg, val_dataloader, rl_model, global_model, local_models, 'val', False)
                compute_metrics(cfg, test_dataloader, rl_model, global_model, local_models, 'test', False)
            if epoch != 0 and cfg.checkpoint_save_interval != 0 and ((epoch + 1) % cfg.checkpoint_save_interval == 0):
                log.info(f"Saving model checkpoints for epoch {epoch}")
                # save rl model checkpoint
                utils.save_model_checkpoint(epoch, cfg.rl_model.device, rl_model, rl_optimizer, output_dir, prefix=cfg.rl_model.name)
                for name, model, optimizer in zip(local_models.keys(), local_models.values(), local_optimizers.values()):
                    # save local model checkpoint
                    utils.save_model_checkpoint(epoch, cfg.local_models[name].device, model, optimizer, output_dir, prefix=name)
            wandb.log({'epoch': epoch}, commit=True)

def compute_metrics(cfg: DictConfig, dataloader: DataLoader, rl_model: nn.Module, global_model: nn.Module, local_models: dict, setname: str, commit=False):
    log.info('Computing metrics')
    n_classes = cfg.global_model.n_classes
    num_patches = cfg.num_horizontal_patches * cfg.num_vertical_patches
    num_models = len(local_models) + 1
    global_dev = cfg.global_model.device
    image = dataloader.dataset[0]['image']
    # get image region mappings which are tuples of the form (w1, h1, w2, h2)
    mappings = get_img_mappings(cfg, image)
    with torch.no_grad():
        rl_jscore, rl_reg_jscore, global_jscore, global_reg_jscore = 0.0, 0.0, 0.0, 0.0
        acc, reg_acc = 0.0, 0.0
        denom = 0
        for i, batch in enumerate(dataloader):
            # load images and labels
            images = batch['image'].to(global_dev)
            one_hot_masks = batch['one_hot_mask'].to(global_dev)
            true_masks = batch['true_mask'].to(global_dev)
            ids = batch['id']
            # run global model num_unc_samples times to get monte carlo entropy
            global_logits = torch.zeros((cfg.num_unc_samples, images.shape[0], cfg.global_model.n_classes, images.shape[2], images.shape[3]), device=global_dev)
            global_probs = torch.zeros((cfg.num_unc_samples, images.shape[0], cfg.global_model.n_classes, images.shape[2], images.shape[3]), device=global_dev)
            # global_logits = global_model(images)
            for j in range(cfg.num_unc_samples):
                global_logits[j] = global_model(images)
                global_probs[j] += F.softmax(global_logits[j], dim=1)
            mean_global_probs = torch.mean(global_probs, 0)
            global_preds = mean_global_probs.argmax(dim=1)
            # compute entropy map
            global_ents = Categorical(probs=torch.permute(mean_global_probs, (0, 2, 3, 1))).entropy()
            # create rl input and move to correct device
            rl_input = torch.stack((global_preds, global_ents), dim=1)
            rl_input = rl_input.to(cfg.rl_model.device)
            # run the rl policy on the segmentation and entropy maps
            rl_logits = rl_model(rl_input)
            # reshape the rl_logits to be 32 x 16 x 3
            rl_logits = rl_logits.reshape((rl_logits.shape[0], num_patches, num_models))
            rl_probs = F.softmax(rl_logits, dim=2)
            # use categorical probs to sample region-model pairs
            rl_dist = Categorical(probs=rl_probs)
            rl_sample = rl_dist.sample()
            # get global_logits for just one batch sample
            global_logits = global_logits[0]
            global_probs = F.softmax(global_logits, dim=1)
            global_preds = global_probs.argmax(dim=1)
            # compute jaccard score with global model only on all images and regions in batch
            global_img_jscores, global_reg_jscores = compute_jaccard(cfg, global_preds, true_masks, mappings)
            global_jscore += global_img_jscores.mean()
            global_reg_jscore += global_reg_jscores.mean()
            rl_img_probs = torch.zeros(global_probs.shape)
            # for each image in batch
            for b in range(images.shape[0]):
                # for each region, model pair pass that region to the appropriate model
                for r, m in enumerate(rl_sample[b]):
                    if m > 0:
                        mod = list(local_models.values())[m-1]
                        # get region to run local model on
                        img_region = get_img_region(images[b], mappings, r).unsqueeze(0)
                        # move region to correct device
                        dev = list(OmegaConf.to_container(cfg.local_models).values())[m-1]['device']
                        img_region = img_region.to(dev)
                        # torch.Size([1, 3, 56, 64])
                        logits = mod(img_region)
                        # torch.Size([3, 56, 64])
                        rl_img_probs[b, :, mappings[r][1]:mappings[r][3], mappings[r][0]:mappings[r][2]] = F.softmax(logits, dim=1)
                        # torch.Size([3, 56, 64])
                    elif m == 0:
                        rl_img_probs[b, :, mappings[r][1]:mappings[r][3], mappings[r][0]:mappings[r][2]] = global_probs[b, :, mappings[r][1]:mappings[r][3], mappings[r][0]:mappings[r][2]]
            rl_img_preds = rl_img_probs.argmax(dim=1)
            # compute rl policy jaccard score for each region
            rl_jscores, rl_reg_jscores = compute_jaccard(cfg, rl_img_preds, true_masks, mappings)
            rl_jscore += rl_jscores.mean()
            rl_reg_jscore += rl_reg_jscores.mean()
        global_jscore /= len(dataloader)
        global_reg_jscore /= len(dataloader)
        rl_jscore /= len(dataloader)
        rl_reg_jscore /= len(dataloader)
        wandb.log({f'avg_global_img_jaccard_{setname}': global_jscore,
            f'avg_global_reg_jaccard_{setname}': global_reg_jscore,
            f'avg_rl_img_jaccard_{setname}': rl_jscore,
            f'avg_rl_reg_jaccard_{setname}': rl_reg_jscore
            }, commit=commit)

def compute_cost(cfg: DictConfig, rl_sample: torch.Tensor):
    num_patches = cfg.num_horizontal_patches * cfg.num_vertical_patches
    # {'global': 16571, 'rl': 14736, 'local0': 1080595, 'local1': 17275459}
    gcost = 16571.0/(16571+1080595+17275459)
    l1cost = 1080595.0/(16571+1080595+17275459)
    l2cost = 17275459.0/(16571+1080595+17275459)
    # 32 x 16 sample
    costs = torch.zeros(rl_sample.shape[0])
    costs_reg = torch.zeros(rl_sample.shape)
    for b in range(rl_sample.shape[0]):
        for n in range(num_patches):
            if rl_sample[b, n] == 1:
                costs[b] += l1cost
                costs_reg[b, n] = l1cost
            elif rl_sample[b, n] == 2:
                costs[b] += l2cost
                costs_reg[b, n] = l2cost
    return costs, costs_reg

def compute_jaccard(cfg: DictConfig, preds: torch.Tensor, true_labels: torch.Tensor, mappings: list):
    n_classes = cfg.global_model.n_classes
    num_patches = cfg.num_horizontal_patches * cfg.num_vertical_patches
    preds = preds.to('cpu')
    true_labels = true_labels.to('cpu')
    img_jscores = torch.zeros((preds.shape[0]))
    reg_jscores = torch.zeros((preds.shape[0], num_patches))
    for b in range(preds.shape[0]):
        img_jscores[b] = jaccard_index(preds[b], true_labels[b], task='multiclass', num_classes=n_classes)
        for r in range(num_patches):
            reg_jscores[b, r] = jaccard_index(get_img_region(preds[b], mappings, r), get_img_region(true_labels[b], mappings, r), task='multiclass', num_classes=n_classes)
    return img_jscores, reg_jscores

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

def load_datasets(cfg: DictConfig):
    if cfg.dataset.name in {'battery', 'battery_2'}:
        log.info('Loading battery dataset')
        train_dataset = BatteryDataset(Path(cfg.dataset.path, cfg.dataset.train_images_dir), Path(cfg.dataset.path, cfg.dataset.train_labels_dir))
        val_dataset = BatteryDataset(Path(cfg.dataset.path, cfg.dataset.val_images_dir), Path(cfg.dataset.path, cfg.dataset.val_labels_dir))
        test_dataset = BatteryDataset(Path(cfg.dataset.path, cfg.dataset.test_images_dir), Path(cfg.dataset.path, cfg.dataset.test_labels_dir))
        # split dataset with fixed random seed
        split_dataset = random_split(train_dataset, [0.34, 0.33, 0.33], generator=torch.Generator().manual_seed(42))
        return {'train_dataset': split_dataset[2], 'val_dataset': val_dataset, 'test_dataset': test_dataset}
    else:
        raise ValueError(f'Dataset {cfg.dataset.name} is not supported')

def create_dataloaders(cfg: DictConfig, datasets: dict):
    train_dataloader = DataLoader(datasets['train_dataset'], batch_size=cfg.batch_size, shuffle=cfg.shuffle, pin_memory=cfg.pin_memory)
    val_dataloader = DataLoader(datasets['val_dataset'], batch_size=cfg.batch_size, shuffle=cfg.shuffle, pin_memory=cfg.pin_memory)
    test_dataloader = DataLoader(datasets['test_dataset'], batch_size=cfg.batch_size, shuffle=cfg.shuffle, pin_memory=cfg.pin_memory)
    return {'train_dataloader': train_dataloader, 'val_dataloader': val_dataloader, 'test_dataloader': test_dataloader}

def load_local_seg_models(local_models_cfg: DictConfig, optimizer_cfg: DictConfig):
    local_models, local_opts = {}, {}
    for name, val in local_models_cfg.items():
        local_models[name], local_opts[name] = utils.load_model_train(val, optimizer_cfg)
    return local_models, local_opts

if __name__ == "__main__":
    finetune()

