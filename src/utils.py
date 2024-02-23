import logging
import glob
import tarfile
from pathlib import Path
import random

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

import numpy as np

import torch
from torch import nn
from torch.optim import Optimizer
from fvcore.nn import FlopCountAnalysis

from models.unet import UNet, UNetSmallChannels, UNetTinyChannels, UNetTinyThreeBlock, UNetTinyTwoBlock
from models.resnet import ResNet, BasicBlock

# logger
log = logging.getLogger(__name__)

def save_model_checkpoint(epoch: int, device: str, model: nn.Module, optimizer: Optimizer, output_dir: str, prefix = None, suffix = None, overwrite=False):
    # consider abstracting the prefix and suffix string building into a separate function
    if overwrite == False:
        filepath = f'{output_dir}/{prefix + "_" if prefix is not None else ""}model_checkpoint_{epoch}{"_" + suffix if suffix is not None else ""}.pt'
    else:
        filepath = f'{output_dir}/{prefix + "_" if prefix is not None else ""}model_checkpoint{"_" + suffix if suffix is not None else ""}.pt'
    torch.save({'epoch': epoch,
        'device': device,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        filepath)

def load_seg_model(model_cfg: DictConfig, dropout=False):
    # get device
    dev = torch.device(model_cfg.device)
    if model_cfg.name == 'unet_tiny':
        model = UNetTinyChannels(n_channels=model_cfg.n_channels, n_classes=model_cfg.n_classes, bilinear=model_cfg.bilinear, drop_train=dropout)
    elif model_cfg.name in ['unet_small', 'unet_small_ft_final']:
        model = UNetSmallChannels(n_channels=model_cfg.n_channels, n_classes=model_cfg.n_classes, bilinear=model_cfg.bilinear, drop_train=dropout)
    elif model_cfg.name in ['unet', 'unet_ft_final']:
        model = UNet(n_channels=model_cfg.n_channels, n_classes=model_cfg.n_classes, bilinear=model_cfg.bilinear, drop_train=dropout)
    elif model_cfg.name in ['unet_tiny_two_block', 'unet_tiny_two_block_distill']:
        model = UNetTinyTwoBlock(n_channels=model_cfg.n_channels, n_classes=model_cfg.n_classes, bilinear=model_cfg.bilinear, drop_train=dropout)
    elif model_cfg.name in ['resnet1_tiny', 'resnet1_tiny_ft', 'resnet1_tiny_ft_final', 'resnet1_tiny_mnist'] and model_cfg.block_type == 'basic':
        model = ResNet(BasicBlock, model_cfg.layer_list, model_cfg.initial_kernel_size, model_cfg.action_space_size)
    else:
        raise ValueError(f'model {model_cfg.name} is not supported')
    checkpoint = torch.load(model_cfg.checkpoint, map_location='cpu')
    #checkpoint = torch.load(model_cfg.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(dev)
    model.eval()

    return model

def load_model_train(model_cfg: DictConfig, optimizer_cfg: DictConfig, dropout=True):
    dev = torch.device(model_cfg.device)
    if model_cfg.name == 'unet_tiny':
        model = UNetTinyChannels(n_channels=model_cfg.n_channels, n_classes=model_cfg.n_classes, bilinear=model_cfg.bilinear, drop_train=dropout)
    elif model_cfg.name == 'unet_small':
        model = UNetSmallChannels(n_channels=model_cfg.n_channels, n_classes=model_cfg.n_classes, bilinear=model_cfg.bilinear, drop_train=dropout)
    elif model_cfg.name == 'unet':
        model = UNet(n_channels=model_cfg.n_channels, n_classes=model_cfg.n_classes, bilinear=model_cfg.bilinear, drop_train=dropout)
    elif model_cfg.name == 'unet_tiny_two_block':
        model = UNetTinyTwoBlock(n_channels=model_cfg.n_channels, n_classes=model_cfg.n_classes, bilinear=model_cfg.bilinear, drop_train=dropout)
    elif model_cfg.name in ['resnet1_tiny', 'resnet1_tiny_mnist'] and model_cfg.block_type == 'basic':
        model = ResNet(BasicBlock, model_cfg.layer_list, model_cfg.initial_kernel_size, model_cfg.action_space_size)
    else:
        raise ValueError(f'Model {model_cfg.name} is not supported')
    if optimizer_cfg.name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_cfg.learning_rate)
    else:
        raise ValueError(f'Optimizer {optimizer_cfg.name} is not supported')
    checkpoint = torch.load(model_cfg.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(dev)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.train()
    return model, optimizer

def save_code_snapshot(cfg: DictConfig):
    if cfg.save_code_snapshot:
        log.info('Compressing and saving source archive')
        # get all python files, notebook files, and shell scripts
        source_files = glob.glob('**/*.py', recursive=True) + glob.glob('**/*.ipynb', recursive=True) + glob.glob('**/*.sh', recursive=True)
        # get the hydra config output directory
        output_dir = HydraConfig.get().run.dir
        # create output path for source archive
        p = Path(output_dir, 'source.tar.gz')
        # compress and save source files
        with tarfile.open(p, 'w:gz') as source_tar:
            for f in source_files:
                # only save source files that are not in the outputs directory
                if 'outputs' not in f:
                    source_tar.add(f)

# this should make runs deteriministic for testing purposes
def remove_randomness(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    #torch.set_deterministic_debug_mode(1)

def get_memory_usage(model: nn.Module):
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    return mem

def get_num_params(cfg: DictConfig, global_model: nn.Module, local_models: dict[nn.Module], rl_model: nn.Module, trainable: bool = False):
    num_params = {}
    if trainable is False:
        num_params['global'] = sum(param.numel() for param in global_model.parameters())
        num_params['rl'] = sum(param.numel() for param in rl_model.parameters())
        for i, model in enumerate(local_models.values()):
            num_params[f'local{i}'] = sum(param.numel() for param in model.parameters())
    else:
        num_params['global'] = sum(param.numel() for param in global_model.parameters() if param.requires_grad)
        num_params['rl'] = sum(param.numel() for param in rl_model.parameters() if param.requires_grad)
        for i, model in enumerate(local_models.values()):
            num_params[f'local{i}'] = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return num_params

def get_flops(model: nn.Module, input: torch.Tensor):
    flops = FlopCountAnalysis(model, input)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    flops.tracer_warnings("none")
    return flops.total()

