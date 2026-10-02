'''
* @name: opts.py
* @description: Hyperparameter configuration.
'''

import argparse
from dataclasses import dataclass,field

__all__ = ['Config','parse_opts']


@dataclass
class Config:
    # dataset
    datasetName: str = 'mosi'  # mosi, mosei or sims
    dataPath: str = 'datasets/mosi/unaligned_50.pkl'
    seq_lens: list = field(default_factory=lambda:[50,50,50])
    num_workers: int = 8
    train_mode: str = 'regression'
    # network
    CUDA_VISIBLE_DEVICES: str = '6'
    fusion_layer_depth: int = 2
    # common
    project_name: str = 'DEVA_Demo'
    is_test: int = 1
    seed: list = field(default_factory=lambda:[1000,1001,1002,1003,1004])
    models_save_root: str = './checkpoint'
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    n_epochs: int = 80
    early_stop: int = 8


def parse_opts():
    """命令行入口：解析参数并转成 Config（保留向后兼容）"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetName',type=str,default='mosi',help='mosi, mosei or sims')
    parser.add_argument('--dataPath',type=str,default='./datasets/unaligned_50.pkl')
    parser.add_argument('--seq_lens',type=int,nargs='+',default=[50,50,50])
    parser.add_argument('--num_workers',type=int,default=8)
    parser.add_argument('--train_mode',type=str,default='regression')
    parser.add_argument('--CUDA_VISIBLE_DEVICES',type=str,default='6')
    parser.add_argument('--fusion_layer_depth',type=int,default=2)
    parser.add_argument('--project_name',type=str,default='DEVA_Demo')
    parser.add_argument('--is_test',type=int,default=1)
    parser.add_argument('--seed',type=int,default=18)
    parser.add_argument('--models_save_root',type=str,default='./checkpoint')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--weight_decay',type=float,default=1e-4)
    parser.add_argument('--n_epochs',type=int,default=80)
    parser.add_argument('--early_stop',type=int,default=8)
    args = parser.parse_args()
    return Config(**vars(args))
