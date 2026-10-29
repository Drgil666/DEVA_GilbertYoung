# -*- coding: utf-8 -*-
"""
DEVA_mosi_train.py
===================
MOSI 训练入口：在这里写 opt 参数，然后调用 train.main(config)。

用法：
    python DEVA_mosi_train.py

注意：
  - dataPath 要指向「合并后」的 pkl（含 vision_text 和 audio_text 两个字段），
    否则 core/dataset.py 读不到会报 KeyError。
"""

from opts import Config
from train import main

# 在这里配置 MOSI 训练参数（未列出的用 Config 里的默认值）
config = Config(
    datasetName='mosi',
    dataPath=r'E:\PycharmProject\DEVA\dataset\mosi\unaligned_50.pkl',  # 合并后的 pkl
    seq_lens=[50,50,50],
    num_workers=0,
    train_mode='regression',
    CUDA_VISIBLE_DEVICES='0',  # 改成你的 GPU 编号
    fusion_layer_depth=2,
    project_name='DEVA_mosi',
    is_test=1,
    seed=[1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015],
    models_save_root='./checkpoint',
    batch_size=16,
    lr=1e-4,
    weight_decay=1e-3,
    n_epochs=80,
    early_stop=10,
)

if __name__ == '__main__':
    main(config)
