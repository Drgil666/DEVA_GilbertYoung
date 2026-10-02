# -*- coding: utf-8 -*-
"""
sims_train.py
=============
SIMS 训练入口：在这里写 opt 参数，然后调用 train.main(config)。

用法：
    python sims_train.py

注意：
  - dataPath 要指向「合并后」的 pkl（含 vision_text 和 audio_text 两个字段），
    否则 core/dataset.py 读不到会报 KeyError。
"""

from opts import Config
from train import main

# 在这里配置 SIMS 训练参数（未列出的用 Config 里的默认值）
config = Config(
    datasetName='sims',
    dataPath=r'E:\PycharmProject\DEVA\dataset\sims\unaligned_39.pkl',  # 合并后的 pkl
    seq_lens=[39,55,400],
    num_workers=0,
    train_mode='regression',
    CUDA_VISIBLE_DEVICES='0',  # 改成你的 GPU 编号
    fusion_layer_depth=2,
    project_name='DEVA_sims',
    is_test=1,
    seed=[1000,1001,1002,1003,1004],
    models_save_root='./checkpoint',
    batch_size=16,
    lr=1e-5,
    weight_decay=1e-3,
    n_epochs=80,
    early_stop=8,
)

if __name__ == '__main__':
    main(config)
