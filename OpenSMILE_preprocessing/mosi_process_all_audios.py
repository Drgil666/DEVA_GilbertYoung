# -*- coding: utf-8 -*-
"""
mosi_process_all_audios.py
===========================
由 opensmile_mosi.ipynb 转换而来，处理对象：dataset/mosi/raw。

流程：
  1. 从视频提取音频（mp4 -> wav，用 moviepy）
  2. 用 SMILExtract 从 wav 提取声学特征（-> csv，IS10_paraling 配置）
  3. 解析 csv，提取 loudness / jitter / shimmer / f0 四个均值
  4. 按三分位数把每个特征分成 low/normal/high，生成自然语言描述
  5. 保存 audio_text.pkl
  6. 统计平均/最大长度
  7. 合并 vision_text / audio_text 进主 pkl（BERT 编码后写进 unaligned_50.pkl）
  8. 验证合并结果

依赖：
  - moviepy（提取音频，需要 ffmpeg）
  - OpenSMILE 3.0（SMILExtract.exe，见下方 SMILE_BIN_DIR / SMILE_CONFIG）
  - transformers + torch（BERT 编码）
"""

import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.bert import BertTextEncoder

# ============ 路径配置（按需修改） ============
RAW_DIR = r'E:\PycharmProject\DEVA\dataset\mosi\raw'  # 输入：原始视频 mp4
WAV_DIR = r'E:\PycharmProject\DEVA\dataset\mosi\raw_wav'  # 中间：提取出的 wav
SMILE_OUTPUT_DIR = r'E:\PycharmProject\DEVA\dataset\mosi\opensmile'  # 中间：OpenSMILE 特征 csv
AUDIO_TEXT_PKL = r'E:\PycharmProject\DEVA\dataset\mosi\audio_text.pkl'  # 输出：audio_text pkl
PKL_PATH = r'E:\PycharmProject\DEVA\dataset\mosi\unaligned_50.pkl'  # 主 pkl（标签）
MERGED_PKL = r'E:\PycharmProject\DEVA\dataset\mosi\unaligned_50.pkl'  # 合并后直接覆盖原 pkl（同名）

# OpenSMILE 安装位置（注意改成你自己的路径）
SMILE_BIN_DIR = r'E:\ChromeDownload\opensmile-3.0.2-windows-x86_64\bin'
SMILE_CONFIG = r'E:\ChromeDownload\opensmile-3.0.2-windows-x86_64\config\is09-13\IS10_paraling.conf'
SMILE_EXE = os.path.join(SMILE_BIN_DIR,'SMILExtract.exe')


# ============ 公共函数 ============
def save_pkl(path,obj):
    with open(path,'wb') as f:
        pickle.dump(obj,f)
    print("保存成功")


def load_pkl(path):
    with open(path,'rb') as f:
        obj = pickle.load(f)
    print("读取成功")
    return obj


def extract_audio(video_path,audio_path):
    from moviepy.editor import VideoFileClip
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)


def find_key_value(path):
    """从 OpenSMILE 输出的 csv 里提取 loudness/jitter/shimmer/f0 四个均值"""
    data = pd.read_csv(path,sep='\t',header=0)
    value = data.iloc[-1,:]
    # 最后一行是逗号分隔的所有特征值，取 [1:-1] 去掉首尾多余项
    split_value = [float(item) for item in value[0].split(",")[1:-1]]
    # 音频过短时 OpenSMILE 可能产出空/截断的 csv，跳过该片段
    if len(split_value) < 1488:
        print('[WARN] 跳过 {}: 特征数不足（{}）'.format(path,len(split_value)))
        return {}
    key_value = {
        "loudness_sma_amean":split_value[2],
        "jitterLocal_sma_amean":split_value[1449],
        "shimmerLocal_sma_amean":split_value[1487],
        "F0final_sma_amean":split_value[1430],
    }
    wav_id = path.split('\\')[-2]+'$_$'+path.split('\\')[-1][:-4]
    return {wav_id:key_value}


def classify(value,low_th,high_th):
    """按 [low_th, high_th) 把数值分成 low / normal / high"""
    if value < low_th:
        return 'low'
    elif value < high_th:
        return 'normal'
    else:
        return 'high'


def bert_encode(texts,encoder,max_length=50,batch_size=32):
    """用 BertTextEncoder 把文本列表编码成 (N, max_length, 768) 的 float32 数组"""
    tokenizer = encoder.get_tokenizer()
    embeddings = []
    for i in tqdm(range(0,len(texts),batch_size),desc='BERT 编码'):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch,return_tensors='pt',max_length=max_length,
                        truncation=True,padding='max_length')
        # 拼成 (batch, 3, seq_len)：input_ids, attention_mask, token_type_ids
        input_tensor = torch.stack([enc['input_ids'],enc['attention_mask'],enc['token_type_ids']],dim=1)
        out = encoder.forward(input_tensor)  # (batch, seq_len, 768)
        embeddings.append(out.cpu().numpy())
    return np.concatenate(embeddings,axis=0).astype(np.float32)


if __name__ == '__main__':
    video_folders = sorted([d for d in os.listdir(RAW_DIR)
                            if os.path.isdir(os.path.join(RAW_DIR,d))])
    print('共 {} 个视频文件夹'.format(len(video_folders)))

    # ============ Step 1: 提取音频（mp4 -> wav） ============
    print('==== Step 1: 提取音频 ====')
    for folder in video_folders:
        folder_path = os.path.join(RAW_DIR,folder)
        wav_folder = os.path.join(WAV_DIR,folder)
        mp4_files = sorted([f for f in os.listdir(folder_path)
                            if f.lower().endswith('.mp4') and not f.startswith('._')])
        os.makedirs(wav_folder,exist_ok=True)
        # 逐个检查每个 mp4 对应的 wav 是否存在（按文件名一一对应，而非只按数量）
        for file in mp4_files:
            wavname = file.rsplit('.',1)[0]+'.wav'
            wav_path = os.path.join(wav_folder,wavname)
            if os.path.exists(wav_path):
                continue
            extract_audio(os.path.join(folder_path,file),wav_path)
        print('folder:{} complete'.format(folder))

    # ============ Step 2: SMILExtract 提取特征（wav -> csv） ============
    print('==== Step 2: SMILExtract 提取特征 ====')
    for folder in video_folders:
        wav_folder = os.path.join(WAV_DIR,folder)
        output_folder = os.path.join(SMILE_OUTPUT_DIR,folder)
        os.makedirs(output_folder,exist_ok=True)
        for wav in sorted(os.listdir(wav_folder)):
            if not wav.lower().endswith('.wav'):
                continue
            wav_path = os.path.join(wav_folder,wav)
            csv_path = os.path.join(output_folder,wav[:-4]+'.csv')
            # 检查 csv 是否存在且非空（空的说明上次提取失败，需重跑）
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                continue
            cmd = [SMILE_EXE,'-C',SMILE_CONFIG,'-I',wav_path,'-O',csv_path]
            subprocess.run(cmd,cwd=SMILE_BIN_DIR)
        print('folder:{} smilextract complete'.format(folder))

    # ============ Step 3-5: 解析 csv -> 生成 audio_text -> 保存 ============
    # 若 audio_text.pkl 已存在，跳过这几步
    if os.path.exists(AUDIO_TEXT_PKL):
        print('==== audio_text.pkl 已存在，跳过 Step 3/4/5 ====')
    else:
        # Step 3: 解析 csv -> LJSF
        print('==== Step 3: 解析 csv 提取 LJSF ====')
        LJSF = dict()
        for folder in video_folders:
            output_folder = os.path.join(SMILE_OUTPUT_DIR,folder)
            for csv_file in sorted(os.listdir(output_folder)):
                if csv_file.lower().endswith('.csv'):
                    LJSF.update(find_key_value(os.path.join(output_folder,csv_file)))
        print('共 {} 条音频特征'.format(len(LJSF)))

        # Step 4: 按三分位数生成自然语言描述
        print('==== Step 4: 生成 audio_text ====')
        loudness_list = sorted([v['loudness_sma_amean'] for v in LJSF.values()])
        jitter_list = sorted([v['jitterLocal_sma_amean'] for v in LJSF.values()])
        shimmer_list = sorted([v['shimmerLocal_sma_amean'] for v in LJSF.values()])
        f0_list = sorted([v['F0final_sma_amean'] for v in LJSF.values()])

        mid_point = len(loudness_list)//3
        upper_limit = mid_point+len(loudness_list)//3
        Lm,Lu = loudness_list[mid_point],loudness_list[upper_limit]
        Jm,Ju = jitter_list[mid_point],jitter_list[upper_limit]
        Sm,Su = shimmer_list[mid_point],shimmer_list[upper_limit]
        Fm,Fu = f0_list[mid_point],f0_list[upper_limit]

        audio_text_list = []
        for key,value in LJSF.items():
            Lat = classify(value['loudness_sma_amean'],Lm,Lu)+' loudness'
            Jat = classify(value['jitterLocal_sma_amean'],Jm,Ju)+' jitter'
            Sat = classify(value['shimmerLocal_sma_amean'],Sm,Su)+' shimmer'
            Fat = classify(value['F0final_sma_amean'],Fm,Fu)+' pitch'
            audio_text = 'The speaker speaks in the following tone: '+Lat+', '+Jat+', '+Sat+', '+Fat+'.'
            audio_text_list.append({'id':key,'raw_audio_text':audio_text})

        print('共生成 {} 条 audio_text'.format(len(audio_text_list)))
        print(audio_text_list[0])

        # Step 5: 保存 pkl
        print('==== Step 5: 保存 audio_text pkl ====')
        save_pkl(AUDIO_TEXT_PKL,audio_text_list)
        print('已保存到 {}'.format(AUDIO_TEXT_PKL))

    # ============ Step 6: 统计长度 ============
    print('==== Step 6: 统计长度 ====')
    data = load_pkl(AUDIO_TEXT_PKL)
    max_len = 0
    total = 0
    for item in data:
        n = len(item['raw_audio_text'].split())
        if n > max_len:
            max_len = n
        total += n
    mean_len = total/len(data)
    print('mean_len: {}'.format(mean_len))
    print('max_len: {}'.format(max_len))

    # ============ Step 7: 合并 audio_text 进主 pkl ============
    print('==== Step 7: 合并 audio_text 进主 pkl ====')
    with open(PKL_PATH,'rb') as f:
        main_data = pickle.load(f)

    # audio_text（本脚本刚生成，必有）
    with open(AUDIO_TEXT_PKL,'rb') as f:
        audio_list = pickle.load(f)
    audio_map = {str(item['id']):item['raw_audio_text'] for item in audio_list}
    print('加载到 {} 条 audio_text'.format(len(audio_map)))

    # 检查 id 匹配率
    all_ids = []
    for split in ['train','valid','test']:
        all_ids.extend([str(i) for i in main_data[split]['id']])
    at_hit = sum(1 for i in all_ids if i in audio_map)
    print('audio_text id 匹配: {}/{}'.format(at_hit,len(all_ids)))

    print('加载 BERT 模型 bert-base-uncased ...')
    encoder = BertTextEncoder(use_finetune=False,pretrained="bert-base-uncased")

    for split in ['train','valid','test']:
        ids = [str(i) for i in main_data[split]['id']]
        print('  {}: {} 个样本，编码 audio_text ...'.format(split,len(ids)))
        main_data[split]['audio_text'] = bert_encode([audio_map.get(i,'') for i in ids],encoder)

    tmp_path = MERGED_PKL+'.tmp'
    with open(tmp_path,'wb') as f:
        pickle.dump(main_data,f)
    os.replace(tmp_path,MERGED_PKL)  # 原子替换：写入中断也不会损坏原 pkl
    print('合并完成，已保存到 {}'.format(MERGED_PKL))

    # ============ Step 8: 验证合并结果 ============
    print('==== Step 8: 验证合并结果 ====')
    with open(MERGED_PKL,'rb') as f:
        new_data = pickle.load(f)
    for split in ['train','valid','test']:
        has_at = 'audio_text' in new_data[split]
        at_shape = new_data[split]['audio_text'].shape if has_at else None
        print('  {}: {} 样本, audio_text={} {}'.format(
            split,len(new_data[split]['id']),has_at,at_shape))
    print('示例 id: {}'.format(new_data['train']['id'][0]))
    print('示例 audio_text[0] shape: {}'.format(new_data['train']['audio_text'][0].shape))
