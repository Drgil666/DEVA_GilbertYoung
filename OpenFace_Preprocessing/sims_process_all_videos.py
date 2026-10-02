import glob
import os
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.bert import BertTextEncoder

# 原始数据的目录
RAW_DIR = r'E:\PycharmProject\DEVA\dataset\sims\raw'
# 总目录
SIMS_DIR = r'E:\PycharmProject\DEVA\dataset\sims'
# 处理的中间目录
OUT_DIR = os.path.join(SIMS_DIR,'processed')
# ——sims
#  |-raw
#  |-processed
#  |-各个样本的csv
# openface的目录,注意！不要用openface 2.2.0，用2.1.0
OPENFACE_EXE = r'E:\ChromeDownload\OpenFace_2.1.0_win_x64\FeatureExtraction.exe'
LOG_FILE = os.path.join(SIMS_DIR,'process_log.txt')
# 原pkl的目录
PKL_PATH = os.path.join(SIMS_DIR,'unaligned_39.pkl')
# 视觉模态pkl的保存
VISION_TEXT_PKL = os.path.join(SIMS_DIR,'vision_text.pkl')
RAW_BACKUP_DIR = os.path.join(SIMS_DIR,'openface_raw')  # 原始 csv 备份目录
AUDIO_TEXT_PKL = os.path.join(SIMS_DIR,'audio_text.pkl')  # OpenSMILE 生成的原始 audio_text（可选）
MERGED_PKL = os.path.join(SIMS_DIR,'unaligned_39.pkl')  # 合并后直接覆盖原 pkl（同名）


def log(msg):
    line = '[{}] {}'.format(time.strftime('%H:%M:%S'),msg)
    print(line,flush=True)
    with open(LOG_FILE,'a',encoding='utf-8') as f:
        f.write(line+'\n')


def load_pkl(path):
    with open(path,'rb') as f:
        obj = pickle.load(f)
    print("读取成功")
    return obj


def save_pkl(path,obj):
    with open(path,'wb') as f:
        pickle.dump(obj,f)
    print("保存成功")


def generate_id(path):
    parts = path.split("\\")
    video_name = parts[-2]  # 父文件夹名 = 视频 ID
    for part in parts:
        if part.endswith(".csv"):
            file_name = part[:-4]  # csv 文件名 = 片段号
    return f"{video_name}$_${file_name}"


AU_to_des = {
    ' AU01_c':'抬起内眉',
    ' AU02_c':'抬起外眉',
    ' AU04_c':'低眉',
    ' AU05_c':'抬起上眼睑',
    ' AU06_c':'抬起脸颊',
    ' AU07_c':'紧闭眼皮',
    ' AU09_c':'皱鼻头',
    ' AU10_c':'扬起上唇',
    ' AU12_c':'拉唇角',
    ' AU15_c':'下压唇角',
    ' AU20_c':'拉伸嘴唇',
    ' AU23_c':'抿紧嘴唇',
    ' AU25_c':'分开嘴唇',
    ' AU26_c':'掉下巴',
    ' AU28_c':'嘬嘴',
    ' AU45_c':'眨眼',
}

TARGET_COLUMNS = [' AU01_c',' AU02_c',' AU04_c',' AU05_c',' AU06_c',' AU07_c',
                  ' AU09_c',' AU10_c',' AU12_c',' AU15_c',' AU20_c',' AU23_c',
                  ' AU25_c',' AU26_c',' AU28_c',' AU45_c']


def find_AU(path):
    """读取单个 csv，找出「连续 >=3 帧值为 1」的 AU，按数值和降序取前 8 个"""
    all_data = pd.read_csv(path)
    column_names = all_data.columns.tolist()
    selected_columns = [idx for idx,col in enumerate(column_names) if col in TARGET_COLUMNS]
    data = all_data.iloc[:,selected_columns]

    selected = []
    max_consecutive_ones = 3
    for column in data.columns:
        consecutive_ones = 0
        found = False
        for value in data[column]:
            if value == 1:
                consecutive_ones += 1
                if consecutive_ones >= max_consecutive_ones:
                    found = True
                    break
            else:
                consecutive_ones = 0
        if found:
            selected.append(column)

    selected_data = data[selected]
    sorted_sums = sorted(selected_data.sum().to_dict().items(),key=lambda x:x[1],reverse=True)
    return dict(sorted_sums[:8])


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


def process_video(video_id):
    video_folder = os.path.join(RAW_DIR,video_id)
    target_folder = os.path.join(SIMS_DIR,video_id)
    if os.path.isdir(target_folder):
        log('[SKIP] {} 已处理'.format(video_id))
        return
    files = sorted([os.path.join(video_folder,f) for f in os.listdir(video_folder)
                    if f.lower().endswith('.mp4') and not f.startswith('._')])
    if not files:
        log('[WARN] {} 无 mp4 文件'.format(video_id))
        return
    cmd = [OPENFACE_EXE,'-out_dir',OUT_DIR]
    for f in files:
        cmd += ['-f',f]
    log('[RUN] {} 开始，{} 个片段'.format(video_id,len(files)))
    t0 = time.time()
    os.makedirs(OUT_DIR,exist_ok=True)  # 确保输出目录存在（防止改名后 OpenFace 无法重建）
    result = subprocess.run(cmd,cwd=SIMS_DIR,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    if result.returncode != 0 or not os.path.isdir(OUT_DIR):
        stderr_msg = result.stderr.decode('utf-8',errors='replace') if result.stderr else ''
        log('[ERROR] {} 失败（exit={}），跳过'.format(video_id,result.returncode))
        if stderr_msg.strip():
            log('  OpenFace 输出: {}'.format(stderr_msg.strip()[-500:]))
        return
    # 备份原始 csv（在清理和改名之前，把 .csv 复制一份到 openface_raw/<video_id>）
    os.makedirs(RAW_BACKUP_DIR,exist_ok=True)
    backup_folder = os.path.join(RAW_BACKUP_DIR,video_id)
    if os.path.isdir(backup_folder):
        log('[BACKUP] {} 已有备份，跳过'.format(video_id))
    else:
        os.makedirs(backup_folder)
        for entry in os.listdir(OUT_DIR):
            if entry.endswith('.csv'):
                shutil.copy2(os.path.join(OUT_DIR,entry),os.path.join(backup_folder,entry))
        log('[BACKUP] {} 原始 csv 已备份到 {}'.format(video_id,backup_folder))
    n_csv = 0
    for entry in os.listdir(OUT_DIR):
        p = os.path.join(OUT_DIR,entry)
        if os.path.isdir(p) and entry.endswith('_aligned'):
            shutil.rmtree(p)
        elif os.path.isfile(p) and entry.endswith(('.txt','.avi','.hog')):
            os.remove(p)
        elif os.path.isfile(p) and entry.endswith('.csv'):
            n_csv += 1
    os.rename(OUT_DIR,target_folder)
    log('[DONE] {} 完成，{} 个 csv，耗时 {}s'.format(video_id,n_csv,int(time.time()-t0)))


if __name__ == '__main__':
    videos = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR,d))])
    done = [v for v in videos if os.path.isdir(os.path.join(SIMS_DIR,v))]
    log('共 {} 个视频，已完成 {}，待处理 {}'.format(len(videos),len(done),len(videos)-len(done)))
    for v in videos:
        process_video(v)
    log('全部处理完成')

    # ============ 验证主 pkl ============
    log('==== 验证主 pkl ====')
    data = load_pkl(PKL_PATH)
    log('valid 字段: {}'.format(list(data['valid'].keys())))
    log('id 示例: {}'.format(data['valid']['id'][0]))

    # ============ 4ced0301 + aafb3bf5 + 943bcab1：收集 CSV → 提取 AU → 保存 ============
    # 若 vision_text pkl 已存在，跳过这几步，直接到回读验证 + 统计长度
    if os.path.exists(VISION_TEXT_PKL):
        log('==== vision_text pkl 已存在，跳过 CSV 收集 / AU 提取 / 生成 ====')
    else:
        # 收集所有 CSV（只收视频文件夹一级的 csv，排除中间目录）
        log('==== 收集所有 CSV ====')
        EXCLUDE_DIRS = {'raw','processed','openface_raw','raw_wav','opensmile'}
        csv_files = []
        for d in os.listdir(SIMS_DIR):
            if d in EXCLUDE_DIRS or not os.path.isdir(os.path.join(SIMS_DIR,d)):
                continue
            csv_files.extend(glob.glob(os.path.join(SIMS_DIR,d,'*.csv')))
        csv_files = sorted(csv_files)
        log('共收集到 {} 个 CSV'.format(len(csv_files)))
        for csv_file in csv_files[:10]:
            log('  {} -> {}'.format(csv_file,generate_id(csv_file)))
        if len(csv_files) > 10:
            log('  ...（共 {} 个，如需完整列表去掉 [:10] 限制）'.format(len(csv_files)))

        # 提取 AU 转自然语言描述
        log('==== 提取 AU 转自然语言描述 ====')
        vision_text_list = []
        for path in csv_files:
            selected_data = find_AU(path)
            description = [AU_to_des[au] for au in selected_data.keys() if au in AU_to_des]
            vision_text = '说话人做出如下表情: '
            for des in description:
                vision_text += des+', '
            vision_text_list.append({'id':generate_id(path),'raw_vision_text':vision_text})
        log('共生成 {} 条 vision_text'.format(len(vision_text_list)))
        for item in vision_text_list[:5]:
            log('  {}'.format(item))

        # 保存 pkl
        log('==== 保存 vision_text pkl ====')
        save_pkl(VISION_TEXT_PKL,vision_text_list)
        log('已保存到 {}'.format(VISION_TEXT_PKL))

    # ============ b28ef662 + 999587ba：回读验证 + 统计长度 ============
    log('==== 回读验证 + 统计长度 ====')
    data = load_pkl(VISION_TEXT_PKL)
    for item in data[:10]:
        log('  {}'.format(item))
    max_len = 0
    total = 0
    for item in data:
        n = len(item['raw_vision_text'].split())
        if n > max_len:
            max_len = n
        total += n
    mean_len = total/len(data)
    log('mean_len: {}'.format(mean_len))
    log('max_len: {}'.format(max_len))

    # ============ 合并 vision_text / audio_text 进主 pkl ============
    log('==== 合并 vision_text / audio_text 进主 pkl ====')
    with open(PKL_PATH,'rb') as f:
        main_data = pickle.load(f)
    with open(VISION_TEXT_PKL,'rb') as f:
        vision_list = pickle.load(f)
    vision_map = {str(item['id']):item['raw_vision_text'] for item in vision_list}

    audio_map = {}
    if os.path.exists(AUDIO_TEXT_PKL):
        with open(AUDIO_TEXT_PKL,'rb') as f:
            audio_list = pickle.load(f)
        audio_map = {str(item['id']):item['raw_audio_text'] for item in audio_list}
        log('加载到 {} 条 audio_text'.format(len(audio_map)))
    else:
        log('[WARN] 未找到 {}，跳过 audio_text 合并'.format(AUDIO_TEXT_PKL))

    # 检查 id 匹配率
    all_ids = []
    for split in ['train','valid','test']:
        all_ids.extend([str(i) for i in main_data[split]['id']])
    vt_hit = sum(1 for i in all_ids if i in vision_map)
    log('vision_text id 匹配: {}/{}'.format(vt_hit,len(all_ids)))
    if audio_map:
        at_hit = sum(1 for i in all_ids if i in audio_map)
        log('audio_text id 匹配: {}/{}'.format(at_hit,len(all_ids)))

    log('加载 BERT 模型 bert-base-chinese（首次会下载，编码较慢请耐心等待）...')
    encoder = BertTextEncoder(use_finetune=False,pretrained="bert-base-chinese")

    for split in ['train','valid','test']:
        ids = [str(i) for i in main_data[split]['id']]
        log('  {}: {} 个样本，编码 vision_text ...'.format(split,len(ids)))
        main_data[split]['vision_text'] = bert_encode([vision_map.get(i,'') for i in ids],encoder)
        if audio_map:
            log('  {}: 编码 audio_text ...'.format(split))
            main_data[split]['audio_text'] = bert_encode([audio_map.get(i,'') for i in ids],encoder)

    tmp_path = MERGED_PKL+'.tmp'
    with open(tmp_path,'wb') as f:
        pickle.dump(main_data,f)
    os.replace(tmp_path,MERGED_PKL)  # 原子替换：写入中断也不会损坏原 pkl
    log('合并完成，已保存到 {}'.format(MERGED_PKL))

    # ============ 验证合并结果 ============
    log('==== 验证合并结果 ====')
    with open(MERGED_PKL,'rb') as f:
        new_data = pickle.load(f)
    for split in ['train','valid','test']:
        keys = list(new_data[split].keys())
        has_vt = 'vision_text' in keys
        has_at = 'audio_text' in keys
        n = len(new_data[split]['id'])
        vt_shape = new_data[split]['vision_text'].shape if has_vt else None
        at_shape = new_data[split]['audio_text'].shape if has_at else None
        log('  {}: {} 样本, vision_text={} {}, audio_text={} {}'.format(
            split,n,has_vt,vt_shape,has_at,at_shape))
    log('示例 id: {}'.format(new_data['train']['id'][0]))
    log('示例 vision_text[0] shape: {}'.format(new_data['train']['vision_text'][0].shape))
