import csv
import os
import pickle
import time

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from core.dataset import MMDataLoader
from core.metric import MetricsTop
from core.scheduler import get_scheduler
from core.utils import AverageMeter,setup_seed,save_best_model,load_best_model
from models.deva import build_model
from opts import *

device = None


train_mae, val_mae = [], []


def save_pkl(path, obj):
    pickle_file = open(path, 'wb')
    pickle.dump(obj, pickle_file)
    pickle_file.close()
    print("保存成功")

def load_pkl(path):
    pickle_file = open(path, 'rb')
    obj = pickle.load(pickle_file)
    pickle_file.close()
    print("读取成功")
    return obj


def get_metric_columns(datasetName):
    if datasetName == 'sims':
        return ['F1_score','Mult_acc_2','Mult_acc_3','Mult_acc_5','MAE','Corr']
    else:  # mosi / mosei
        return ['Has0_acc_2','Has0_F1_score','Non0_acc_2','Non0_F1_score',
                'Mult_acc_5','Mult_acc_7','MAE','Corr']


def get_acc2_name(datasetName):
    """返回「Acc_2」指标名：mosi/mosei 用 Non0_acc_2，sims 用 Mult_acc_2"""
    if datasetName == 'sims':
        return 'Mult_acc_2'
    else:
        return 'Non0_acc_2'


def write_test_csv(results,train_start_time,log_path,datasetName,seed=None):
    columns = get_metric_columns(datasetName)
    csv_path = os.path.join(log_path,'test_metrics.csv')
    os.makedirs(os.path.dirname(csv_path),exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path,'a',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['time','seed']+columns)
        if write_header:
            writer.writeheader()
        row = {'time':train_start_time,'seed':seed}
        for c in columns:
            v = results.get(c)
            row[c] = round(float(v),4) if v is not None else ''
        writer.writerow(row)


def write_log_line(log_path,phase,epoch,results,seed=None):
    log_file = os.path.join(log_path,'training_log.txt')
    os.makedirs(os.path.dirname(log_file),exist_ok=True)
    with open(log_file,'a',encoding='utf-8') as f:
        f.write('[{}][seed {}][epoch {}][{}] {}\n'.format(
            time.strftime('%Y-%m-%d %H:%M:%S'),seed,epoch,phase,results))


def average_metrics(results):
    """对多个指标字典求平均（数值字段取平均）"""
    avg = {}
    for key in results[0].keys():
        vals = [float(r[key]) for r in results if r.get(key) is not None]
        avg[key] = round(sum(vals)/len(vals),4) if vals else ''
    return avg


def write_summary_csv(avg_results,csv_path,datasetName):
    columns = get_metric_columns(datasetName)
    os.makedirs(os.path.dirname(csv_path),exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path,'a',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow({c:avg_results.get(c,'') for c in columns})


def main(config=None):
    global device
    if config is None:
        config = parse_opts()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print("device: {}:{}".format(device,config.CUDA_VISIBLE_DEVICES))

    # 归一化 seed：支持单个 int 或列表
    seeds = config.seed
    if seeds is None:
        seeds = [None]
    elif not isinstance(seeds,(list,tuple)):
        seeds = [seeds]

    log_path = os.path.join(".","log",config.project_name)
    os.makedirs(log_path,exist_ok=True)
    print("log_path :", log_path)

    save_path = os.path.join(config.models_save_root,config.project_name)
    os.makedirs(save_path,exist_ok=True)
    print("model_save_path :", save_path)

    dataLoader = MMDataLoader(config)
    loss_fn = torch.nn.MSELoss()
    metrics = MetricsTop().getMetics(config.datasetName)

    all_seed_results = []

    for seed in seeds:
        if seed is not None:
            setup_seed(seed)
        print("seed: {}".format(seed))

        model = build_model(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config.lr,
                                      weight_decay=config.weight_decay)
        scheduler_warmup = get_scheduler(optimizer,config)
        writer = SummaryWriter(logdir=log_path)

        train_start_time = time.strftime('%Y-%m-%d %H:%M:%S')
        best_test = None
        best_val = float('inf')
        best_epoch = 0
        patience = 0
        for epoch in range(1,config.n_epochs+1):
            train(model,dataLoader['train'],optimizer,loss_fn,epoch,writer,metrics,config,seed)
            val_results = evaluate(model,dataLoader['valid'],optimizer,loss_fn,epoch,writer,save_path,metrics,config,
                                   seed)
            if config.is_test is not None:
                test(model,dataLoader['test'],optimizer,loss_fn,epoch,writer,metrics,config,seed)
            if val_results is not None and val_results.get('MAE') is not None:
                if val_results['MAE'] < best_val:
                    best_val = val_results['MAE']
                    best_epoch = epoch
                    patience = 0
                    save_best_model(save_path,seed,model,optimizer,epoch)
                else:
                    patience += 1
                    if patience >= config.early_stop:
                        print('Early stopping at epoch {}'.format(epoch))
                        break
            scheduler_warmup.step()
        # 回滚到验证集最优权重，并用其重新评估测试集
        load_best_model(save_path,seed,model)
        print('Rolled back to best validation weights (epoch {}, best MAE {:.4f})'.format(best_epoch,best_val))
        if config.is_test is not None:
            best_test = test(model,dataLoader['test'],optimizer,loss_fn,best_epoch,writer,metrics,config,seed)
        writer.close()

        if best_test is not None:
            write_test_csv(best_test,train_start_time,log_path,config.datasetName,seed)
            all_seed_results.append(best_test)

    # 平均所有 seed 的结果
    if len(all_seed_results) > 1:
        avg_results = average_metrics(all_seed_results)
        print('==== 平均结果（{} 个 seed）===='.format(len(all_seed_results)))
        print(avg_results)
        summary_csv = os.path.join(log_path,'seed_avg_metrics.csv')
        write_summary_csv(avg_results,summary_csv,config.datasetName)
        print('平均结果已保存到 {}'.format(summary_csv))


def train(model,train_loader,optimizer,loss_fn,epoch,writer,metrics,config=None,seed=None):
    train_pbar = tqdm(enumerate(train_loader))
    losses = AverageMeter()

    y_pred, y_true = [], []

    model.train()
    for cur_iter, data in train_pbar:
        img, audio, text, audio_text, vision_text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device), data['audio_text'].to(device), data['vision_text'].to(device)
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        batchsize = img.shape[0]

        # output, _, _, _, _, _, _, _, _  = model(img, audio, text, audio_text, vision_text)
        output = model(img, audio, text, audio_text, vision_text)
        loss = loss_fn(output, label)

        losses.update(loss.item(), batchsize)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

        train_pbar.set_description('train')
        train_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                'loss': '{:.5f}'.format(losses.value_avg),
                                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    train_results = metrics(pred, true)
    print('train: ', train_results)
    train_mae.append(train_results['MAE'])

    if config is not None:
        log_path = os.path.join(".","log",config.project_name)
        write_log_line(log_path,'train',epoch,train_results,seed)

    writer.add_scalar('train/loss', losses.value_avg, epoch)


def evaluate(model,eval_loader,optimizer,loss_fn,epoch,writer,save_path,metrics,config=None,seed=None):
    test_pbar = tqdm(enumerate(eval_loader))

    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for cur_iter, data in test_pbar:
            img, audio, text, audio_text, vision_text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device), data['audio_text'].to(device), data['vision_text'].to(device)
            
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            output = model(img, audio, text, audio_text, vision_text)

            loss = loss_fn(output, label)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            losses.update(loss.item(), batchsize)

            test_pbar.set_description('eval')
            test_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                   'loss': '{:.5f}'.format(losses.value_avg),
                                   'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        print('valid: ',test_results)

        if config is not None:
            log_path = os.path.join(".","log",config.project_name)
            write_log_line(log_path,'val',epoch,test_results,seed)

        writer.add_scalar('evaluate/loss', losses.value_avg, epoch)
        # if epoch == 28:
        #     save_model(save_path, epoch, model, optimizer)

        return test_results


def test(model,test_loader,optimizer,loss_fn,epoch,writer,metrics,config=None,seed=None):

    test_pbar = tqdm(enumerate(test_loader))

    losses = AverageMeter()
    y_pred, y_true = [], []
    # h_at_list = []
    # h_a_list = []
    # h_fusion_a_list = []
    # h_vt_list = []
    # h_v_list = []
    # h_fusion_v_list = []
    # h_t_list = []
    # h_fusion_t_list = []
    model.eval()
    with torch.no_grad():
        for cur_iter, data in test_pbar:
            img, audio, text, audio_text, vision_text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device), data['audio_text'].to(device), data['vision_text'].to(device)
            raw_text, id, labels = data['raw_text'], data['id'], data['labels']['M'].to(device).view(-1, 1)
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            # output, h_at, h_a, h_fusion_a, h_vt, h_v, h_fusion_v, h_t, h_fusion_t = model(img, audio, text, audio_text, vision_text)
            output = model(img, audio, text, audio_text, vision_text)

            loss = loss_fn(output, label)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())
            # h_at_list.append(h_at.cpu())
            # h_a_list.append(h_a.cpu())
            # h_fusion_a_list.append(h_fusion_a.cpu())
            # h_vt_list.append(h_vt.cpu())
            # h_v_list.append(h_v.cpu())
            # h_fusion_v_list.append(h_fusion_v.cpu())
            # h_t_list.append(h_t.cpu())
            # h_fusion_t_list.append(h_fusion_t.cpu())
            # x_list.append(x.cpu())
            losses.update(loss.item(), batchsize)

            test_pbar.set_description('test')
            test_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                   'loss': '{:.5f}'.format(losses.value_avg),
                                   'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])})

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)
        print('test: ',test_results)

        if config is not None:
            log_path = os.path.join(".","log",config.project_name)
            write_log_line(log_path,'test',epoch,test_results,seed)

        writer.add_scalar('test/loss', losses.value_avg, epoch)

        # present_dict = {'h_at':h_at_list, 'h_a':h_a_list, 'h_fusion_a': h_fusion_a_list,
        #                 'h_vt':h_vt_list, 'h_v': h_v_list, 'h_fusion_v': h_fusion_v_list,
        #                 'h_t': h_t_list, 'h_fusion_t':h_fusion_t_list}

        return test_results


if __name__ == '__main__':
    main()
