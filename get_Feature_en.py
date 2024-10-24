#生成数据的特征和标签文件

import logging

from myModel.word_sentence_Feature import ws_Feature

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)
import torch
torch.cuda.empty_cache()
import argparse
import os
from torch.utils.data import DataLoader

from tqdm import tqdm
from myModel.dataProcess_Emotion import emotionDataset
from  myModel import  utils
import pandas as pd
from myModel.ws_Feature_en import ws_Feature_en


parser = argparse.ArgumentParser()
parser.add_argument('--model_save_path', default='../checkpoint/Emotion/', type=str, help='model save path')
parser.add_argument('--model_pt_path', default='../checkpoint/Emotion/epoch10/', type=str, help='model pt path')
parser.add_argument('--model_pt', default='best.pth', type=str, help='last.pt or best.pt  choose pt file')

parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--max_len', default=256, type=int, help='max token length')

parser.add_argument('--data_dir', default='../data', type=str, help='data dir')
parser.add_argument('--train_data', default='Newdata.csv', type=str, help='data ')

args = parser.parse_args()


def start_train(
        model_save_path,
        model_pt_path,
        model_pt,
        batch_size,
        max_len,
        data_dir,
        train_data
        ):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = ws_Feature_en(drop_out=True,model_path = '../bert-base-uncased')

    #加载已有模型
    last_model = os.path.join(model_pt_path, model_pt)
    if os.path.exists(os.path.join(model_pt_path, model_pt)):
        loaded_paras = torch.load(last_model)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")
    else:
        logging.info("## 不存在已有模型，进行训练......")

    model = model.to(device)

    # 准备需要提取特征的数据
    data_path = os.path.join(data_dir, train_data)
    emo_dataset = emotionDataset(csv_file_path=data_path, model_path = '../bert-base-uncased',max_len=max_len)
    emo_data_len = emo_dataset.__len__()
    emo_dataloader = DataLoader(emo_dataset, batch_size=batch_size, shuffle=True)
    emo_dataloader_len = emo_dataloader.__len__()

    # 保存loss数据
    loss_epoch = []  # 存放每轮epoch的平均损失
    cur_min_loss = 999  # 记录最小loss


    torch.cuda.empty_cache()
    model.eval()
    loop = tqdm((emo_dataloader), total=emo_dataloader_len)
    features_list = []
    labels_list = []

    # 初始化一个空的DataFrame，用于存储所有批次的数据
    all_features_df = pd.DataFrame()
    all_labels_df = pd.DataFrame()

    for data in loop:
        torch.cuda.empty_cache()
        inputs = data['input_ids'].to(device)  # [batch_size,max_len]
        labels = data['label'].to(device)  # # [batch_size]
        # 获取特征向量
        with torch.no_grad():  # 确保不会计算梯度
            features = model(input_data=inputs)  # features [batch size, 992]

        # 将特征向量和标签从GPU移到CPU并转换为NumPy数组
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # 将特征和标签转换为DataFrame
        features_df = pd.DataFrame(features_np)
        labels_df = pd.DataFrame(labels_np, columns=['Label'])

        # 将当前批次的特征和标签添加到总DataFrame中
        all_features_df = pd.concat([all_features_df, features_df], ignore_index=True)
        all_labels_df = pd.concat([all_labels_df, labels_df], ignore_index=True)

    # 合并特征和标签DataFrame
    all_data_df = pd.concat([all_features_df, all_labels_df], axis=1)

    # 保存到CSV文件
    all_data_df.to_csv('en_features.csv', index=False)

if __name__ == '__main__':
    start_train(
        model_save_path=args.model_save_path,
        model_pt_path=args.model_pt_path,
        model_pt=args.model_pt,
        batch_size=args.batch_size,
        max_len=args.max_len,
        data_dir=args.data_dir,
        train_data=args.train_data,
        )