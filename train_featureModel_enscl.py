# #测试数据的获取
# import os
# from myModel.dataProcess_Emotion import emotionDataset
# from torch.utils.data import DataLoader
#
#
#
# # 准备训练集数据
# data_path = os.path.join('./data', 'ywy_data.csv')
# emo_dataset = emotionDataset(csv_file_path=data_path, model_path='./bert-base-uncased', max_len=256)
# emo_data_len = emo_dataset.__len__()
# emo_dataloader = DataLoader(emo_dataset, batch_size=4, shuffle=True)
# emo_dataloader_len = emo_dataloader.__len__()
#
#
# for data in emo_dataloader:
#     inputs = data['input_ids']  # [batch_size,max_len]
#     labels = data['label']



import logging
from myModel.en_FeatureModel import BertCNNFeature_en
from myModel.FeatureModel import BertCNNFeature

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
from myModel.SCL import SupConLoss

parser = argparse.ArgumentParser()
parser.add_argument('--model_save_path', default='./checkpoint/Emotion/', type=str, help='model save path')
parser.add_argument('--model_pt_path', default='./checkpoint/Emotion/1/', type=str, help='model pt path')
parser.add_argument('--model_pt', default='last.pt', type=str, help='last.pt or best.pt  choose pt file')

parser.add_argument('--epochs', default=10, type=int, help='data dir')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--max_len', default=256, type=int, help='max token length')

parser.add_argument('--data_dir', default='./data', type=str, help='data dir')
parser.add_argument('--train_data', default='en_train_data.csv', type=str, help='train data file name')
parser.add_argument('--test_data', default='en_test_data.csv', type=str, help='test data file name')

args = parser.parse_args()


def start_train(
        model_save_path,
        model_pt_path,
        model_pt,
        epochs,
        batch_size,
        max_len,
        data_dir,
        train_data,
        test_data
        ):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = BertCNNFeature_en(drop_out=True,model_path = './bert-base-uncased')
    contrastive_loss_fn = SupConLoss()
    #加载已有模型
    last_model = os.path.join(model_pt_path, model_pt)
    if os.path.exists(os.path.join(model_pt_path, model_pt)):
        loaded_paras = torch.load(last_model)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")
    else:
        logging.info("## 不存在已有模型，进行训练......")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.train()

    #创建保存路径
    i = 1
    save_pt_path = f"epoch{epochs}"
    while True:
        if not os.path.exists(os.path.join(model_save_path, save_pt_path)):
            break
        i += 1
        save_pt_path = f"epoch{epochs}_{i}"
    new_save_path = os.path.join(model_save_path, save_pt_path)
    if not os.path.exists(new_save_path):
        os.makedirs(new_save_path)
    loss_best_path = os.path.join(new_save_path, 'best.pth')
    lost_last_path = os.path.join(new_save_path, 'last.pth')


    # 准备训练集数据
    data_path = os.path.join(data_dir, train_data)
    emo_dataset = emotionDataset(csv_file_path=data_path, model_path = './bert-base-uncased',max_len=max_len)
    emo_data_len = emo_dataset.__len__()
    emo_dataloader = DataLoader(emo_dataset, batch_size=batch_size, shuffle=True)
    emo_dataloader_len = emo_dataloader.__len__()

    # 准备验证集数据
    test_data_path = os.path.join(data_dir, test_data)
    emo_test_dataset = emotionDataset(csv_file_path=test_data_path, model_path='./bert-base-uncased', max_len=max_len)
    emo_test_data_len =  emo_test_dataset.__len__()
    emo_test_dataloader = DataLoader(emo_test_dataset, batch_size=batch_size, shuffle=True)
    emo_test_dataloader_len = emo_dataloader.__len__()

    # 保存loss数据
    loss_epoch = []  # 存放每轮epoch的平均损失
    cur_min_loss = 999  # 记录最小loss

    for epoch in range(1, epochs + 1):
        torch.cuda.empty_cache()
        model.train()
        loss_iter = 0
        loop = tqdm((emo_dataloader), total=emo_dataloader_len)
        for data in loop:
            torch.cuda.empty_cache()
            inputs = data['input_ids'].to(device)  # [batch_size,max_len]
            attention_mask = data['attention_mask']  # [batch_size,max_len]
            labels = data['label'].to(device)  # # [batch_size]
            # review_text = data['review_text']
            loss, logits = model(input_data=inputs, labels=labels)
            features = inputs.unsqueeze(1)
            contrastive_loss = contrastive_loss_fn(features, labels)
            loss = loss + contrastive_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_iter += loss.item()
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(loss=loss.item())

        avg_loss = loss_iter / emo_data_len
        loss_epoch.append(avg_loss)
        if avg_loss < cur_min_loss:
            cur_min_loss = avg_loss
            # 保存最小loss模型
            torch.save(model.state_dict(), loss_best_path)#只保存模型权重参数，不保存模型结构
        #验证精确率

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            test_loop = tqdm((emo_test_dataloader), total=emo_test_dataloader_len)
            for t_data in test_loop:
                inputs = t_data['input_ids'].to(device)  # [batch_size,max_len]
                attention_mask = t_data['attention_mask']  # [batch_size,max_len]
                labels = t_data['label'].to(device)  # # [batch_size]
                loss, logits = model(input_data=inputs, labels=labels)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = correct / total
        logging.info(f'Epoch {epoch}/{epochs}, Avg Loss: {avg_loss}, Accuracy: {val_accuracy}')


    # 保存最后一轮的模型权重
    torch.save(model.state_dict(), lost_last_path)

    # 存储loss曲线，以防丢失
    # 保存DataFrame到CSV文件
    df = pd.DataFrame({'loss': loss_epoch})
    # 指定CSV文件名，并确保文件名唯一
    file_name = "loss_values.csv"
    i = 1
    while True:
        if not os.path.exists(file_name):
            break
        file_name = f"loss_values_{i}.csv"
        i += 1
    df.to_csv(file_name, index=False)

    # 绘制loss曲线，并且保存曲线图
    utils.plot_loss(loss_epoch, label="loss")
    print("min loss: {0}".format(cur_min_loss))


if __name__ == '__main__':
    start_train(
        model_save_path=args.model_save_path,
        model_pt_path=args.model_pt_path,
        model_pt=args.model_pt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        data_dir=args.data_dir,
        train_data=args.train_data,
        test_data=args.test_data
        )