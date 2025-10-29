
'''
import pandas as pd
data_list = pd.read_csv("../data/weibo_senti_100k(1).csv").values.tolist()
import matplotlib.pyplot as plt
# 计算每个句子的长度
sentence_lengths = [len(sentence[1]) for sentence in data_list]
# 绘制句子长度分布图
plt.figure(figsize=(10, 6))
plt.hist(sentence_lengths, bins=range(1,max(sentence_lengths)+2), align='left', edgecolor='black', alpha=0.7)
plt.title('Sentence Length Distribution')
plt.xlabel('Sentence Length')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.xticks(list([1,30,60,90,120,150]))
plt.show()
print(sentence_lengths)
'''


import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
import pandas as pd

# 定义Bert模型和标签数
model_name = '../bert-base-chinese'
data_dir = '../data'
num_labels = 2  # 'O' ‘1’
# 数据集类
class emotionDataset(Dataset):
    def __init__(self,csv_file_path,model_path='../bert-base-chinese',max_len=128):
        '''
        csv_file_path: 数据集路径
        model_name： 预训练模型路径
        max_len: token最大长度
        '''
        self.data = pd.read_csv(csv_file_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def data_process(self): #处理长度超过max_len的评论以及处理杂乱的评论
        pass

    def __getitem__(self, idx):
        label = self.data.loc[idx, 'label']
        review = self.data.loc[idx, 'review']

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

