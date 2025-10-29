import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class ws_Feature(nn.Module):
    def __init__(self,num_labels = 2,squence_length = 256,model_path = '../bert-base-chinese',drop_out=True):
        super(ws_Feature, self).__init__()
        self.squence_length = squence_length
        # BERT 模型
        self.bert = BertModel.from_pretrained(model_path)

        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=32, kernel_size=2) #2-gram
        self.conv2 = nn.Conv1d(in_channels=768, out_channels=64, kernel_size=3) #3-gram
        self.conv3 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=4) #4-gram

        self.dropout = nn.Dropout(0.1)
        self.classifier =nn.Linear(992, num_labels)

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        # BERT 输出
        outputs = self.bert(input_data, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)  词
        word_embeding = last_hidden_state[:,1:,:] #只取CLS之后的

        # 调整输入数据的形状以适应Conv1d
        x = word_embeding.permute(0, 2, 1)

        # 应用第一个卷积层
        x1 = self.conv1(x)
        x1 = F.relu(x1)
        x1 = F.max_pool1d(x1, x1.size(2))

        # 应用第二个卷积层
        x2 = self.conv2(x)
        x2 = F.relu(x2)
        x2 = F.max_pool1d(x2, x2.size(2))

        # 应用第三个卷积层
        x3 = self.conv3(x)
        x3 = F.relu(x3)
        x3 = F.max_pool1d(x3, x3.size(2))

        # 将所有卷积层的输出拼接起来
        x_concat = torch.cat((x1, x2, x3), dim=1).squeeze(-1)

        pooler_output = outputs.pooler_output  # (batch_size, hidden_size)  句

        combined_features = torch.cat((pooler_output, x_concat), dim=1)

        return combined_features


