from torch import nn
import torch
import  torch.nn.functional as F
import torch.optim as optim
from nltk import word_tokenize
from torchtext import data, datasets
import pandas as pd


def remove_punctuations(text):
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"

    for p in punctuations:
        text = text.replace(p, f' {p} ')

    text = text.replace('...', ' ... ')

    if '...' not in text:
        text = text.replace('..', ' ... ')

    return text


def tokens(sentence):
    return word_tokenize(remove_punctuations(sentence))


# def readTxt(path):
#     with open(path) as f:
#         text = f.readlines()
#
#     return list(map(lambda x:x.strip(), text))
#
#
# def writeCSV(path):
#     premise = readTxt(r"D:\DOWNLOAD\nli-master\nli-master\data\word_sequence\premise_snli_1.0_test.txt")
#     hypothesis = readTxt(r"D:\DOWNLOAD\nli-master\nli-master\data\word_sequence\hypothesis_snli_1.0_test.txt")
#     label = readTxt(r"D:\DOWNLOAD\nli-master\nli-master\data\word_sequence\label_snli_1.0_test.txt")
#     file = pd.DataFrame()
#     file['premise'] = premise
#     file['hypothesis'] = hypothesis
#     file['label'] = label
#     file.to_csv(path)
#
#
# writeCSV(r"D:\Desktop\ML\ESIM\test.csv")
# data = pd.read_csv(r"D:\Desktop\ML\ESIM\test.csv")
# print(data.loc[:3, 'premise'])
# print(data.loc[:3, 'hypothesis'])



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL = data.Field(sequential=False, use_vocab=False)
TEXT = data.Field(sequential=True, tokenize=tokens, lower=True)

train = data.TabularDataset(r'D:\Desktop\ML\ESIM\train.csv', format='csv', skip_header=True,
                            fields=[("id", None), ("premise", TEXT), ('hypothesis', TEXT),
                                    ('label', LABEL)])
test = data.TabularDataset(r'/kaggle/input/standfordsnl/test.csv', format='csv', skip_header=True,
                            fields=[("id", None), ("premise", TEXT), ('hypothesis', TEXT),
                                    ('label', LABEL)])

TEXT.build_vocab(train, vectors='glove.6B.100d')
train_iter = data.BucketIterator(train, batch_size=128, sort_key=lambda x: len(x.premise),
                                 sort_within_batch=False, device=DEVICE)
test_iter = data.BucketIterator(test, batch_size=128, sort_key=lambda x: len(x.premise),
                                 sort_within_batch=False, device=DEVICE)


class ESIM(nn.Module):
    def __init__(self):
        super(ESIM, self).__init__()
        self.dropout = 0.2
        self.embed_dim = 100
        self.embeddings = nn.Embedding(len(TEXT.vocab), self.embed_dim)
        # self.bn_embed = nn.BatchNorm1d(self.embed_dim)

        self.lstm1 = nn.LSTM(self.embed_dim, 128, 3, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(128 * 8, 128, 3, batch_first=True, bidirectional=True)

        self.fn = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 16),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(16, 3),
            nn.Softmax()
        )

    def soft_alignment_layer(self, x1, x2):
        return torch.matmul(x1.permute(2, 0, 1), x2.permute(2, 1, 0))

    def Local_inference_collected_over_sequences(self, x1, x2, weight):
        return torch.matmul(F.softmax(weight, dim=-1), x2.permute(2, 0, 1)), \
               torch.matmul(F.softmax(weight.transpose(1, 2), dim=-1), x1.permute(2, 0, 1))

    def Enhancement_of_local_inference_information(self, x1, x2):
        return torch.cat([x1, x2, x1 - x2, x1 * x2], -1)

    def Pooling(self, x):
        return  torch.cat([F.avg_pool1d(x.permute(1, 2, 0), x.size(0)).squeeze(-1),
                          F.max_pool1d(x.permute(1, 2, 0), x.size(0)).squeeze(-1)], 1)

    def forward(self, *x):
        premise, hypothesis = x[0], x[1]
        # premise = self.bn_embed(self.embeddings(premise)
        #                         .transpose(1, 2)
        #                         .contiguous()).transpose(1, 2)
        # hypothesis = self.bn_embed(self.embeddings(hypothesis)
        #                         .transpose(1, 2)
        #                         .contiguous()).transpose(1, 2)
        # print(premise.shape, hypothesis.shape)
        premise = self.embeddings(premise)
        hypothesis = self.embeddings(hypothesis)
        premise, _ = self.lstm1(premise)
        hypothesis, _ = self.lstm1(hypothesis)
        # print(premise.shape, hypothesis.shape)
        attention = self.soft_alignment_layer(premise, hypothesis)

        premise_attn, hypothesis_attn = self.Local_inference_collected_over_sequences(premise, hypothesis, attention)
        # print(premise_attn.shape, hypothesis_attn.shape)

        premise = self.Enhancement_of_local_inference_information(premise,
                                                                  premise_attn.permute(1, 2, 0))
        hypothesis = self.Enhancement_of_local_inference_information(hypothesis,
                                                                     hypothesis_attn.permute(1, 2, 0))
        # print(premise.shape, hypothesis.shape)

        premise, _ = self.lstm2(premise)
        hypothesis, _ = self.lstm2(hypothesis)

        premise = self.Pooling(premise)
        hypothesis = self.Pooling(hypothesis)
        # print(premise.shape, hypothesis.shape)

        v = torch.cat([premise, hypothesis], -1)
        # print(v.shape)
        v = self.fn(v)
        return v


model = ESIM()
model.embeddings.weight.data.copy_(TEXT.vocab.vectors)
model.to(DEVICE)

criterition = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

n_epoch = 15

for epoch in range(n_epoch):
    for bacth_idx, batch in enumerate(train_iter):
        premise = batch.premise.to(DEVICE)
        hypothsis = batch.hypothesis.to(DEVICE)
        label = batch.label.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(premise, hypothsis)
        loss = criterition(outputs, label)
        loss.backward()
        optimizer.step()

        if (bacth_idx + 1) % 1000 == 0:
            _, y_pred = torch.max(outputs, -1)
            acc = torch.mean((torch.tensor(y_pred == label, dtype=torch.float)))
            print('epoch: %d \t batch_id : %d \t loss: %.4f \t train_acc: %.4f'
                  % (epoch, bacth_idx + 1, loss, acc))
