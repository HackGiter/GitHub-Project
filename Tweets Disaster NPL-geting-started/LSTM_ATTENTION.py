import torch
from torchtext import data, datasets
from torch import nn
import pandas as pd
from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords, brown, wordnet
import torch.nn.functional as F
import nltk
import re
import numpy as np

# train = pd.read_csv('D:\\DOWNLOAD\\nlp-getting-started\\train.csv')
# test = pd.read_csv('D:\\DOWNLOAD\\nlp-getting-started\\test.csv')

# train1, val = train_test_split(train, test_size=0.2)
# train1.to_csv("D:\\DOWNLOAD\\nlp-getting-started\\train1.csv", index=False)
# val.to_csv("D:\\DOWNLOAD\\nlp-getting-started\\val.csv", index=False)


# nltk.download('stopwords')
# nltk.download('universal_tagset')
# nltk.download('brown')
# nltk.download('wordnet')

# print(train.shape[0], test.shape[0])
#
# labels = train['target'].values
# idx = len(labels)


# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

# Define a function to remove emojis
def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert

# Define a function to remove punctuations


def remove_punctuations(text):
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"

    for p in punctuations:
        text = text.replace(p, f' {p} ')

    text = text.replace('...', ' ... ')

    if '...' not in text:
        text = text.replace('..', ' ... ')

    return text


def fil_stopwords(words):
    return [word for word in words if word not in stopwords.words('english')]


# train["text"] = train["text"].apply(remove_emoji)
#
# train['text'] = train['text'].apply(remove_punctuations)
#
# train['token_words'] = train['text'].apply(word_tokenize)
#
# train['token_words'] = train['token_words'].apply(fil_stopwords)
#
lancaster_stemmer = LancasterStemmer()


def get_stem(words):
    return [lancaster_stemmer.stem(word) for word in words]


# train['stem_words'] = train['token_words'].apply(get_stem)

# Converting part of speeches to wordnet format.
def simplify_tag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# part of speech
# POS tagging :part-of-speech tagging , or word classes or lexical categories
# 所以会出现NN，IN
# train['pos_tag'] = train['stem_words'].apply(pos_tag, args=("universal",))
# train['pos_tag'] = train['stem_words'].apply(pos_tag)
# train['pos_tag'] = train['pos_tag'].apply(lambda x: [(word, simplify_tag(tag)) for word, tag in x])
# train['pos_tag'] = train['stem_words'].apply(brown.tagged_words, simplify_tags=True)

wordnet_lemmatizer = WordNetLemmatizer()


def get_lemma(words):
    return [wordnet_lemmatizer.lemmatize(word, pos=tag) for word, tag in words]


# print(train.sample(10))
# train['lemma'] = train['pos_tag'].apply(get_lemma)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def deal_with_tokens(text):
    text = remove_emoji(text)
    text = remove_punctuations(text)
    text = word_tokenize(text)
    text = fil_stopwords(text)
    text = get_stem(text)
    text = pos_tag(text)
    text = [(word, simplify_tag(tag)) for word, tag in text]
    text = get_lemma(text)
    return text


LABEL = data.Field(sequential=False, use_vocab=False)
TEXT = data.Field(sequential=True, tokenize=deal_with_tokens, lower=True)

# train,val = data.TabularDataset.splits(
#         path='./', train='train1.csv',validation='val.csv', format='csv',skip_header=True,
#         fields=[('id',None),('keyword',None),('location', None), ('text', TEXT), ('target', LABEL)])
train = data.TabularDataset('D:\\DOWNLOAD\\nlp-getting-started\\train.csv', format='csv', skip_header=True,
                            fields=[('id', None), ('keyword', None), ('location', None), ('text', TEXT),
                                    ('target', LABEL)])
test = data.TabularDataset('D:\\DOWNLOAD\\nlp-getting-started\\test.csv', format='csv', skip_header=True,
                           fields=[('id', None), ('keyword', None), ('location', None), ('text', TEXT)])

TEXT.build_vocab(train, vectors='glove.6B.100d')
len_vocab = len(TEXT.vocab)
train_iter = data.BucketIterator(train, batch_size=128, sort_key=lambda x: len(x.text),
                                 sort_within_batch=False, device=DEVICE)

# val_iter = data.BucketIterator(val, batch_size=128, sort_key=lambda x: len(x.text),
#                                  sort_within_batch=False, device=DEVICE)

test_iter = data.Iterator(dataset=test, batch_size=128, train=False,
                          sort=False, device=DEVICE)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(len_vocab, 100)
        # self.rnn = nn.RNN(100,128)
        # self.linear1 = nn.Sequential(
        #     nn.Linear(128, 40),
        #     nn.ReLU()
        # )
        # self.linear2 = nn.Linear(40, 2)
        # self.lstm = nn.LSTM(100, 128, 3, bidirectional=True)
        self.lstm = nn.LSTM(100, 128, 3, bidirectional=True, dropout=0.2)
        self.linear1 = nn.Sequential(
            nn.Linear(256, 60),
            nn.ReLU()
        )
        self.linear2 = nn.Linear(60, 2)
        # self.linear = nn.Linear(256, 2)

    def self_attention(self, lstm_output, final_hidden):
        final_hidden = final_hidden.view(-1, 256, 3)
        lstm_output = lstm_output.permute(1, 0, 2)  #[batch_size, sequence_length, hidden_num * 2]
        attn_weight = torch.bmm(lstm_output, final_hidden)
        attn_weight = torch.sum(attn_weight, dim=2)
        attn_weight = F.softmax(attn_weight, dim=1)
        context = torch.bmm(lstm_output.transpose(1, 2), attn_weight.unsqueeze(2))
        return context, attn_weight

    def forward(self, x):
        seq_len, batch_size = x.shape  # x.shape=(seq_len, batch_size)
        # print(seq_len, batch_size)
        vec = self.embedding(x)  # vec的维度为(seq_len, batch_size, 100)
        # output,hidden = self.rnn(vec) # RNN初始化的hidden如果不提供则默认为全0张量
        # output的维度 (seq_len, batch, hidden_size 128)
        # hidden的维度 (numlayers 1, batch_size, hidden_size 128)
        # out = self.linear(hidden.view(batch_size, -1))
        output, (hidden, cn) = self.lstm(vec) #hidden_shape:[batch_size, layer_num * 2, hidden_num]
        # print(output.shape)
        # print(hidden.shape)
        # print(output[:, -1, :].shape)
        # out = self.linear1(hidden.view(batch_size, -1))
        context, attention = self.self_attention(output, hidden)
        # out = self.linear1(output[-1, :, :])
        out = self.linear1(context.squeeze(2))
        out = self.linear2(out)
        # out = self.linear(output[-1, :, :])
        return out
        # out的维度 (batch_size, 5)


model = Model()
model.embedding.weight.data.copy_(TEXT.vocab.vectors)
model.to(DEVICE)

criterition = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

n_epoch = 20

for epoch in range(n_epoch):
    for bacth_idx, batch in enumerate(train_iter):
        data = batch.text.to(DEVICE)
        target = batch.target.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterition(outputs, target)
        loss.backward()
        optimizer.step()

        if (bacth_idx + 1) % 30 == 0:
            _, y_pred = torch.max(outputs, -1)
            acc = torch.mean((torch.tensor(y_pred == target, dtype=torch.float)))
            print('epoch: %d \t batch_id : %d \t loss: %.4f \t train_acc: %.4f'
                  % (epoch, bacth_idx + 1, loss, acc))

outcome = np.array([])

for bacth_idx, batch in enumerate(test_iter):
    data = batch.text.to(DEVICE)
    optimizer.zero_grad()
    outputs = model(data)
    outcome = np.append(outcome, torch.max(outputs, 1)[1].cpu().numpy())

test = pd.read_csv('D:\\DOWNLOAD\\nlp-getting-started\\sample_submission.csv')
test['target'] = outcome
test.to_csv("D:\\DOWNLOAD\\nlp-getting-started\\outcomes.csv")
