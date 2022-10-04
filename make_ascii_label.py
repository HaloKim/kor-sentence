import pandas as pd
from tokenizers import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
tokenizer = Tokenizer.from_file("tokenizer-trained.json")

df = pd.read_csv('상담사례답변모음.csv').drop(['row'], axis=1)
df2 = pd.read_csv('상담사례답변모음2.csv').dropna()
df = pd.concat([df, df2], ignore_index=True)
df.drop_duplicates(keep='first', inplace=True)
df = df.reset_index(drop=True)

for i in range(len(df)):
    df.loc[i, 'text'] = re.sub('[^ 가-힣ㅏ-ㅣㄱ-ㅎ]', '', df.loc[i, 'text']).strip()

# plt.hist([len(l) for l in df['text']], bins=1000)
# plt.xlabel('length of samples')
# plt.ylabel('number of samples')
# plt.title('Space Distribution')
# plt.show()

df['labels'] = None
max_len = 250
for i in tqdm(range(len(df))):
    labels = [0]
    for j in range(len(df.text[i])-1):
        if df.text[i][j+1] == ' ':
            labels += [1] # 그대로
        else:
            labels += [0] # 다음문자가 띄어쓰기
    while len(labels)-1 < max_len:
        labels += [0]
    df.labels[i] = labels[1:]
df = df[df.text.str.len() <= max_len].reset_index(drop=True)

tokenized = []
pad = []
for i in tqdm(range(len(df))):
    tmp = [ord(text) for text in df.text[i]]
    if len(tmp) < max_len:
        pad.append(len(tmp))
        while len(tmp) < max_len:
            tmp += [0]
    else:
        pad.append(0)
    tokenized.append(tmp)
df['tokenized'], df['pad'] = tokenized, pad
df[(df['pad'] > 0) & (df['text'].str.len() >= 10)].reset_index(drop=True).to_csv('pre.csv', index=False, encoding='utf-8')

tmp = df.tokenized[0]
for i in df.tokenized[1:]:
    tmp += i
tmp = np.array(tmp, ndmin=1)
print('VOCAB : ', np.unique(tmp, return_counts=True))
