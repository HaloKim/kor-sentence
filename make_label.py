import pandas as pd
from tokenizers import Tokenizer
import numpy as np
tokenizer = Tokenizer.from_file("./tokenizer-trained.json")


df = pd.read_csv('상담사례답변모음.csv')
df['labels'] = None
for i in range(len(df)):
    labels = [0]
    tmp = tokenizer.encode(i).ids
    for j in range(len(df.text[i])-1):
        if df.text[i][j+1] == ' ':
            labels += [1] # 그대로
        else:
            labels += [0] # 다음문자가 띄어쓰기
    while len(labels)-1 < 256:
        labels += [0]
    df.labels[i] = labels[1:]
df['tokenized'] = None
tokenized = []
for i in range(len(df)):
    tmp = tokenizer.encode(df.text[i]).ids
    while len(tmp) < 256:
        tmp += [0]
    tokenized.append(tmp)
df['tokenized'] = tokenized
df = df[df.text.str.len() <= 256].drop(['row'], axis=1).reset_index(drop=True)
df = df[df.text.str.len() >= 5]
test = [i for i in df.tokenized]
print(np.unique(np.array(test).reshape(-1), return_counts=True))
df.to_csv('pre.csv', index=False, encoding='utf-8')
