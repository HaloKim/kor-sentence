import pandas as pd
import re

df = pd.read_csv('상담사례답변모음.csv')
df.text.to_csv('for_vocab.txt', index=False, header=False)
