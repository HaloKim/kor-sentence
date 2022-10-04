from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd

# Initialize a tokenizer
tokenizer = Tokenizer(Unigram())

trainer = UnigramTrainer(vocab_size=50000, special_tokens=["<pad>", "<unk>"])
# tokenizer.pre_tokenizer = Whitespace()

df = pd.read_csv('상담사례답변모음.csv').drop(['row'], axis=1)
df2 = pd.read_csv('상담사례답변모음2.csv').dropna()
df = pd.concat([df, df2], ignore_index=True).reset_index(drop=True)
df = df['text']
df.to_csv("/home/halo/PycharmProjects/상담사례모음/for_vocab.txt", index=False)
tokenizer.train(['/home/halo/PycharmProjects/상담사례모음/for_vocab.txt'], trainer)
tokenizer.save("./tokenizer-trained.json")
print("FIN")
