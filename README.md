# kor-sentence

한글 띄어쓰기 모델 만들어보기 pytorch + BiLSTM

# 학습에 쓰인 데이터

|text|labels|tokenized|
|------|---|---|
|나는 파이토치 모델을 만들었다|010000100010000|...|

라벨글자별로 다음이 띄어쓰기면 1 아니면 0

# 모델

```python
class Net(nn.Module):
    def __init__(self, device, batch_size, input_size=max_len, hidden_size=max_len, num_layers=3):
        super(Net, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.vocab_size = 55193 + 1
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=input_size,
            padding_idx=0
        )

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size*2, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(input_size * 2, max_len)

    def forward(self, x, seq_len):
        embed = self.embedding(x)
        packed_embed = pack_padded_sequence(embed, seq_len, batch_first=True)
        # Initialize hidden state, cell state with zeros
        h0 = torch.zeros(self.num_layers * 2, x.size()[0], self.hidden_size).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size()[0], self.hidden_size).requires_grad_().to(self.device)
        out, (final_hidden_state, final_cell_state) = self.lstm(packed_embed, (h0, c0))
        out, (final_hidden_state, final_cell_state) = self.lstm2(out)
        output_unpacked, output_lengths = pad_packed_sequence(out, batch_first=True)
        h_t = output_unpacked[:, -1, :]
        out = self.dropout(h_t)
        out = self.fc(out)
        return out
```

# 결과

ACC 68

```
"술마시다물마시다커피마시다처럼쓰는것이의미상좀더적확하다고볼수는있겠으나그렇다고하여먹다를쓸수없는것은아닙니다"
"술 마 시 다물마 시 다커 피마시 다 처럼 쓰 는것 이의미상 좀 더적확 하 다고 볼수는있 겠으나 그 렇다 고 하여 먹다를 쓸수 없 는것 은 아닙니다 "
```

문제가 많다
