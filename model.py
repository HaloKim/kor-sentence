import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorchtools import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
import toascii

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
max_len = 250


def preprocess(batch_size):
    df = pd.read_csv('pre.csv', encoding='utf-8').sort_values(by='pad', ascending=False)
    df['tokenized'] = df['tokenized'].str.strip('[]').str.split(',').apply(lambda x: [int(i) for i in x])
    df['labels'] = df['labels'].str.strip('[]').str.split(',').apply(lambda x: [int(i) for i in x])
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True)
    train_df = train_df.sort_values(by='pad', ascending=False).reset_index(drop=True)
    val_df = val_df.sort_values(by='pad', ascending=False).reset_index(drop=True)
    train_data = torch.utils.data.TensorDataset(torch.tensor(train_df.tokenized),
                                                torch.tensor(train_df.labels, dtype=float),
                                                torch.tensor(train_df['pad']))
    val_data = torch.utils.data.TensorDataset(torch.tensor(val_df.tokenized),
                                              torch.tensor(val_df.labels, dtype=float),
                                              torch.tensor(val_df['pad']))
    train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_iter = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_iter, val_iter


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


def zero_loss(seq):
    zero = torch.zeros(len(seq), max_len)
    for i, number in enumerate(seq):
        zero[i][number:] = -torch.tensor(float('Inf'))
    return zero.to(DEVICE)


def train(model, optimizer, scheduler, train_iter):
    model.train()
    for batch_index, (texts, labels, seq) in enumerate(train_iter):
        x, y = texts.to(DEVICE), labels.to(DEVICE, dtype=torch.float32)
        optimizer.zero_grad()

        logit = model(x, seq)
        logit = logit + zero_loss(seq.detach().cpu())
        logit = torch.sigmoid(logit)
        loss = F.cross_entropy(logit, y)

        loss.backward()
        optimizer.step()
    # scheduler.step(loss)  # you can set it like this!


def evaluate(model, val_iter, sigmoid):
    """evaluate model"""
    model.eval()
    corrects, total_loss = 0, 0
    for batch_index, (texts, labels, seq) in enumerate(val_iter):
        x, y = texts.to(DEVICE), labels.to(DEVICE, dtype=torch.float32)
        logit = model(x, seq)
        logit = logit + zero_loss(seq.detach().cpu())
        logit = F.sigmoid(logit)
        loss = F.cross_entropy(logit, y)
        total_loss += loss.item()
        corrects += torch.eq(sigmoid(logit).round(), y.data).sum().item()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / (size * max_len)
    return avg_loss, avg_accuracy


def predict(model, x, len_seq):
    """학습된 모델로 정답 파일 예측"""
    model.eval()
    logit = model(x, [torch.tensor(len_seq).int()])
    return logit


best_val_loss = None
sigmoid = torch.sigmoid
EPOCHS = 100
batch = 128
model = Net(device=DEVICE, batch_size=batch).to(DEVICE)
optimizer = torch.optim.RMSprop(model.parameters(), lr=5e-1)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
early_stopping = EarlyStopping(patience=20, verbose=True)

for e in range(1, EPOCHS + 1):
    train_batch, val_batch = preprocess(batch_size=batch)
    train(model, optimizer, scheduler, train_batch)
    with torch.no_grad():
        val_loss, val_accuracy = evaluate(model, val_batch, sigmoid)

    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))

    if not best_val_loss or val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    with torch.no_grad():
        text = "환원하다를물질적경제적인것에만사용할수있는것은아닙니다개인의문제로환원하다처럼쓸수도있습니다"
        pred = predict(model, toascii.asci(text, max_len), len(text))
        pred = sigmoid(pred).round()
        text = [i for i in text]
        result = []
        for index, label in enumerate(pred):
            a = torch.eq(label, torch.ones(max_len).to(DEVICE)).detach().cpu()
            for i, tex in enumerate(text):
                if a[i]:
                    result.append(tex)
                else:
                    result.append(tex + ' ')
        print("".join(result))
