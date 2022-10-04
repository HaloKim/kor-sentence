import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import toascii

max_len = 250
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


model = Net(DEVICE, 1).cuda()
model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()
sigmoid = torch.sigmoid


def predict(model, x, len_seq):
    """학습된 모델로 정답 파일 예측"""
    logit = model(x, [torch.tensor(len_seq).int()])
    return logit


with torch.no_grad():
    text = "술마시다 물마시다 커피 마시다처럼 쓰는 것이 의미상 좀 더 적확하다고 볼 수는 있겠으나 그렇다고 하여 먹다를 쓸 수 없는 것은 아닙니다 한국어의 먹다는 액체뿐만 아니라 기체에도 사용됨을 참고하시기 바랍니다"
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
