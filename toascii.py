import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigmoid = nn.Sigmoid()


def asci(text, max_len):
    tmp = [ord(i)for i in text]
    while len(tmp) < max_len:
        tmp += [0]
    tmp = torch.tensor(tmp).to(DEVICE).reshape(1, -1)
    return tmp
