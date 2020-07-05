import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from transformers import DistilBertModel, DistilBertTokenizer

MAX_LEN = 128
BATCH = 32

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        d = pd.read_csv(path).drop(['selected_text', 'textID'], axis=1, errors="ignore").dropna()
        d["sentiment"] = d["sentiment"].map({'neutral': 0, 'negative': 1, 'positive': 2})
        self.val = d.values
        # self.val = self.val[:512]
        self.tok = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def __len__(self):
        return len(self.val)

    def __getitem__(self, index):
        x, y = self.val[index]
        x = self.tok.tokenize(x)
        x += ["[PAD]"] * (MAX_LEN - len(x))
        x = self.tok.convert_tokens_to_ids(x)
        return np.array(x), y

def rune(gen, train=True):
    ls = 0
    ll = 0
    la = 0
    with torch.set_grad_enabled(train):
        for x, y in gen:
            x = x.to('cuda')
            y = y.to('cuda')
            if train:
                opt.zero_grad()
            outputs = bert(x)
            outputs = fine(outputs[0].view(-1, MAX_LEN * 768))
            loss = crit(outputs, y)
            if train:
                loss.backward()
                opt.step()
            ls += loss.detach().item()
            la += y.eq(outputs.max(1)[1]).sum().item()
            ll += len(y)
    return ls / ll, la / ll

bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
fine = torch.nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(MAX_LEN * 768, 3)
)
bert.to('cuda')
fine.to('cuda')
training_generator = torch.utils.data.DataLoader(Dataset("tweet/train.csv"), batch_size=BATCH, pin_memory=True, shuffle=True)
test_generator = torch.utils.data.DataLoader(Dataset("tweet/test.csv"), batch_size=BATCH, pin_memory=True)
crit = nn.CrossEntropyLoss()
opt = optim.Adam(list(bert.parameters()) + list(fine.parameters()), lr=0.0001)

for e in range(100):
    bert.train()
    fine.train()
    loss, acc = rune(training_generator, True)
    print(f"Train {e}, {loss:.5f}, {acc:.2f}")
    bert.eval()
    fine.eval()
    loss, acc = rune(test_generator, False)
    print(f"Test  {e}, {loss:.5f}, {acc:.2f}")