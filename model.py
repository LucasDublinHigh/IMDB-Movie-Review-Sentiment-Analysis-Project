import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer

# tokenizer
tokenizer = get_tokenizer("basic_english")

# -----------------------------
# MODEL
# -----------------------------
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return torch.sigmoid(out).squeeze()

# -----------------------------
# TEXT PIPELINE
# -----------------------------
def text_pipeline(text, vocab):
    return vocab(tokenizer(text))