import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from model import SentimentModel, text_pipeline, tokenizer


# -----------------------------
# 1. CREATE DATASET
# -----------------------------
def create_dataset():
    positive_reviews = [
        "This movie was amazing and I loved it",
        "Absolutely fantastic film with great acting",
        "One of the best movies I have ever seen",
        "Incredible story and wonderful performances",
        "I really enjoyed this movie, it was great",
        "A brilliant masterpiece with stunning visuals",
        "The acting was superb and the story was touching",
        "I was completely engaged from start to finish",
        "A truly inspiring and uplifting film",
        "Everything about this movie was perfect",
        "The plot was exciting and full of surprises",
        "Great direction and excellent cinematography",
        "This film exceeded all my expectations",
        "An outstanding movie with a powerful message",
        "I would definitely watch this again",
        "The characters were well developed and relatable",
        "A very entertaining and enjoyable experience",
        "The soundtrack was beautiful and emotional",
        "A fantastic journey from beginning to end",
        "Highly recommend this movie to everyone",
        "It was a fun and heartwarming film",
        "The story kept me interested the whole time",
        "Amazing visuals and a compelling story",
        "This movie was beautifully made",
        "I absolutely loved every moment of this film",
        "The performances were incredibly strong",
        "A great film with lots of emotional depth",
        "The ending was satisfying and well done",
        "This movie was creative and unique",
        "A delightful and charming story"
    ]

    negative_reviews = [
        "This movie was terrible and boring",
        "I hated this film, it was awful",
        "Worst movie I have ever watched",
        "Very bad acting and poor storyline",
        "It was a waste of time, really disappointing",
        "The plot made no sense at all",
        "The acting was weak and unconvincing",
        "I almost fell asleep watching this",
        "A completely forgettable movie",
        "Nothing interesting happened in the entire film",
        "The story was dull and predictable",
        "Terrible directing and bad editing",
        "This film failed to keep my attention",
        "A boring and lifeless experience",
        "I regret watching this movie",
        "The characters were annoying and unrealistic",
        "This movie was poorly executed",
        "The dialogue was awkward and unnatural",
        "A disappointing and frustrating film",
        "The ending was terrible and unsatisfying",
        "This was not worth my time at all",
        "The movie dragged on forever",
        "A very weak and uninspired story",
        "I expected much more but got nothing",
        "This film was messy and confusing",
        "The pacing was slow and painful",
        "It lacked emotion and depth",
        "A poorly written and badly acted film",
        "This movie was just plain bad",
        "I would not recommend this to anyone"
    ]

    train_data = []

    # 1000 positive + 1000 negative
    for _ in range(1000):
        train_data.append(("pos", random.choice(positive_reviews)))

    for _ in range(1000):
        train_data.append(("neg", random.choice(negative_reviews)))

    random.shuffle(train_data)
    return train_data


# -----------------------------
# 2. BUILD VOCAB
# -----------------------------
def yield_tokens(data):
    for label, text in data:
        yield tokenizer(text)


def build_vocab(data):
    vocab = build_vocab_from_iterator(yield_tokens(data), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab


# -----------------------------
# 3. COLLATE FUNCTION
# -----------------------------
def collate_batch(batch, vocab):
    texts, labels = [], []

    for label, text in batch:
        tokens = torch.tensor(text_pipeline(text, vocab), dtype=torch.long)
        texts.append(tokens)
        labels.append(1 if label == "pos" else 0)

    texts = pad_sequence(texts, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.float)

    return texts, labels


# -----------------------------
# 4. MAIN TRAINING LOOP
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = create_dataset()
    vocab = build_vocab(train_data)

    loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, vocab)
    )

    model = SentimentModel(len(vocab), 100, 128).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(5):
        model.train()
        total_loss = 0

        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} -> Loss: {avg_loss:.4f}")

    # -----------------------------
    # SAVE MODEL + VOCAB
    # -----------------------------
    torch.save(model.state_dict(), "model.pth")

    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    print("\nModel and vocab saved!")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()