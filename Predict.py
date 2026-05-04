import torch
import argparse
import pickle

from model import SentimentModel, text_pipeline, tokenizer

# LOAD VOCAB
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# LOAD MODEL
model = SentimentModel(len(vocab), 100, 128)
model.load_state_dict(torch.load("model.pth"))
model.eval()

def predict(text):
    tokens = torch.tensor(text_pipeline(text, vocab)).unsqueeze(0)

    with torch.no_grad():
        prob = model(tokens).item()

    label = 1 if prob > 0.5 else 0
    sentiment = "positive" if label else "negative"

    print(f"text: {text}")
    print(f"predicted_class: {label} ({sentiment})")
    print(f"predicted_probability: {prob:.4f}")

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    predict(args.text)