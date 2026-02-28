import os
import pandas as pd
import numpy as np
import torch
import re
import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

# --- 1. Setup and Arguments ---
parser = argparse.ArgumentParser(description="Test a saved DistilBERT model on the Fake News dataset.")
parser.add_argument('--config', type=str, default="1", help="Configuration number (e.g., 1)")
parser.add_argument('--run', type=int, default=1, help="Run number (e.g., 1)")
args = parser.parse_args()

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BATCH_SIZE = 32
MAX_LENGTH = 128
MODEL_NAME = 'distilbert-base-uncased'

# --- 2. Data Loading (Replicating exact splits) ---
print("Loading and splitting dataset to recreate the test set...")
# Accommodate both local and slurm execution paths
data_dir = 'data' if os.path.exists('../data/Fake.csv') else os.path.join(os.path.dirname(__file__), '../data')

fake_df = pd.read_csv(os.path.join(data_dir, 'Fake.csv'))
true_df = pd.read_csv(os.path.join(data_dir, 'True.csv'))

fake_df['label'] = 0
true_df['label'] = 1
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Combine columns
if 'title' in df.columns and 'text' in df.columns:
    df['content'] = df['title'] + ' ' + df['text']
else:
    df['content'] = df['text']


# Clean Reuters tags
def clean_reuters_tag(s):
    if not isinstance(s, str): return s
    s = re.sub(r'\(?\bReuters\b\)?\s*-\s*', '', s, flags=re.IGNORECASE)
    return re.sub(r'\(?\bReuters\b\)?', '', s, flags=re.IGNORECASE).strip()


df['content'] = df['content'].apply(clean_reuters_tag)

# Split: Extract exactly the 15% test set used in training
_, test_df = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df['label'])
print(f"Test set ready: {len(test_df)} samples.")


# --- 3. Dataset and DataLoader ---
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_length,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
test_dataset = FakeNewsDataset(test_df['content'].values, test_df['label'].values, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --- 4. Evaluation Function ---
def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc='Evaluating')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)
            losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (correct_predictions.double() / total_predictions, np.mean(losses),
            np.array(all_preds), np.array(all_labels))


# --- 5. Model Loading and Testing ---
OUTPUT_DIR = Path("bestmodel")
best_model_path = OUTPUT_DIR / f"bestmodel_{args.config}_run{args.run}.pt"

if not best_model_path.exists():
    print(f"\n[ERROR] Model not found at: {best_model_path}")
    print("Please check your --config and --run arguments.")
    exit(1)

print(f"\nLoading model architecture and weights from {best_model_path} ...")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
# Load safely
state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)

print("\nStarting evaluation on test set...")
test_acc, test_loss, preds, labels_true = eval_model(model, test_loader, device)

# --- 6. Results Output ---
print(f"\n{'=' * 40}")
print(f"RESULTS FOR CONFIG {args.config} | RUN {args.run}")
print(f"{'=' * 40}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Acc:  {test_acc:.4f}\n")

print("Classification Report:")
print(classification_report(labels_true, preds, target_names=["Fake (0)", "True (1)"], digits=4))

print("Confusion Matrix:")
print(confusion_matrix(labels_true, preds))