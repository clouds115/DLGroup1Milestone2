import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm.auto import tqdm
import json
from pathlib import Path
from datetime import datetime
import time
import re

# Part 1 - Imports and device selection
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Part 2 - Load and preprocess the dataset (FIXED PATHS)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')

fake_df = pd.read_csv(os.path.join(data_dir, 'Fake.csv'))
true_df = pd.read_csv(os.path.join(data_dir, 'True.csv'))

fake_df['label'] = 0
true_df['label'] = 1
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Combine title and text
if 'title' in df.columns and 'text' in df.columns:
    df['content'] = df['title'] + ' ' + df['text']
elif 'text' in df.columns:
    df['content'] = df['text']
elif 'title' in df.columns:
    df['content'] = df['title']
else:
    text_cols = [col for col in df.columns if col not in ['label', 'subject', 'date']]
    if text_cols:
        df['content'] = df[text_cols[0]]
    else:
        raise ValueError("Could not find text column in dataset")

# Clean Reuters tags
def clean_reuters_tag(s):
    if not isinstance(s, str):
        return s
    s = re.sub(r'\(?\bReuters\b\)?\s*-\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\(?\bReuters\b\)?', '', s, flags=re.IGNORECASE)
    return s.strip()

df['content'] = df['content'].apply(clean_reuters_tag)

# Train-test split
train_val_df, test_df = train_test_split(
    df, test_size=0.15, random_state=SEED, stratify=df['label']
)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.1765, random_state=SEED, stratify=train_val_df['label']
)

# Part 4 - Global constants
BATCH_SIZE = 32
MAX_LENGTH = 128
MODEL_NAME = 'distilbert-base-uncased'
OUTPUT_DIR = Path("../DLMilestone2experiments")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

configs = [
    {"name": "1", "epochs": 1,  "per_device_train_batch_size": BATCH_SIZE, "learning_rate": 3e-5,  "weight_decay": 0.02},
    {"name": "2", "epochs": 1, "per_device_train_batch_size": BATCH_SIZE, "learning_rate": 1.5e-5,  "weight_decay": 0.02},
    {"name": "3", "epochs": 2,  "per_device_train_batch_size": BATCH_SIZE, "learning_rate": 3e-5,  "weight_decay": 0.02},
    {"name": "4", "epochs": 2, "per_device_train_batch_size": BATCH_SIZE, "learning_rate": 1.5e-5, "weight_decay": 0.02},
    {"name": "5", "epochs": 3, "per_device_train_batch_size": BATCH_SIZE, "learning_rate": 3e-5,  "weight_decay": 0.02},
    {"name": "6", "epochs": 3,  "per_device_train_batch_size": BATCH_SIZE, "learning_rate": 1.5e-5,  "weight_decay": 0.02},
]

# Part 5 - Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Part 6 - FIXED Dataset class (uniform tensor sizes)
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

        # FIXED: Proper tokenization with consistent sizing
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # [max_length]
            'attention_mask': encoding['attention_mask'].squeeze(0),  # [max_length]
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets and dataloaders
train_dataset = FakeNewsDataset(train_df['content'].values, train_df['label'].values, tokenizer)
val_dataset = FakeNewsDataset(val_df['content'].values, val_df['label'].values, tokenizer)
test_dataset = FakeNewsDataset(test_df['content'].values, test_df['label'].values, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    losses = []
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(data_loader, desc='Training')
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)
        losses.append(loss.item())

        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{(correct_predictions.double() / total_predictions).item():.4f}"
        })

    return correct_predictions.double() / total_predictions, np.mean(losses)

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

    return (
        correct_predictions.double() / total_predictions,
        np.mean(losses),
        np.array(all_preds),
        np.array(all_labels)
    )

def setup_optimizer_and_scheduler(model, train_loader, learning_rate, weight_decay, epochs):
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    print(f"Total training steps: {total_steps}")
    return optimizer, scheduler

def save_run_result(run_metadata: dict, metrics: dict, filename: str = None):
    payload = {"timestamp": datetime.utcnow().isoformat() + "Z", "config": run_metadata, "metrics": metrics}
    if filename is None:
        safe_name = f"run_{run_metadata['name']}_e{run_metadata['epochs']}_lr{run_metadata['learning_rate']}_wd{run_metadata['weight_decay']}.json"
        filename = OUTPUT_DIR / safe_name
    else:
        filename = OUTPUT_DIR / filename
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {filename}")

def main():
    all_results = []
    for cfg in configs:
        config_num = cfg["name"]
        config_dir = OUTPUT_DIR / f"cfg{config_num}"
        config_dir.mkdir(exist_ok=True, parents=True)

        print("=" * 60)
        print(f"Starting run for CONFIG {cfg['name']} | epochs={cfg['epochs']}, lr={cfg['learning_rate']}, wd={cfg['weight_decay']}, batch_size={cfg['per_device_train_batch_size']}, device={device}, max_length={MAX_LENGTH}")
        print("=" * 60)

        for run in range(1, 7):
            print("-" * 40)
            print(f"Run {run}/6 for config {cfg['name']}")
            print("-" * 40)

            run_start_time = time.time()
            model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

            optimizer, scheduler = setup_optimizer_and_scheduler(
                model, train_loader, learning_rate=cfg['learning_rate'],
                weight_decay=cfg['weight_decay'], epochs=cfg['epochs']
            )

            history = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}
            epoch_times = []
            best_val_acc = 0

            for epoch in range(cfg['epochs']):
                print(f"Epoch {epoch + 1}/{cfg['epochs']}")
                epoch_start_time = time.time()

                train_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

                val_acc, val_loss, _, _ = eval_model(model, val_loader, device)
                print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                epoch_duration = time.time() - epoch_start_time
                epoch_times.append(epoch_duration)
                print(f"Epoch time: {epoch_duration:.2f} seconds")

                # FIXED: Safe .item() handling
                history["train_acc"].append(train_acc.item() if hasattr(train_acc, 'item') else float(train_acc))
                history["train_loss"].append(float(train_loss))
                history["val_acc"].append(val_acc.item() if hasattr(val_acc, 'item') else float(val_acc))
                history["val_loss"].append(float(val_loss))

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model_path = config_dir / f"bestmodel_{cfg['name']}_run{run}.pt"
                    torch.save(model.state_dict(), model_path)
                    print(f"> Best model saved with validation accuracy {best_val_acc:.4f}")

            run_duration = time.time() - run_start_time
            print(f"Run time: {run_duration:.2f} seconds")

            metrics = {
                "history": history,
                "epoch_times": epoch_times,
                "run_time": run_duration,
                "best_val_acc": best_val_acc.item() if hasattr(best_val_acc, 'item') else float(best_val_acc),
            }

            save_run_result(cfg, metrics, f"run_{cfg['name']}_{run}.json")
            all_results.append({"config": cfg, "metrics": metrics})

if __name__ == "__main__":
    main()
