from __future__ import annotations

import argparse
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from student_model import AdaptiveStudentNetwork, DistillationLoss, DistillationLossConfig


class DummySexismDataset(Dataset):
    """Replace this with your real sexism dataset."""

    def __init__(self, tokenizer, max_len: int = 128, n_samples: int = 128) -> None:
        texts = [
            "Women are bad at coding",  # sexist example
            "Everyone can be a great engineer",  # non-sexist example
        ]
        labels = [1, 0]
        self.samples = []
        for i in range(n_samples):
            t = texts[i % len(texts)]
            y = labels[i % len(labels)]
            enc = tokenizer(
                t,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            self.samples.append(
                {
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "labels": torch.tensor(y, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name)
    teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher_model_name)
    teacher.to(device)
    teacher.eval()

    student = AdaptiveStudentNetwork(
        vocab_size=tokenizer.vocab_size,
        num_labels=args.num_labels,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        max_len=args.max_len,
        dropout=args.dropout,
    ).to(device)

    loss_fn = DistillationLoss(DistillationLossConfig(alpha=args.alpha, temperature=args.temperature))
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    dataset = DummySexismDataset(tokenizer=tokenizer, max_len=args.max_len, n_samples=args.n_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        student.train()
        epoch_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

            student_logits = student(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            loss = loss_fn(student_logits=student_logits, teacher_logits=teacher_logits, labels=labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {avg:.4f}")

    torch.save(student.state_dict(), args.output_path)
    print(f"Saved student weights to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train student sexism classifier via distillation")
    parser.add_argument("--teacher_model_name", type=str, required=True)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="student_model.pt")

    args = parser.parse_args()
    train(args)
