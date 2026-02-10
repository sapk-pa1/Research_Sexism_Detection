"""Minimal distillation loop with teacher model presets (EXIST21 / CMSB)."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from sexism_distillation.losses.distillation import DistillationLoss, DistillationLossConfig
from sexism_distillation.models.student import AdaptiveStudentConfig, AdaptiveStudentNetwork
from sexism_distillation.models.teacher import TeacherModelFactory


def train_one_epoch(
    student: AdaptiveStudentNetwork,
    teacher,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: DistillationLoss,
) -> float:
    student.train()
    teacher.eval()

    running_loss = 0.0
    total_items = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            teacher_out = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
            )

        student_out = student(input_ids=input_ids, attention_mask=attention_mask, return_aux=True)

        loss = loss_fn(
            student_logits=student_out["logits"],
            teacher_logits=teacher_out.logits,
            labels=labels,
            student_hidden=student_out["hidden_states"][-1],
            teacher_hidden=teacher_out.hidden_states[-1],
            student_attention=student_out["attention_weights"],
            teacher_attention=teacher_out.attentions[-1] if teacher_out.attentions is not None else None,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = input_ids.size(0)
        running_loss += loss.item() * batch_size
        total_items += batch_size

    return running_loss / max(total_items, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dataset", choices=["exist21", "cmsb"], default="exist21")
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--checkpoint-config", default=None, help="Optional YAML file with dataset->checkpoint mapping")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint_config:
        TeacherModelFactory.load_presets_from_yaml(args.checkpoint_config)

    ckpt = TeacherModelFactory.resolve_checkpoint(args.teacher_dataset, args.teacher_checkpoint)
    teacher = TeacherModelFactory.load_from_hf(ckpt, device=str(device))

    student = AdaptiveStudentNetwork(
        AdaptiveStudentConfig(vocab_size=30522, num_classes=2, hidden_size=256)
    ).to(device)

    distill_loss = DistillationLoss(
        DistillationLossConfig(alpha_logits=0.6, alpha_hidden=0.25, alpha_attention=0.15, temperature=3.0)
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4)

    print(f"Teacher loaded: {ckpt}")
    print("Student initialized.")
    print("Loss config:", distill_loss.config)
    print("Optimizer:", optimizer.__class__.__name__)
    print("Create a tokenized DataLoader and call train_one_epoch(...) to train.")


if __name__ == "__main__":
    main()
