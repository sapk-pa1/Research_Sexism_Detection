"""End-to-end distillation pipeline reflecting the provided architecture."""

from __future__ import annotations

import argparse

import torch

from sexism_distillation.compression.rule_based import (
    RuleBasedCollapsingConfig,
    compute_gradient_sensitivity,
    compute_importance_scores,
    filter_blocks,
)
from sexism_distillation.evaluation.metrics import model_size_mb
from sexism_distillation.losses.distillation import DistillationLoss, DistillationLossConfig
from sexism_distillation.models.student import AdaptiveStudentConfig, AdaptiveStudentNetwork
from sexism_distillation.models.teacher import TeacherModelFactory
from sexism_distillation.search.nas_controller import NASController, NASSearchSpace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dataset", choices=["exist21", "cmsb"], default="exist21")
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--checkpoint-config", default=None, help="Optional YAML file with dataset->checkpoint mapping")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=30522)
    parser.add_argument("--use-nas", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.checkpoint_config:
        TeacherModelFactory.load_presets_from_yaml(args.checkpoint_config)

    checkpoint = TeacherModelFactory.resolve_checkpoint(args.teacher_dataset, args.teacher_checkpoint)
    teacher = TeacherModelFactory.load_from_hf(checkpoint=checkpoint, device=args.device)

    if args.use_nas:
        controller = NASController(NASSearchSpace())
        student_cfg = controller.sample_config(vocab_size=args.vocab_size, num_classes=args.num_classes)
    else:
        student_cfg = AdaptiveStudentConfig(vocab_size=args.vocab_size, num_classes=args.num_classes)

    student = AdaptiveStudentNetwork(student_cfg).to(args.device)
    loss_fn = DistillationLoss(DistillationLossConfig())

    candidate_blocks = ["cnn1", "cnn2", "light_attention", "ffn"]
    importance_scores = compute_importance_scores({"cnn1": 0.7, "cnn2": 0.6, "light_attention": 0.8, "ffn": 0.4})
    gradient_sensitivity = compute_gradient_sensitivity(
        {"cnn1": 0.2, "cnn2": 0.15, "light_attention": 0.3, "ffn": 0.08}
    )

    kept_blocks = filter_blocks(
        candidate_blocks,
        importance_scores,
        gradient_sensitivity,
        flops_saving_ratio=0.30,
        config=RuleBasedCollapsingConfig(),
    )

    print(f"Teacher checkpoint: {checkpoint}")
    print(f"Teacher model: {teacher.__class__.__name__}")
    print(f"Student configuration: {student_cfg}")
    print(f"Distillation loss configured: {loss_fn.config}")
    print(f"Rule-based kept blocks: {kept_blocks}")
    print(f"Student model size (MB): {model_size_mb(student):.2f}")
    print("Next step: plug your tokenized DataLoader and run optimization loop.")


if __name__ == "__main__":
    main()
