# Sexism Detection Distillation Repository

A properly structured repository for training an **Adaptive Student Network** from a **BERT teacher model** .

## Folder structure

```text
.
├── configs/
│   └── teacher_checkpoints.example.yaml
├── scripts/
│   ├── train_distillation_example.py
│   └── train_distillation_pipeline.py
├── sexism_distillation/
│   ├── compression/
│   │   └── rule_based.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── losses/
│   │   └── distillation.py
│   ├── models/
│   │   ├── student.py
│   │   └── teacher.py
│   ├── pipelines/
│   │   ├── example.py
│   │   └── pipeline.py
│   └── search/
│       └── nas_controller.py
├── tests/
│   ├── test_repository_structure.py
│   └── test_rule_based.py
├── pyproject.toml
├── requirements.txt
├── student_model.py
├── train_distillation_example.py   # backward-compatible wrapper
└── train_distillation_pipeline.py  # backward-compatible wrapper
```

## Architecture coverage

1. Domain corpus / teacher checkpoint
2. Pretrained domain BERT teacher (EXIST21/CMSB)
3. Adaptive Student Network:
   - Embedding
   - Local CNN block ×2
   - Lightweight attention
   - FFN
   - Key attention
   - Output layer
4. Knowledge Distillation loss:
   - Logits KL
   - Hidden-state alignment
   - Selective attention alignment
5. Rule-based collapsing:
   - Importance score
   - Gradient sensitivity
   - FLOPs threshold
6. NAS controller:
   - CNN kernels
   - Attention heads
   - FFN/hidden-size options
   - Skip connections
7. Evaluation hooks:
   - Accuracy
   - Latency
   - Model size

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python scripts/train_distillation_pipeline.py --teacher-dataset exist21 --device cpu
```

or

```bash
python scripts/train_distillation_example.py --teacher-dataset cmsb
```

## Teacher checkpoints

Update `configs/teacher_checkpoints.example.yaml`, and replace placeholder checkpoint names in `sexism_distillation/models/teacher.py`.
