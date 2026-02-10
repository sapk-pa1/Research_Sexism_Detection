from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class TeacherModelConfig:
    checkpoint: str
    num_labels: int


class TeacherModelFactory:
    """Load teacher models fine-tuned on sexism-related datasets."""

    PRESET_CHECKPOINTS: Dict[str, str] = {
        "exist21": "your-org/exist21-bert-sexism-teacher",
        "cmsb": "your-org/cmsb-bert-sexism-teacher",
    }

    @classmethod
    def load_presets_from_yaml(cls, yaml_path: str) -> Dict[str, str]:
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint config not found: {yaml_path}")

        try:
            import yaml
        except ImportError as exc:
            raise ImportError("pyyaml is required to load checkpoint yaml files.") from exc

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError("Checkpoint yaml must be a mapping of dataset_name -> checkpoint")

        normalized = {str(k).lower(): str(v) for k, v in data.items()}
        cls.PRESET_CHECKPOINTS.update(normalized)
        return cls.PRESET_CHECKPOINTS

    @classmethod
    def resolve_checkpoint(cls, dataset_name: str, fallback_checkpoint: str | None = None) -> str:
        key = dataset_name.lower()
        if key in cls.PRESET_CHECKPOINTS:
            return cls.PRESET_CHECKPOINTS[key]
        if fallback_checkpoint:
            return fallback_checkpoint
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Use one of {list(cls.PRESET_CHECKPOINTS.keys())} "
            "or provide fallback_checkpoint."
        )

    @staticmethod
    def load_from_hf(checkpoint: str, device: str = "cpu"):
        try:
            from transformers import AutoModelForSequenceClassification
        except ImportError as exc:
            raise ImportError("transformers is required to load the teacher model.") from exc

        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        return model.to(device)
