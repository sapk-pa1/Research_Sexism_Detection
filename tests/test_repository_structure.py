from pathlib import Path


def test_expected_directories_exist():
    required = [
        Path("sexism_distillation/models"),
        Path("sexism_distillation/losses"),
        Path("sexism_distillation/compression"),
        Path("sexism_distillation/search"),
        Path("sexism_distillation/evaluation"),
        Path("sexism_distillation/pipelines"),
        Path("scripts"),
        Path("configs"),
    ]
    for path in required:
        assert path.exists(), f"Missing required directory: {path}"
