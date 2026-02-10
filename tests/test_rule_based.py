from sexism_distillation.compression.rule_based import (
    RuleBasedCollapsingConfig,
    compute_gradient_sensitivity,
    compute_importance_scores,
    filter_blocks,
)


def test_filter_blocks_keeps_important_layers():
    blocks = ["cnn1", "cnn2", "ffn"]
    importance = compute_importance_scores({"cnn1": 0.9, "cnn2": 0.1, "ffn": 0.2})
    sensitivity = compute_gradient_sensitivity({"cnn1": 0.9, "cnn2": 0.01, "ffn": 0.2})

    kept = filter_blocks(
        blocks,
        importance,
        sensitivity,
        flops_saving_ratio=0.4,
        config=RuleBasedCollapsingConfig(),
    )

    assert "cnn1" in kept
