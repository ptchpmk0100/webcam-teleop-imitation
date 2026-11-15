"""The final-scores table: every training seed is printed, then the mean; nothing is chosen."""
from teleop_il.final_scores import summarise


def test_every_training_seed_is_listed_and_the_mean_follows():
    rows = [{"run": "a", "seed": 0, "n": 100, "python": 96, "cpp": 96, "agree": 100, "assist": 99},
            {"run": "b", "seed": 1, "n": 100, "python": 90, "cpp": 90, "agree": 100, "assist": 100}]
    lines = summarise("100 human demos", rows)
    assert len(lines) == 3
    assert "96/100 (0.90-0.98)" in lines[0] and "| a |" in lines[0]
    assert "90/100" in lines[1] and "| b |" in lines[1]
    assert "mean of 2 | 93.0 | 93.0 | | 99.5" in lines[2]
