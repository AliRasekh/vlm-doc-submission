"""Pascal P3 — fixed-budget three-seed LoRA reproducibility (development only)."""

SEEDS = (42, 43, 44)
ORIGINAL_V2_SEED = 42
MAX_UPDATES = 200
GRAD_ACCUM = 8
PRESENTATIONS_PER_RUN = MAX_UPDATES * GRAD_ACCUM  # 1600
