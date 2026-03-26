from dataclasses import dataclass


@dataclass
class Config:
    solver_steps: int = 300
    lr: float = 0.05

    # Number of interlopers to consider in loss 3
    num_interlopers: int = 10

    # Top-p (nucleus) selection for determining k
    temperature: float = 0.6
    min_p: float = 0.03  # minimum probability threshold
    min_k: int = 1  # always consider at least this many tokens
    max_k: int = 10  # cap to avoid degenerate cases

    # Loss 1: Target ranking loss
    # Pure ordering — fires when any pair (i, j) with p_i > p_j has s_j > s_i
    ranking_weight: float = 3.0

    # Loss 2: Target margin loss
    # Adjacent pairs only — demands similarity gap proportional to probability gap
    margin_weight: float = 2.0
    margin_min: float = 0.02
    margin_scale: float = 0.5

    # Loss 3: Interloper similarity loss
    # Pushes interloper similarities below the weakest target's similarity
    interloper_weight: float = 5.0
    interloper_margin_relative: float = 1 / 5

    # Loss 4: Target similarity loss
    # Pulls the embedding toward all target tokens (unweighted)
    target_sim_weight: float = 1.0


CFG = Config()
