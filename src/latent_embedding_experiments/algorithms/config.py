from dataclasses import dataclass, field


@dataclass
class SolverConfig:
    steps: int = 300
    lr: float = 0.05

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
    interloper_margin_threshold: float = 0.2

    # Loss 4: Target similarity loss
    # Pulls the embedding toward all target tokens (unweighted)
    target_sim_weight: float = 1.0


@dataclass
class RLConfig:
    lora_r: int = 64
    lora_alpha: int = 128
    lora_targets: tuple[str] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    group_size: int = 4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    learning_rate: float = 1e-6
    temperature_start: float = 0.6
    temperature_end: float = 2.0
    temperature_increment_every: int = 50

    max_tokens: int = 256
    max_update_steps: int = 2000


@dataclass
class Config:
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    approaches: list[str] = field(
        default_factory=lambda: [
            "discrete_top1",
            # "discrete_cleaned",
            # "discrete_cleaned_dot_rescaled",
            "noisy_discrete",
            "soft_thinking",
            # "clean_soft",
            # "clean_soft_aggregate",
            # "latent_head",
            # "solver",
            # "centroid",
            # "coconut",
            "noisy_discrete_bernoulli",
            "noisy_discrete_pc_random",
            "noisy_discrete_pc_deterministic",
        ]
    )

    temperature: float = 0.6
    top_p: float = 0.97
    top_k: int = 10  # not actually used in training or eval
    num_interlopers: int = 10

    solver_config: SolverConfig = field(default_factory=SolverConfig)

    rl_config: RLConfig = field(default_factory=RLConfig)


CFG = Config()
