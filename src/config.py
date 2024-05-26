from dataclasses import dataclass


@dataclass
class ModelConfig:
    batch_size: int
    block_size: int
    dropout: float
    n_embed: int
    n_head: int
    n_layer: int
    vocab_size: int


@dataclass
class OptimizerConfig:
    learning_rate: float


@dataclass
class TrainConfig:
    always_save_ckpt: bool  # save checkpoint after each eval
    eval_interval: int
    eval_iters: int
    log_interval: int
    max_iters: int  # total number of training iterations
    output_dir: str
    resume: bool  # begin training from scratch or resume from checkpoint
