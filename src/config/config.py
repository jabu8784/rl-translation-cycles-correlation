import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ExperimentConfig:
    mode: Literal["train", "eval", "correlation"]
    language: str
    pretrained_model: str = "Qwen/Qwen3-0.6B"
    model_type: Literal["decoder", "seq2seq"] = "decoder"

    max_tokens: int = 256
    stop_string: str | None = "\n"

    train_dataset: str = "HuggingFaceFW/fineweb"
    train_dataset_subset: str = "sample-10BT"
    train_num_sentences: int = 20000
    train_max_sentence_len: int = 128

    eval_dataset: str = "openlanguagedata/flores_plus"
    eval_num_sentences: int | None = None

    # GRPO
    alpha: float = 0  # alpha * fwd_loss + (1-alpha) * bwd_loss
    grpo_group_size: int = 4
    grpo_beta: float = 0.1
    grpo_temperature: float = 0.5
    grpo_top_p: float = 0.9
    grpo_top_k: int = 100
    grpo_epsilon: float = (
        0.2  # Not used for now, used if doing multiple updates/rollout
    )
    reward_metric: Literal["bleu", "chrf", "comet22", "bleurt", "bertscore", "both", "all"] = "both"
    grpo_num_backward: int = 1  # Number of backward translations (sampled)
    grpo_greedy_backward: bool = False

    # Training
    max_epochs: int = 3
    learning_rate: float = 1e-5
    batch_size: int = 4
    grad_acc_steps: int = 4
    inner_update_steps: int = 2  # Number of "inner updates" per sample of rollouts
    grad_norm: float | None = None
    optimizer: str = "adamw"
    warmup_steps: int = 100
    eval_every_n_steps: int = 50
    reference_update_steps: int = -1  # If >0, how often to update the reference policy

    # Logging
    wandb_project: str = "rl-translation-cycles-2"
    wandb_run_name: str | None = None

    # Checkpointing
    models_dir: str = "checkpoints"

    # System
    seed: int = 42
    slurm_job_id: str | None = field(
        default_factory=lambda: os.environ.get("SLURM_JOB_ID")
    )
    skip_initial_eval: bool = False

    @property
    def is_nllb(self):
        return "nllb" in self.pretrained_model
