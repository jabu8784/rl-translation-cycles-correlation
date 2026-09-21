"""Main entry point for rl-translation-cycles experiments."""

from __future__ import annotations

import argparse
import logging
import random
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

import wandb
from src.config import ExperimentConfig, config_to_dataclass
from src.data import FloresEvalDataset, NLLBEvalDataset
from src.distributed import cleanup_distributed, setup_distributed
from src.evaluate import evaluate
from src.evaluate_correlation import evaluate_correlation
from src.train import train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RL Translation Cycles: GRPO for low-resource translation"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to .cfg experiment config file",
    )
    parser.add_argument(
        "--overrides",
        "-o",
        nargs="*",
        default=[],
        help="Config overrides as key=value pairs",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    config = config_to_dataclass(args.config, args.overrides, ExperimentConfig)
    dist_config = setup_distributed()
    random.seed(0)
    torch.manual_seed(0)

    if dist_config.is_main:
        logger.info(f"Config: {config}")
        logger.info(f"Mode: {config.mode}")
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={**vars(config), "distributed": vars(dist_config)},
        )
        # log_pip_freeze_artifact(f"pip-freeze-{wandb.run.id}")  # type:ignore

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model,
            padding_side="left" if config.model_type == "decoder" else "right",
            **(
                {"src_lang": "eng_Latn", "tgt_lang": config.language}
                if config.is_nllb
                else {}
            ),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            import flash_attn  # noqa: F401 # type:ignore

            attn_impl = "flash_attention_2"
            logger.info("Using flash_attention")
        except ImportError:
            attn_impl = "eager"
            logger.info("Falling back to eager attention")
        if config.model_type == "decoder":
            model: Any = AutoModelForCausalLM.from_pretrained(
                config.pretrained_model,
                # dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
        elif config.model_type == "seq2seq":
            model: Any = AutoModelForSeq2SeqLM.from_pretrained(
                config.pretrained_model,
                # dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
        else:
            raise NotImplementedError()
        model = model.to(dist_config.device)
        model.gradient_checkpointing_enable()
        model.generation_config.max_length = None
        if dist_config.distributed:
            logger.info("Wrapping model in DDP")
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[dist_config.local_rank]
            )
        #-------------TESTING----------------
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(
            "Translation model parameters: total=%d, requires_grad=%d",
            total,
            trainable,
        )
        #------------------------------------
        if config.eval_dataset == "breakend/nllb-multi-domain":
            dev_dataset = NLLBEvalDataset("dev", config)
            test_dataset = NLLBEvalDataset("test", config)
        else:
            dev_dataset = FloresEvalDataset("dev", config)
            test_dataset = FloresEvalDataset("test", config)

        if config.mode == "train":
            train(model, tokenizer, dev_dataset, test_dataset, config, dist_config)
        elif config.mode == "eval":
            dev_metrics = evaluate(model, tokenizer, dev_dataset, config, dist_config)
            test_metrics = evaluate(model, tokenizer, test_dataset, config, dist_config)
            if dist_config.is_main:
                wandb.log({"dev": dev_metrics, "test": test_metrics})
        elif config.mode == "correlation":
            # Correlational analysis between cycle consistency + main metric
            evaluate_correlation(model, tokenizer, dev_dataset, config, dist_config)
        else:
            raise ValueError(f"Unknown mode: {config.mode}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
