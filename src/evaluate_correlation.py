from __future__ import annotations

import logging
import pathlib
from collections import defaultdict
from typing import Any

import pandas
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.config import ExperimentConfig
from src.data import EvalDataset
from src.distributed import DistributedConfig
from src.modeling.generation import greedy_decode, sample_completions
from src.modeling.prompts import make_backward_prompt, make_forward_prompt
from src.modeling.rewards import compute_sentence_metric, load_metric_scorers

logger = logging.getLogger(__name__)


def evaluate_correlation(
    model: Any,
    tokenizer: Any,
    dataset: EvalDataset,
    config: ExperimentConfig,
    dist_config: DistributedConfig,
):
    if dist_config.distributed:
        raise NotImplementedError()
    #group_size = number of output translations
    #batch_size = number of sentences to translate
    gs = config.grpo_group_size
    loader = DataLoader(
        dataset,
        #batch_size=config.batch_size * config.grpo_group_size,
        batch_size=config.batch_size,
        shuffle=False,
    )
    model.eval()
    metrics: dict[str, list[float]] = defaultdict(list)
    running_idx = 0

    # (Scorer for certain metrics)
    scorers = load_metric_scorers(
        use_comet=config.reward_metric in ("comet22", "all"),
        use_bleurt=config.reward_metric in ("bleurt", "all"),
        use_bertscore=config.reward_metric in ("bertscore", "all"),
        bleurt_checkpoint="BLEURT-20",
    )

    for batch in tqdm(loader, desc="Evaluating", disable=not dist_config.is_main):
        eng_sentences = batch["eng"]
        tgt_sentences = batch["tgt"]
        bs = len(eng_sentences)
        # Generate prompts to ask model to translate sentences
        fwd_prompts = [make_forward_prompt(s, config) for s in eng_sentences]
        fwd_preds, _ = sample_completions(
            model,
            tokenizer,
            fwd_prompts,
            target_lang=config.language,
            num_samples=gs,
            config=config,
        )
        # Uncomment to see example forward translation
        # logger.info("Example source: %s", eng_sentences[0])
        # logger.info("Example forward candidates: %s", fwd_preds[0][:2])
        bwd_prompts = [
            make_backward_prompt(s, config) for group in fwd_preds for s in group
        ]
        bwd_preds = greedy_decode(
            model,
            tokenizer,
            bwd_prompts,
            target_lang="eng_Latn",
            config=config,
        )
        bwd_preds = [bwd_preds[i * gs : (i + 1) * gs] for i in range(bs)]
        # Uncomment to see example back-translation
        # logger.info("Example back-translation: %s", bwd_preds[0])

        # Compute metrics
        if config.reward_metric == "both":
            metric_list = ["bleu", "chrf"]
        elif config.reward_metric == "all":
            metric_list = ["bleu", "chrf", "comet22", "bleurt", "bertscore"]
        else:
            metric_list = [config.reward_metric]
        for metric in metric_list:
            fwd_scores = torch.tensor(
                [
                    compute_sentence_metric(
                        fwd_preds[idx],
                        [tgt_sentences[idx]] * gs,
                        metric,
                        sources=[eng_sentences[idx]] * gs,
                        scorers=scorers,
                        bertscore_lang=config.language,
                    )
                    for idx in range(bs)
                ]
            )
            # Normalize fwd sentence score
            fwd_scores_norm = (
                fwd_scores - fwd_scores.mean(dim=-1, keepdim=True)
            ) / fwd_scores.std(dim=-1, keepdim=True)
            fwd_scores_norm = fwd_scores_norm.reshape(bs * gs).tolist()
            # Store fwd scores
            metrics[metric].extend(fwd_scores_norm)
            if metric == "comet22":
                round_trip_scores = torch.tensor(
                    [
                        compute_sentence_metric(
                            bwd_preds[idx],
                            [eng_sentences[idx]] * gs,
                            metric,
                            sources=fwd_preds[idx],
                            scorers=scorers,
                            bertscore_lang="en",
                        )
                        for idx in range(bs)
                    ]
                )
            else:
                round_trip_scores = torch.tensor(
                    [
                        compute_sentence_metric(
                            bwd_preds[idx],
                            [eng_sentences[idx]] * gs,
                            metric,
                            sources=[eng_sentences[idx]] * gs,
                            scorers=scorers,
                        )
                        for idx in range(bs)
                    ]
                )
            # Normalize round trip scores
            round_trip_scores_norm = (
                round_trip_scores - round_trip_scores.mean(dim=-1, keepdim=True)
            ) / round_trip_scores.std(dim=-1, keepdim=True)
            round_trip_scores_norm = round_trip_scores_norm.reshape(bs * gs).tolist()
            # Store round scores
            metrics[f"roundtrip_{metric}"].extend(round_trip_scores_norm)
        #store indices for related translations
        sent_indices = []
        for idx in range(running_idx, running_idx + bs):
            sent_indices.extend([idx] * gs)
        running_idx = running_idx + bs
        metrics["sent_idx"].extend(sent_indices)
    # Make df of metrics, rows are translations, columns are scores + sentence index
    df = pandas.DataFrame.from_dict(metrics)
    path = pathlib.Path(f"{config.language}_metrics.csv")
    df.to_csv(path)
    logger.info(f"Wrote metrics to {path.resolve()}")
