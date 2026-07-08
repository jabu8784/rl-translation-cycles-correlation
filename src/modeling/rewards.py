"""Cycle-consistency reward computation."""

import logging
from pprint import pformat
from dataclasses import dataclass
from typing import Any, Optional, cast
import sacrebleu
import torch

logger = logging.getLogger(__name__)

@dataclass
class MetricScorers:
    comet: Optional[Any] = None
    bleurt: Optional[Any] = None


def load_metric_scorers(
    use_comet: bool = True,
    use_bleurt: bool = True,
    bleurt_checkpoint: str = "BLEURT-20",
) -> MetricScorers:
    scorers = MetricScorers()

    if use_comet:
        from comet import download_model, load_from_checkpoint

        comet_path = download_model("Unbabel/wmt22-comet-da")
        scorers.comet = load_from_checkpoint(comet_path)

    if use_bleurt:
        from bleurt import score

        scorers.bleurt = score.BleurtScorer(bleurt_checkpoint)

    return scorers

def compute_sentence_metric(
    predictions: list[str],
    references: list[str],
    metric: str = "chrf",
    sources: Optional[list[str]] = None,
    scorers: Optional[MetricScorers] = None,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> list[float]:
    """Compute per-sentence BLEU or chrF scores.

    Supported metrics:
        - "bleu"       : sacreBLEU sentence BLEU, 0-100
        - "chrf"       : chrF++, 0-100
        - "comet22"    : COMET-22, usually 0-1
        - "bleurt"     : BLEURT, unbounded-ish learned score
        - "bertscore"  : BERTScore F1, usually 0-1

    Args:
        predictions: predicted sentences
        references: reference sentences
        metric: metric name
        sources: source sentences, required for COMET-22
        scorers: preloaded MetricScorers for COMET/BLEURT
        batch_size: batch size for neural metrics
        device: "cuda", "mps", or "cpu"; mostly used for BERTScore

    Returns:
        list of per-sentence scores (0-100 scale)
    """

    if metric == "bleu":
        return [sacrebleu.sentence_bleu(pred, [ref]).score
        for pred, ref, in zip(predictions, references)
        ]
    if metric == "chrf":
        return [sacrebleu.sentence_chrf(pred, [ref]).score
        for pred, ref in zip(predictions, references)
        ]
    if metric == "comet22":
        if sources is None:
            raise ValueError("sources is required for COMET-22")
        if scorers is None or scorers.comet is None:
            raise ValueError("COMET scorer is not loaded")

        data = [
            {"src": src, "mt": pred, "ref": ref}
            for src, pred, ref in zip(sources, predictions, references)
        ]

        output = scorers.comet.predict(
            data,
            batch_size=batch_size,
            gpus=1 if torch.cuda.is_available() else 0,
        )

        # COMET versions differ slightly in return object shape.
        if hasattr(output, "scores"):
            return [float(x) for x in output.scores]
        return [float(x) for x in output["scores"]]
    if metric == "bleurt":
        if scorers is None or scorers.bleurt is None:
            raise ValueError("BLEURT scorer is not loaded")

        return [
            float(x)
            for x in scorers.bleurt.score(
                references=references,
                candidates=predictions,
            )
        ]
    if metric == "bertscore":
        from bert_score import score as bert_score

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        P, R, f1 = bert_score(
            predictions,
            references,
            lang="en",
            rescale_with_baseline=True,
            device=device,
            batch_size=batch_size,
        )
        print(type(f1))
        F1 = cast(torch.Tensor, f1)
        
        return [float(x) for x in F1.detach().cpu().tolist()]
    raise ValueError(f"Unknown metric: {metric}")

def compute_cycle_rewards(
    original_english: list[str],
    forward_translations: list[list[str]],
    back_translations: list[list[list[str]]],
    log=True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cycle-consistency rewards for GRPO.

    For each original sentence i:
      - forward_translations[i] has g candidates (eng -> target)
      - back_translations[i][j] has g candidates for forward_translations[i][j]
        (target -> eng)

    Back-translation reward: metric(back_translation, original_english)
    Forward reward for candidate j: mean of back-translation rewards
        for all back-translations of candidate j
    Total reward = alpha * forward_reward + back_reward

    Args:
        original_english: batch of original English sentences
        forward_translations: [batch_size, g] forward translations
        back_translations: [batch_size, g, g] back translations

    Returns:
        forward_rewards: [2, batch_size, g] rewards for forward step (first dim is bleu/chrf rewards)
        backward_rewards: [2, batch_size, g, g] rewards for backward step
    """
    batch_size = len(original_english)
    g_fwd = len(forward_translations[0])
    g_bwd = len(back_translations[0][0])

    backward_rewards = torch.zeros(2, batch_size, g_fwd, g_bwd)
    forward_rewards = torch.zeros(2, batch_size, g_fwd)

    for i in range(batch_size):
        for j in range(g_fwd):
            back_preds = back_translations[i][j]
            refs = [original_english[i]] * g_bwd
            bleu_scores = compute_sentence_metric(back_preds, refs, "bleu")
            chrf_scores = compute_sentence_metric(back_preds, refs, "chrf")

            for k in range(g_bwd):
                backward_rewards[0, i, j, k] = bleu_scores[k]
                backward_rewards[1, i, j, k] = chrf_scores[k]

            # Forward reward = mean of backward rewards
            forward_rewards[0, i, j] = sum(bleu_scores) / len(bleu_scores)
            forward_rewards[1, i, j] = sum(chrf_scores) / len(chrf_scores)

    if log:
        logger.info(f"""First example:
Original eng: {original_english[0]}
Forw pred: {forward_translations[0]}
Forw rwd (bleu):  {forward_rewards[0, 0].tolist()}
Forw rwd (chrf):  {forward_rewards[1, 0].tolist()}
Back pred: {pformat(back_translations[0])}
Back rwd (bleu): {backward_rewards[0, 0]}
Back rwd (chrf): {backward_rewards[1, 0]}""")

    return forward_rewards, backward_rewards
