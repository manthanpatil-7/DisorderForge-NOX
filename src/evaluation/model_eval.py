"""Model evaluation pipeline integration.

Reference: Phase 3 Plan §P3-S14

Wires model output (per-residue probabilities) into the Phase 1
evaluation orchestrator for validation monitoring and final evaluation.
"""

import torch
import numpy as np

from src.evaluation.evaluate import evaluate_on_benchmark


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    batches: list[dict],
    accession_lists: list[list[str]] | None = None,
    benchmark_name: str | None = None,
    benchmark_subtrack: str | None = None,
    device: str = "cpu",
    frozen_threshold: float | None = None,
) -> dict:
    """Run inference and evaluate using the Phase 1 metric pipeline.

    Args:
        model: Trained model producing [B, L, 1] logits.
        batches: List of batch dicts with 'features', 'labels', 'lengths'.
        accession_lists: Parallel list of accession lists per batch.
        benchmark_name: If provided, evaluate using the benchmark evaluation
            pipeline (evaluate_on_benchmark).
        benchmark_subtrack: Benchmark subtrack name.
        device: Torch device.
        frozen_threshold: Threshold for binary predictions.

    Returns:
        Dict with predictions and optional EvaluationReport.
    """
    model.eval()
    model = model.to(device)

    # Collect per-protein predictions
    predictions = {}  # accession → numpy probability array

    for batch_idx, batch in enumerate(batches):
        features = batch["features"].to(device)
        labels = batch["labels"]
        lengths = batch.get("lengths")

        # Forward pass
        if lengths is not None:
            try:
                logits = model(features, lengths=lengths)
            except TypeError:
                logits = model(features)
        else:
            logits = model(features)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

        # Map back to accessions
        if accession_lists is not None and batch_idx < len(accession_lists):
            batch_accs = accession_lists[batch_idx]
        else:
            # Use batch index as fallback
            batch_accs = [f"protein_{batch_idx}_{i}" for i in range(probs.shape[0])]

        for i, acc in enumerate(batch_accs):
            seq_len = int(lengths[i]) if lengths is not None else probs.shape[1]
            predictions[acc] = probs[i, :seq_len]

    result = {"predictions": predictions}

    # Evaluate against benchmark if requested
    if benchmark_name is not None:
        report = evaluate_on_benchmark(
            predictions=predictions,
            benchmark_name=benchmark_name,
            subtrack=benchmark_subtrack,
            frozen_threshold=frozen_threshold,
        )
        result["report"] = report

    return result
