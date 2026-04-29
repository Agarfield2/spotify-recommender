from __future__ import annotations
import numpy as np
from typing import Sequence


def r_precision(recommended: Sequence[int], ground_truth: set[int]) -> float:

    r = len(ground_truth)
    if r == 0:
        return 0.0
    top_r = set(list(recommended)[:r])
    return len(top_r & ground_truth) / r


def ndcg_at_k(recommended: Sequence[int], ground_truth: set[int], k: int = 100) -> float:

    rec = list(recommended)[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(rec)
        if item in ground_truth
    )
    n_relevant = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(recommended: Sequence[int], ground_truth: set[int], k: int = 100) -> float:

    if not ground_truth:
        return 0.0
    hits = sum(1 for item in list(recommended)[:k] if item in ground_truth)
    return hits / len(ground_truth)


def clicks(recommended: Sequence[int], ground_truth: set[int], step: int = 10) -> float:

    rec = list(recommended)[:500]
    for i, item in enumerate(rec):
        if item in ground_truth:
            return i // step
    return 51.0


def average_precision(recommended: Sequence[int], ground_truth: set[int]) -> float:

    if not ground_truth:
        return 0.0
    rec = list(recommended)
    hits, ap = 0, 0.0
    for i, item in enumerate(rec):
        if item in ground_truth:
            hits += 1
            ap += hits / (i + 1)
    return ap / len(ground_truth)


def evaluate_batch(
    recommendations: list[Sequence[int]],
    ground_truths: list[set[int]],
    k: int = 100,
    verbose: bool = True,
) -> dict[str, float]:

    assert len(recommendations) == len(ground_truths)

    rp_scores, ndcg_scores, recall_scores, click_scores, ap_scores = [], [], [], [], []

    for rec, gt in zip(recommendations, ground_truths):
        if not gt:
            continue
        rp_scores.append(r_precision(rec, gt))
        ndcg_scores.append(ndcg_at_k(rec, gt, k=k))
        recall_scores.append(recall_at_k(rec, gt, k=k))
        click_scores.append(clicks(rec, gt))
        ap_scores.append(average_precision(rec, gt))

    results = {
        "r_precision": float(np.mean(rp_scores)),
        "ndcg":        float(np.mean(ndcg_scores)),
        "recall":      float(np.mean(recall_scores)),
        "clicks":      float(np.mean(click_scores)),
        "map":         float(np.mean(ap_scores)),
        "n_evaluated": len(rp_scores),
    }

    if verbose:
        print("\n" + "=" * 45)
        print(f"  Évaluation sur {results['n_evaluated']:,} playlists")
        print("=" * 45)
        print(f"  R-precision : {results['r_precision']:.4f}")
        print(f"  NDCG@{k:<4}   : {results['ndcg']:.4f}")
        print(f"  Recall@{k:<4} : {results['recall']:.4f}")
        print(f"  Clicks      : {results['clicks']:.4f}  (↓ meilleur)")
        print(f"  MAP         : {results['map']:.4f}")
        print("=" * 45)

    return results


def evaluate_model(
    model_recommend_fn,
    test_playlists: list[tuple],
    track_to_idx: dict[str, int],
    n_recommendations: int = 500,
    verbose: bool = True,
    sample_size: int | None = None,
) -> dict[str, float]:

    from tqdm import tqdm

    playlists = test_playlists
    if sample_size and sample_size < len(playlists):
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(playlists), size=sample_size, replace=False)
        playlists = [playlists[i] for i in idxs]

    all_recs, all_gts = [], []

    for pid, context_uris, held_out_uris in tqdm(playlists, desc="Évaluation"):
        context_idxs = [track_to_idx[u] for u in context_uris if u in track_to_idx]
        held_out_idxs = {track_to_idx[u] for u in held_out_uris if u in track_to_idx}

        if not context_idxs or not held_out_idxs:
            continue

        try:
            rec_idxs, _ = model_recommend_fn(context_idxs)
            all_recs.append(rec_idxs[:n_recommendations])
            all_gts.append(held_out_idxs)
        except Exception as e:
            if verbose:
                print(f"Erreur playlist {pid}: {e}")

    return evaluate_batch(all_recs, all_gts, k=n_recommendations, verbose=verbose)


def matrix_to_ground_truths(test_matrix, n_seed: int = 5, seed: int = 42):

    rng = np.random.default_rng(seed)
    seed_dict, gt_dict = {}, {}
    for p_idx in range(test_matrix.shape[0]):
        row = test_matrix.getrow(p_idx)
        tracks = row.indices.tolist()
        if len(tracks) < n_seed + 1:
            continue
        rng.shuffle(tracks)
        seed_dict[p_idx] = tracks[:n_seed]
        gt_dict[p_idx] = set(tracks[n_seed:])
    return seed_dict, gt_dict


def compare_models(results: dict[str, dict[str, float]]) -> None:

    metrics = ["r_precision", "ndcg", "recall", "clicks", "map"]
    col_w = 14
    header = f"{'Modele':<20}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print("\n" + "=" * (20 + col_w * len(metrics)))
    print(header)
    print("-" * (20 + col_w * len(metrics)))
    for model_name, res in results.items():
        row = f"{model_name:<20}"
        for m in metrics:
            val = res.get(m, float("nan"))
            row += f"{val:>{col_w}.4f}"
        print(row)
    print("=" * (20 + col_w * len(metrics)))
    print("  (haut meilleur : R-precision, NDCG, Recall, MAP  |  bas meilleur : Clicks)\n")
