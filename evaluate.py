"""
evaluate.py
-----------
Évaluation rapide d'un modèle LightGCN ou ALS déjà entraîné.

Usage :
    # LightGCN
    python evaluate.py --model lightgcn --path models/saved/lightgcn.pt --processed_dir data/processed

    # ALS
    python evaluate.py --model als --path models/saved/als_model.pkl --processed_dir data/processed

    # Les deux
    python evaluate.py --model both --processed_dir data/processed

Options :
    --n_seed    : nombre de tracks seeds par playlist (défaut: 5)
    --n_recs    : nombre de recommandations à générer  (défaut: 500)
    --k         : seuil K pour NDCG@K et R@K           (défaut: 100)
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def ndcg_at_k(recommended, ground_truth, k):
    gt_set = set(ground_truth)
    dcg  = sum(1.0 / np.log2(i + 2) for i, t in enumerate(recommended[:k]) if t in gt_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(recommended, ground_truth, k):
    gt_set = set(ground_truth)
    hits = sum(1 for t in recommended[:k] if t in gt_set)
    return hits / len(gt_set) if gt_set else 0.0

def r_precision(recommended, ground_truth):
    gt_set = set(ground_truth)
    r = len(gt_set)
    hits = sum(1 for t in recommended[:r] if t in gt_set)
    return hits / r if r > 0 else 0.0

def evaluate_batch(recs_list, gts_list, k=100):
    ndcgs, recalls, rprecs = [], [], []
    for recs, gt in zip(recs_list, gts_list):
        if not gt:
            continue
        ndcgs.append(ndcg_at_k(recs, gt, k))
        recalls.append(recall_at_k(recs, gt, k))
        rprecs.append(r_precision(recs, gt))
    print(f"  Playlists évaluées : {len(ndcgs):,}")
    print(f"  NDCG@{k}      : {np.mean(ndcgs):.4f}")
    print(f"  Recall@{k}    : {np.mean(recalls):.4f}")
    print(f"  R-Precision   : {np.mean(rprecs):.4f}")
    return {"ndcg": np.mean(ndcgs), "recall": np.mean(recalls), "r_precision": np.mean(rprecs)}


def matrix_to_ground_truths(test_matrix, n_seed=5):
    """
    Pour chaque playlist du test set ayant assez de tracks :
      - seed_dict[p_idx]  = les n_seed premiers tracks (entrée)
      - gt_dict[p_idx]    = le reste (vérité terrain)
    """
    seed_dict, gt_dict = {}, {}
    for p_idx in range(test_matrix.shape[0]):
        row = test_matrix.getrow(p_idx)
        tracks = row.indices.tolist()
        if len(tracks) < n_seed + 1:
            continue
        seed_dict[p_idx] = tracks[:n_seed]
        gt_dict[p_idx]   = tracks[n_seed:]
    print(f"  {len(seed_dict):,} playlists de test utilisables (≥ {n_seed+1} tracks)")
    return seed_dict, gt_dict


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_test_matrix(processed_dir):
    path = os.path.join(processed_dir, "test_matrix.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"test_matrix.pkl introuvable dans {processed_dir}")
    print(f"Chargement de test_matrix.pkl...")
    with open(path, "rb") as f:
        mat = pickle.load(f)
    print(f"  Matrice test : {mat.shape[0]:,} playlists x {mat.shape[1]:,} tracks")
    return mat


# ---------------------------------------------------------------------------
# Évaluation LightGCN
# ---------------------------------------------------------------------------

def evaluate_lightgcn(path, test_matrix, seed_dict, gt_dict, n_recs, k):
    print(f"\n{'='*50}")
    print(f"  LightGCN — {path}")
    print(f"{'='*50}")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lightgcn_modelv4_2 import LightGCNTrainer

    print("Chargement du modèle...")
    trainer = LightGCNTrainer.load(path)

    print("Calcul des embeddings (1 forward pass)...")
    t0 = time.time()
    t_embs = trainer._get_track_embs()
    print(f"  Embeddings calculés en {time.time()-t0:.1f}s  —  shape: {t_embs.shape}")

    print(f"Recommandation pour {len(seed_dict):,} playlists...")
    recs_list, gts_list = [], []
    t0 = time.time()
    for p_idx, seeds in tqdm(seed_dict.items(), desc="LightGCN"):
        recs_list.append(trainer._seeds_to_recs(t_embs, seeds, n=n_recs))
        gts_list.append(gt_dict[p_idx])
    print(f"  Recommandations générées en {time.time()-t0:.1f}s")

    print("\nRésultats :")
    return evaluate_batch(recs_list, gts_list, k=k)


# ---------------------------------------------------------------------------
# Évaluation ALS
# ---------------------------------------------------------------------------

def evaluate_als(path, test_matrix, seed_dict, gt_dict, n_recs, k):
    print(f"\n{'='*50}")
    print(f"  ALS — {path}")
    print(f"{'='*50}")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from als_model import ALSRecommender

    print("Chargement du modèle...")
    model = ALSRecommender.load(path)

    print(f"Recommandation pour {len(seed_dict):,} playlists...")
    recs_list, gts_list = [], []
    t0 = time.time()
    for p_idx, seeds in tqdm(seed_dict.items(), desc="ALS"):
        recs_list.append(model.recommend_from_tracks(seeds, n=n_recs))
        gts_list.append(gt_dict[p_idx])
    print(f"  Recommandations générées en {time.time()-t0:.1f}s")

    print("\nRésultats :")
    return evaluate_batch(recs_list, gts_list, k=k)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation rapide LightGCN / ALS")
    parser.add_argument("--model",         choices=["lightgcn", "als", "both"], default="both")
    parser.add_argument("--lightgcn_path", default="models/saved/lightgcn.pt")
    parser.add_argument("--als_path",      default="models/saved/als_model.pkl")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--n_seed",        type=int, default=5,   help="Tracks seeds par playlist")
    parser.add_argument("--n_recs",        type=int, default=500, help="Nb de recommandations")
    parser.add_argument("--k",             type=int, default=100, help="K pour NDCG@K et Recall@K")
    args = parser.parse_args()

    # Chargement du test set
    test_matrix = load_test_matrix(args.processed_dir)

    print(f"\nConstruction des ground truths (n_seed={args.n_seed})...")
    seed_dict, gt_dict = matrix_to_ground_truths(test_matrix, n_seed=args.n_seed)

    results = {}

    # LightGCN
    if args.model in ("lightgcn", "both"):
        if not os.path.exists(args.lightgcn_path):
            print(f"\n[LightGCN] Fichier introuvable : {args.lightgcn_path}")
        else:
            results["lightgcn"] = evaluate_lightgcn(
                args.lightgcn_path, test_matrix, seed_dict, gt_dict, args.n_recs, args.k
            )

    # ALS
    if args.model in ("als", "both"):
        if not os.path.exists(args.als_path):
            print(f"\n[ALS] Fichier introuvable : {args.als_path}")
        else:
            results["als"] = evaluate_als(
                args.als_path, test_matrix, seed_dict, gt_dict, args.n_recs, args.k
            )

    # Comparaison finale
    if len(results) == 2:
        print(f"\n{'='*50}")
        print("  Comparaison finale")
        print(f"{'='*50}")
        print(f"  {'Métrique':<15} {'LightGCN':>12} {'ALS':>12}")
        print(f"  {'-'*39}")
        for metric in ["ndcg", "recall", "r_precision"]:
            gcn = results["lightgcn"][metric]
            als = results["als"][metric]
            winner = "← GCN" if gcn > als else "← ALS"
            print(f"  {metric:<15} {gcn:>12.4f} {als:>12.4f}  {winner}")
