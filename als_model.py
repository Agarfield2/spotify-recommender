import os
import pickle
import time
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

try:
    import implicit
    from implicit.als import AlternatingLeastSquares
except ImportError:
    raise ImportError(
        "La librairie 'implicit' est requise.\n"
        "Installe-la avec : pip install implicit"
    )


class ALSRecommender:

    def __init__(
        self,
        factors: int = 128,
        regularization: float = 0.01,
        iterations: int = 20,
        use_gpu: bool = False,
        random_state: int = 42,
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.model = None
        self.train_matrix = None

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=self.use_gpu,
            random_state=self.random_state,
            calculate_training_loss=True,
        )

    
    # Entraînement
    

    def fit(self, train_matrix: sp.csr_matrix):
        self.train_matrix = train_matrix

        # implicit attend items × users (tracks × playlists)
        item_user_matrix = train_matrix.T.tocsr()

        print(f"\n=== Entraînement ALS ===")
        print(f"  Facteurs       : {self.factors}")
        print(f"  Régularisation : {self.regularization}")
        print(f"  Itérations     : {self.iterations}")
        print(f"  Matrice        : {train_matrix.shape[0]:,} playlists x "
              f"{train_matrix.shape[1]:,} tracks")

        start = time.time()
        self.model.fit(item_user_matrix)
        elapsed = time.time() - start
        print(f"\nOK Entraînement terminé en {elapsed:.1f}s")

    
    # Recommandation
    

    def recommend(
        self,
        playlist_idx: int,
        n: int = 500,
        filter_already_liked: bool = True
    ) -> List[int]:
        
        ids, _ = self.model.recommend(
            userid=playlist_idx,
            user_items=self.train_matrix[playlist_idx],
            N=n,
            filter_already_liked_items=filter_already_liked,
        )
        return ids.tolist()

    def recommend_from_tracks(
        self,
        seed_track_indices: List[int],
        n: int = 500
    ) -> List[int]:
        
        # item_factors = tracks car on a fit sur tracks × playlists
        track_factors = self.model.item_factors  # (n_tracks, k)

        # Vérification de cohérence au premier appel
        n_tracks = self.train_matrix.shape[1]
        if track_factors.shape[0] != n_tracks:
            # implicit a parfois les axes inversés selon la version
            # dans ce cas user_factors contient les tracks
            track_factors = self.model.user_factors  # (n_tracks, k)

        seed_factors = track_factors[seed_track_indices]
        virtual_user = seed_factors.mean(axis=0)

        scores = track_factors @ virtual_user
        # Exclure les seeds
        for idx in seed_track_indices:
            scores[idx] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n]
        return top_indices.tolist()

    def recommend_batch(
        self,
        playlist_indices: List[int],
        n: int = 500,
        filter_already_liked: bool = True
    ) -> Dict[int, List[int]]:
        
        all_ids, _ = self.model.recommend(
            userid=np.array(playlist_indices),
            user_items=self.train_matrix[playlist_indices],
            N=n,
            filter_already_liked_items=filter_already_liked,
        )
        return {playlist_indices[i]: all_ids[i].tolist()
                for i in range(len(playlist_indices))}

    
    # Similarité
    

    def similar_tracks(self, track_idx: int, n: int = 20) -> List[Tuple[int, float]]:
        
        ids, scores = self.model.similar_items(track_idx, N=n + 1)
        return list(zip(ids[1:].tolist(), scores[1:].tolist()))

    def similar_playlists(self, playlist_idx: int, n: int = 20) -> List[Tuple[int, float]]:
        
        ids, scores = self.model.similar_users(playlist_idx, N=n + 1)
        return list(zip(ids[1:].tolist(), scores[1:].tolist()))

    
    # Sauvegarde / chargement
    

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "train_matrix": self.train_matrix,
                "factors": self.factors,
                "regularization": self.regularization,
                "iterations": self.iterations,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"OK Modèle ALS sauvegardé : {path}")

    @classmethod
    def load(cls, path: str) -> "ALSRecommender":
        with open(path, "rb") as f:
            state = pickle.load(f)
        instance = cls(
            factors=state["factors"],
            regularization=state["regularization"],
            iterations=state["iterations"],
        )
        instance.model = state["model"]
        instance.train_matrix = state["train_matrix"]
        print(f"OK Modèle ALS chargé : {path}")
        return instance


# Baselines de référence


class PopularityBaseline:

    def __init__(self):
        self.popular_tracks = None

    def fit(self, train_matrix: sp.csr_matrix):
        track_counts = np.asarray(train_matrix.sum(axis=0)).flatten()
        self.popular_tracks = np.argsort(track_counts)[::-1].tolist()
        print(f"OK Baseline popularité entraîné sur {train_matrix.shape[0]:,} playlists")

    def recommend(self, n: int = 500, exclude: List[int] = None) -> List[int]:
        exclude_set = set(exclude or [])
        result = [t for t in self.popular_tracks if t not in exclude_set]
        return result[:n]

    def recommend_batch(
        self,
        seed_dict: Dict[int, List[int]],
        n: int = 500
    ) -> Dict[int, List[int]]:
        return {
            p_idx: self.recommend(n=n, exclude=seeds)
            for p_idx, seeds in seed_dict.items()
        }


# Script d'entraînement standalone

if __name__ == "__main__":
    import argparse
    import sys

    # Permet de lancer le script depuis n'importe quel dossier
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)          # dossier parent = racine du projet
    sys.path.insert(0, _root)               # pour "from evaluation.metrics import ..."
    sys.path.insert(0, _here)               # pour "from metrics import ..." si même dossier

    # Essai import flexible : d'abord depuis la racine, sinon cherche metrics.py à plat
    try:
        from evaluation.metrics import evaluate_batch, matrix_to_ground_truths
    except ModuleNotFoundError:
        try:
            from metrics import evaluate_batch, matrix_to_ground_truths
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Impossible de trouver metrics.py.\n"
                "Lance le script depuis la racine du projet :\n"
                "  python models/als_model.py\n"
                "ou copie metrics.py dans le même dossier."
            )

    try:
        from data.data_loader import load_preprocessed
    except ModuleNotFoundError:
        from data_loader import load_preprocessed

    parser = argparse.ArgumentParser(description="Entraînement ALS sur le MPD")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--factors", type=int, default=128)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", default="models/saved/als_model.pkl")
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    print("Chargement des données preprocessées...")
    data = load_preprocessed(args.processed_dir)

    # Diagnostic — affiche les clés disponibles si une clé attendue est absente
    available_keys = list(data.keys())
    print(f"  Clés disponibles : {available_keys}")

    # Noms alternatifs possibles selon la version du data_loader utilisée
    def get_key(d, *candidates):
        for k in candidates:
            if k in d:
                return d[k]
        raise KeyError(
            f"Aucune des clés {candidates} trouvée.\n"
            f"Clés disponibles : {list(d.keys())}"
        )

    train_matrix = get_key(data, "train_matrix", "train", "train_mat")
    test_matrix  = get_key(data, "test_matrix",  "test",  "test_mat")

    # Entraînement ALS
    model = ALSRecommender(
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        use_gpu=args.use_gpu,
    )
    model.fit(train_matrix)
    model.save(args.output)

    # Baseline popularité
    print("\n=== Baseline Popularité ===")
    pop = PopularityBaseline()
    pop.fit(train_matrix)

    # Ground truths depuis le test set
    seed_dict, gt_dict = matrix_to_ground_truths(test_matrix, n_seed=5)

    # Évaluation ALS
    print("\n--- Résultats ALS ---")
    recs_als, gts = [], []
    for p_idx, seeds in tqdm(seed_dict.items(), desc="ALS recommend"):
        recs_als.append(model.recommend_from_tracks(seeds, n=500))
        gts.append(gt_dict[p_idx])
    evaluate_batch(recs_als, gts)

    # Évaluation popularité
    print("\n--- Résultats Popularité (baseline) ")
    recs_pop, gts = [], []
    for p_idx, seeds in seed_dict.items():
        recs_pop.append(pop.recommend(n=500, exclude=seeds))
        gts.append(gt_dict[p_idx])
    evaluate_batch(recs_pop, gts)