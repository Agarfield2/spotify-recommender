import json, os, glob, pickle
from collections import defaultdict
from typing import Tuple, Dict, List
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


def load_mpd_slices(data_dir: str, max_slices: int = None) -> List[dict]:
    pattern = os.path.join(data_dir, "mpd.slice.*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier MPD trouvé dans {data_dir}.\n"
            "Télécharge sur https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge"
        )
    if max_slices is not None:
        files = files[:max_slices]
    playlists = []
    for path in tqdm(files, desc="Chargement des slices"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        playlists.extend(data["playlists"])
    print(f"  {len(playlists):,} playlists chargées depuis {len(files)} fichiers.")
    return playlists


def build_mappings(playlists: List[dict]) -> Tuple[Dict, Dict, Dict, Dict]:
    pid2idx, idx2pid = {}, {}
    track2idx, idx2track = {}, {}
    for playlist in playlists:
        pid = playlist["pid"]
        if pid not in pid2idx:
            idx = len(pid2idx)
            pid2idx[pid] = idx
            idx2pid[idx] = pid
        for track in playlist["tracks"]:
            uri = track["track_uri"]
            if uri not in track2idx:
                t_idx = len(track2idx)
                track2idx[uri] = t_idx
                idx2track[t_idx] = uri
    print(f"  {len(pid2idx):,} playlists  |  {len(track2idx):,} tracks uniques")
    return pid2idx, track2idx, idx2pid, idx2track


def build_interaction_matrix(playlists, pid2idx, track2idx) -> sp.csr_matrix:
    rows, cols = [], []
    for playlist in playlists:
        p_idx = pid2idx[playlist["pid"]]
        for track in playlist["tracks"]:
            rows.append(p_idx)
            cols.append(track2idx[track["track_uri"]])
    n_p, n_t = len(pid2idx), len(track2idx)
    matrix = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_p, n_t)
    )
    density = matrix.nnz / (n_p * n_t) * 100
    print(f"  Matrice : {n_p:,} x {n_t:,}  |  {matrix.nnz:,} interactions  |  densité {density:.4f}%")
    return matrix


def build_sequences(playlists, pid2idx, track2idx) -> Dict[int, List[int]]:
    sequences = {}
    for playlist in playlists:
        p_idx = pid2idx[playlist["pid"]]
        sorted_tracks = sorted(playlist["tracks"], key=lambda t: t["pos"])
        sequences[p_idx] = [track2idx[t["track_uri"]] for t in sorted_tracks]
    return sequences


def train_val_test_split(matrix: sp.csr_matrix, val_ratio=0.05, test_ratio=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    indices = rng.permutation(n)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_idx = indices[:n_test]
    val_idx  = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    print(f"  Split → train: {len(train_idx):,}  val: {len(val_idx):,}  test: {len(test_idx):,}")
    return matrix[train_idx], matrix[val_idx], matrix[test_idx]


def save_preprocessed(output_dir: str, **objects):
    os.makedirs(output_dir, exist_ok=True)
    for name, obj in objects.items():
        path = os.path.join(output_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Sauvegardé : {path}")


def load_preprocessed(output_dir: str) -> dict:
    objects = {}
    for path in glob.glob(os.path.join(output_dir, "*.pkl")):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, "rb") as f:
            objects[name] = pickle.load(f)
    return objects


def run_preprocessing(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    max_slices: int = None,
    batch_size: int = 50,
):


    import gc

    pattern = os.path.join(raw_dir, "mpd.slice.*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Aucun fichier MPD trouvé dans {raw_dir}.")
    if max_slices is not None:
        files = files[:max_slices]

    print(f"\n=== Preprocessing MPD ({len(files)} fichiers, batch={batch_size}) ===")

    
    # Passe 1 : construction des mappings (lecture légère)
    
    print("\n[1/4] Construction des mappings (passe 1)...")
    pid2idx: Dict[int, int] = {}
    idx2pid: Dict[int, int] = {}
    track2idx: Dict[str, int] = {}
    idx2track: Dict[int, str] = {}

    for i, path in enumerate(tqdm(files, desc="Mappings")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for playlist in data["playlists"]:
            pid = playlist["pid"]
            if pid not in pid2idx:
                idx = len(pid2idx)
                pid2idx[pid] = idx
                idx2pid[idx] = pid
            for track in playlist["tracks"]:
                uri = track["track_uri"]
                if uri not in track2idx:
                    t_idx = len(track2idx)
                    track2idx[uri] = t_idx
                    idx2track[t_idx] = uri
        del data
        gc.collect()

    n_playlists = len(pid2idx)
    n_tracks = len(track2idx)
    print(f"  {n_playlists:,} playlists  |  {n_tracks:,} tracks uniques")

    
    # Passe 2 : construction de la matrice COO par batch
    
    print("\n[2/4] Construction de la matrice d'interaction (passe 2)...")
    rows_all: List[int] = []
    cols_all: List[int] = []

    for path in tqdm(files, desc="Matrice COO"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for playlist in data["playlists"]:
            p_idx = pid2idx[playlist["pid"]]
            for track in playlist["tracks"]:
                rows_all.append(p_idx)
                cols_all.append(track2idx[track["track_uri"]])
        del data
        gc.collect()

    data_vals = np.ones(len(rows_all), dtype=np.float32)
    interaction_matrix = sp.csr_matrix(
        (data_vals, (rows_all, cols_all)),
        shape=(n_playlists, n_tracks)
    )
    del rows_all, cols_all, data_vals
    gc.collect()

    density = interaction_matrix.nnz / (n_playlists * n_tracks) * 100
    print(f"  Matrice : {n_playlists:,} x {n_tracks:,}  |  "
          f"{interaction_matrix.nnz:,} interactions  |  densité {density:.4f}%")

    
    # Split
    
    print("\n[3/4] Split train / val / test...")
    train_mat, val_mat, test_mat = train_val_test_split(interaction_matrix)

    
    # Sauvegarde
    
    print("\n[4/4] Sauvegarde...")
    save_preprocessed(
        processed_dir,
        pid2idx=pid2idx,
        track2idx=track2idx,
        idx2pid=idx2pid,
        idx2track=idx2track,
        interaction_matrix=interaction_matrix,
        train_matrix=train_mat,
        val_matrix=val_mat,
        test_matrix=test_mat,
    )
    # Note : on ne sauvegarde plus `playlists` (trop lourd en RAM sur 1M playlists)

    print(f"\n✓ Preprocessing terminé. Artefacts dans '{processed_dir}/'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--max_slices", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Nombre de fichiers traités à la fois (réduit si MemoryError)")
    args = parser.parse_args()
    run_preprocessing(args.raw_dir, args.processed_dir, args.max_slices, args.batch_size)