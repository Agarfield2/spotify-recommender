import gc
import os
import time
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError("PyTorch requis : pip install torch")



# Construction du graphe biparti


def build_norm_adj(interaction_matrix: sp.csr_matrix, device: torch.device) -> torch.Tensor:

    n_playlists, n_tracks = interaction_matrix.shape
    N = n_playlists + n_tracks

    csr = interaction_matrix.tocsr()
    deg_playlist = np.diff(csr.indptr).astype(np.float32)
    deg_track    = np.asarray(csr.sum(axis=0)).flatten().astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        dp_inv = np.where(deg_playlist > 0, deg_playlist ** -0.5, 0.0).astype(np.float32)
        dt_inv = np.where(deg_track    > 0, deg_track    ** -0.5, 0.0).astype(np.float32)

    row_lengths = np.diff(csr.indptr)
    dp_inv_rep  = np.repeat(dp_inv, row_lengths)
    dt_inv_col  = dt_inv[csr.indices]
    vals_pt     = (dp_inv_rep * dt_inv_col).astype(np.float32)
    del dp_inv_rep, dt_inv_col
    gc.collect()

    # Indices COO pour les deux blocs off-diagonaux du graphe biparti
    # Bloc haut-droit : (playlist_row, n_playlists + track_col)
    # Bloc bas-gauche : (n_playlists + track_col, playlist_row)  [transposé]
    csr_rows = np.repeat(np.arange(n_playlists, dtype=np.int32), row_lengths)  # lignes playlist
    csr_cols = csr.indices.astype(np.int32)                                     # cols track

    # Concaténation des deux blocs - 2 * nnz entrées au total
    rows = np.concatenate([csr_rows,                   n_playlists + csr_cols])
    cols = np.concatenate([n_playlists + csr_cols,     csr_rows              ])
    vals = np.concatenate([vals_pt,                    vals_pt               ])
    del csr_rows, csr_cols, vals_pt
    gc.collect()

    nnz = len(vals)
    print(f"  Graphe : {n_playlists:,} playlists + {n_tracks:,} tracks | "
          f"{nnz:,} aretes (x2 non-dirige)")

    # Conversion en torch sparse COO
    indices = torch.from_numpy(np.stack([rows, cols], axis=0).astype(np.int64))
    values  = torch.from_numpy(vals)
    del rows, cols, vals
    gc.collect()

    adj = torch.sparse_coo_tensor(indices, values, size=(N, N), dtype=torch.float32)
    adj = adj.coalesce()   # trie + déduplique - requis avant envoi sur CUDA
    del indices, values
    gc.collect()

    adj = adj.to(device)

    # --- Diagnostic : vérifie que le spmm tourne bien sur le device cible ---
    print(f"  Test spmm sur {device}...", end=" ", flush=True)
    t0 = time.time()
    with torch.no_grad():
        test_x = torch.randn(N, 4, device=device)
        _ = torch.sparse.mm(adj, test_x)
        if str(device) != "cpu":
            torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"{elapsed:.2f}s", end="")
    if elapsed > 5.0:
        print("  Erreur  LENT - le spmm tourne peut-être en fallback CPU !")
        print("     -> Essaie : pip install torch --upgrade")
    else:
        print("  OK")

    return adj



# Couche de propagation LightGCN (sparse matmul)


class LightGCNConv(nn.Module):
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adj, x)



# Modèle LightGCN


class LightGCN(nn.Module):
    def __init__(
        self,
        n_playlists: int,
        n_tracks: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        self.n_playlists   = n_playlists
        self.n_tracks      = n_tracks
        self.n_nodes       = n_playlists + n_tracks
        self.embedding_dim = embedding_dim
        self.n_layers      = n_layers

        self.embedding = nn.Embedding(self.n_nodes, embedding_dim)
        self.conv      = LightGCNConv()

        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, adj: torch.Tensor) -> torch.Tensor:

        e   = self.embedding.weight   # (N, d)
        acc = e.clone()
        for _ in range(self.n_layers):
            e   = self.conv(e, adj)
            acc = acc + e
        return acc / (self.n_layers + 1)

    def score(self, p_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return (p_emb * t_emb).sum(dim=-1)



# BPR Loss


def bpr_loss(pos_scores, neg_scores, l2_reg, embs):
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    reg  = l2_reg * (embs ** 2).mean()
    return loss + reg



# Pre-sampler vectorisé (remplace BPRDataset + DataLoader)


class VectorizedBPRSampler:


    def __init__(self, interaction_matrix: sp.csr_matrix):
        self.n_tracks = interaction_matrix.shape[1]
        coo = interaction_matrix.tocoo()

        self.pos_users = coo.row.astype(np.int32)
        self.pos_items = coo.col.astype(np.int32)

        # Structure CSR NumPy pour la rejection sampling vectorisée
        csr = interaction_matrix.tocsr()
        self.indptr  = csr.indptr.astype(np.int32)
        self.indices = csr.indices.astype(np.int32)

        n = len(self.pos_users)
        _mb = (self.pos_users.nbytes + self.pos_items.nbytes
               + self.indptr.nbytes + self.indices.nbytes) / 1024**2
        print(f"  BPR sampler : {n:,} interactions | {_mb:.0f} MB")

    def sample_epoch(self, device: torch.device) -> tuple:

        n   = len(self.pos_users)
        neg = np.random.randint(0, self.n_tracks, size=n, dtype=np.int32)

        mask = neg == self.pos_items
        while mask.any():
            neg[mask] = np.random.randint(0, self.n_tracks, size=int(mask.sum()), dtype=np.int32)
            mask = neg == self.pos_items

        perm = np.random.permutation(n).astype(np.int64)

        # Transfert unique CPU -> GPU (~1-2s pour 3 × 450 MB, puis plus jamais)
        p_arr   = torch.from_numpy(self.pos_users[perm].astype(np.int64)).to(device)
        pos_arr = torch.from_numpy(self.pos_items[perm].astype(np.int64)).to(device)
        neg_arr = torch.from_numpy(neg[perm].astype(np.int64)).to(device)
        return p_arr, pos_arr, neg_arr



# Entraîneur


class LightGCNTrainer:


    def __init__(
        self,
        n_playlists: int,
        n_tracks: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        lr: float = 1e-3,
        l2_reg: float = 1e-4,
        batch_size: int = 4096,
        n_epochs: int = 50,
        device: str = "auto",
        eval_every: int = 5,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"  Device : {self.device}")
        self.n_playlists = n_playlists
        self.n_tracks    = n_tracks
        self.lr          = lr
        self.l2_reg      = l2_reg
        self.batch_size  = batch_size
        self.n_epochs    = n_epochs
        self.eval_every  = eval_every

        self.model = LightGCN(
            n_playlists   = n_playlists,
            n_tracks      = n_tracks,
            embedding_dim = embedding_dim,
            n_layers      = n_layers,
        ).to(self.device)

        # SparseAdam : optimizer conçu pour les embeddings creux.
        # Contrairement à Adam dense, il ne met à jour que les lignes actives
        # du batch -> accès GPU bien cachés même sur 3.16M embeddings.
        # Prérequis : embedding doit être sparse=True pour que le gradient
        # soit un SparseTensor -> SparseAdam peut faire ses updates sélectifs.
        self.model.embedding = nn.Embedding(
            self.model.n_nodes, self.model.embedding_dim, sparse=True
        ).to(self.device)
        nn.init.normal_(self.model.embedding.weight, std=0.01)

        self.optimizer = torch.optim.SparseAdam(
            list(self.model.parameters()), lr=lr
        )
        self._adagrad_state: Optional[torch.Tensor] = None
        self.adj: Optional[torch.Tensor] = None
        self.history = {"train_loss": [], "val_ndcg": []}

    
    # Entraînement

    def fit(
        self,
        train_matrix: sp.csr_matrix,
        val_matrix: Optional[sp.csr_matrix] = None,
        checkpoint_dir: str = "models/checkpoints",
        checkpoint_every: int = 5,
    ):
        print(f"\n=== Entraînement LightGCN ===")
        print(f"  Dim embedding : {self.model.embedding_dim} | "
              f"Couches : {self.model.n_layers}")
        print(f"  LR : {self.lr} | L2 : {self.l2_reg} | "
              f"Batch : {self.batch_size} | Epochs : {self.n_epochs}")
        print(f"  Checkpoint tous les {checkpoint_every} epochs -> {checkpoint_dir}/")

        # Construit l'adj et le met directement sur le bon device
        self.adj = build_norm_adj(train_matrix, self.device)

        # Budget mémoire estimé
        N       = self.n_playlists + self.n_tracks
        dim     = self.model.embedding_dim
        gb_emb  = N * dim * 4 / 1024**3
        nnz     = self.adj._nnz()
        gb_adj  = (nnz * 4 + nnz * 8) / 1024**3   # vals float32 + indices int64
        gb_opt  = gb_emb
        gb_fwd  = gb_emb * 2
        print(f"  Mémoire estimée : emb={gb_emb:.1f}GB  adj={gb_adj:.1f}GB  "
              f"optim={gb_opt:.1f}GB  fwd={gb_fwd:.1f}GB  "
              f"-> total≈{gb_emb+gb_adj+gb_opt+gb_fwd:.1f}GB")

        os.makedirs(checkpoint_dir, exist_ok=True)
        sampler = VectorizedBPRSampler(train_matrix)

        t0 = time.time()
        for epoch in range(1, self.n_epochs + 1):
            t_ep = time.time()

            # Pre-sampling vectorisé + transfert GPU unique (~5-8s, 1 fois/epoch)
            print(f"  [Epoch {epoch}] Sampling + transfert GPU...", end=" ", flush=True)
            t_s = time.time()
            p_arr, pos_arr, neg_arr = sampler.sample_epoch(self.device)
            if str(self.device) != "cpu":
                torch.cuda.synchronize()
            print(f"{time.time()-t_s:.1f}s")

            loss = self._train_epoch(p_arr, pos_arr, neg_arr)
            self.history["train_loss"].append(loss)
            elapsed = time.time() - t_ep
            print(f"  Epoch {epoch:3d}/{self.n_epochs}  loss={loss:.4f}  ({elapsed:.1f}s)")

            if val_matrix is not None and epoch % self.eval_every == 0:
                ndcg = self._quick_val(val_matrix)
                self.history["val_ndcg"].append(ndcg)
                print(f"    -> Val NDCG@100 = {ndcg:.4f}")

            # Checkpoint toutes les N epochs
            if epoch % checkpoint_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"lightgcn_epoch{epoch:03d}.pt")
                self.save(ckpt_path)
                print(f"    OK Checkpoint sauvegardé : {ckpt_path}")

        print(f"\nOK Terminé en {time.time()-t0:.1f}s")

    def _train_epoch(
        self,
        p_arr:   torch.Tensor,
        pos_arr: torch.Tensor,
        neg_arr: torch.Tensor,
    ) -> float:

        self.model.train()
        total   = 0.0
        n       = len(p_arr)
        n_batch = (n + self.batch_size - 1) // self.batch_size

        # Forward GCN complet sans grad - 1 seule fois par epoch
        with torch.no_grad():
            all_embs = self.model(self.adj)   # (N, d), détaché

        p_embs = all_embs[:self.n_playlists].detach()
        t_embs = all_embs[self.n_playlists:].detach()

        for i in tqdm(range(n_batch), desc="  Batches", leave=False,
                      unit="batch", mininterval=2.0):
            sl    = slice(i * self.batch_size, (i + 1) * self.batch_size)
            p_idx = p_arr[sl]
            pos_t = pos_arr[sl]
            neg_t = neg_arr[sl]

            # Embeddings GCN (détachés - pas de grad à travers spmm)
            p_e   = p_embs[p_idx]
            pos_e = t_embs[pos_t]
            neg_e = t_embs[neg_t]

            # Embeddings de base via embedding table (sparse=True -> grad sparse)
            # On passe par embedding.weight directement pour que SparseAdam
            # reçoive un gradient SparseTensor avec uniquement les lignes actives.
            p_e_base   = self.model.embedding(p_idx)
            pos_e_base = self.model.embedding(pos_t + self.n_playlists)
            neg_e_base = self.model.embedding(neg_t + self.n_playlists)

            # Scores BPR avec embeddings GCN (pas de grad) + base (grad)
            pos_score = (p_e * pos_e).sum(-1) + (p_e_base * pos_e).sum(-1)
            neg_score = (p_e * neg_e).sum(-1) + (p_e_base * neg_e).sum(-1)

            loss = -F.logsigmoid(pos_score - neg_score).mean()
            reg  = self.l2_reg * (
                p_e_base.norm(2, dim=1).pow(2).mean() +
                pos_e_base.norm(2, dim=1).pow(2).mean() +
                neg_e_base.norm(2, dim=1).pow(2).mean()
            )
            loss = loss + reg

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total += loss.item()

        return total / n_batch

    @torch.no_grad()
    def _quick_val(self, val_matrix: sp.csr_matrix, k: int = 100) -> float:
        self.model.eval()
        all_embs   = self.model(self.adj)
        t_embs_np  = all_embs[self.n_playlists:].cpu().numpy()

        scores  = []
        n_eval  = min(200, val_matrix.shape[0])
        idxs    = np.random.choice(val_matrix.shape[0], n_eval, replace=False)
        for p_idx in idxs:
            row = val_matrix.getrow(p_idx)
            if row.nnz < 6:
                continue
            tracks       = row.indices.tolist()
            seeds, gt    = tracks[:len(tracks)//2], tracks[len(tracks)//2:]
            recs         = self._seeds_to_recs(t_embs_np, seeds, k)
            scores.append(_ndcg(recs, gt, k))
        return float(np.mean(scores)) if scores else 0.0

    
    # Recommandation
    

    @torch.no_grad()
    def _get_track_embs(self) -> np.ndarray:
        self.model.eval()
        return self.model(self.adj)[self.n_playlists:].cpu().numpy()

    def _seeds_to_recs(
        self, t_embs: np.ndarray, seeds: List[int], n: int
    ) -> List[int]:
        virtual    = t_embs[seeds].mean(axis=0)
        s          = t_embs @ virtual
        s[seeds]   = -np.inf
        return np.argsort(s)[::-1][:n].tolist()

    def recommend_from_tracks(self, seed_track_indices: List[int], n: int = 500) -> List[int]:

        t_embs = self._get_track_embs()
        return self._seeds_to_recs(t_embs, seed_track_indices, n)

    def recommend_batch(
        self, seed_dict: Dict[int, List[int]], n: int = 500
    ) -> Dict[int, List[int]]:

        t_embs = self._get_track_embs()
        return {
            p_idx: self._seeds_to_recs(t_embs, seeds, n)
            for p_idx, seeds in tqdm(seed_dict.items(), desc="LightGCN recommend")
        }

    
    # Sauvegarde / chargement
    

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state":    self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "n_playlists":    self.n_playlists,
            "n_tracks":     self.n_tracks,
            "embedding_dim": self.model.embedding_dim,
            "n_layers":     self.model.n_layers,
            "adj":          self.adj.cpu() if self.adj is not None else None,
            "history":      self.history,
            "lr":           self.lr,
            "l2_reg":       self.l2_reg,
            "batch_size":   self.batch_size,
            "n_epochs":     self.n_epochs,
        }, path)
        print(f"OK LightGCN sauvegardé : {path}")

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "LightGCNTrainer":
        ckpt    = torch.load(path, map_location="cpu")
        trainer = cls(
            n_playlists   = ckpt["n_playlists"],
            n_tracks      = ckpt["n_tracks"],
            embedding_dim = ckpt["embedding_dim"],
            n_layers      = ckpt["n_layers"],
            lr            = ckpt["lr"],
            l2_reg        = ckpt["l2_reg"],
            batch_size    = ckpt["batch_size"],
            n_epochs      = ckpt["n_epochs"],
            device        = device,
        )
        trainer.model.load_state_dict(ckpt["model_state"])
        trainer.model.to(trainer.device)
        if ckpt.get("optimizer_state") is not None:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt["adj"] is not None:
            trainer.adj = ckpt["adj"].to(trainer.device)
        trainer.history = ckpt["history"]
        print(f"OK LightGCN chargé : {path}")
        return trainer



# Utilitaire NDCG


def _ndcg(recommended, ground_truth, k):
    gt_set = set(ground_truth)
    dcg    = sum(1.0 / np.log2(i + 2) for i, t in enumerate(recommended[:k]) if t in gt_set)
    idcg   = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
    return dcg / idcg if idcg > 0 else 0.0



# Script standalone


if __name__ == "__main__":
    import argparse
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)
    sys.path.insert(0, _root)
    sys.path.insert(0, _here)

    try:
        from evaluation.metrics import evaluate_batch, matrix_to_ground_truths
    except ModuleNotFoundError:
        from metrics import evaluate_batch, matrix_to_ground_truths

    try:
        from data.data_loader import load_preprocessed
    except ModuleNotFoundError:
        from data_loader import load_preprocessed

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir",  default="data/processed")
    parser.add_argument("--embedding_dim",  type=int,   default=128)
    parser.add_argument("--n_layers",       type=int,   default=3)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--l2_reg",         type=float, default=1e-4)
    parser.add_argument("--batch_size",     type=int,   default=4096)
    parser.add_argument("--n_epochs",       type=int,   default=50)
    parser.add_argument("--output",           default="models/saved/lightgcn.pt")
    parser.add_argument("--checkpoint_dir",   default="models/checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--device",         default="auto")
    parser.add_argument("--resume", default=None, help="Chemin vers un checkpoint pour reprendre l'entraînement")
    args = parser.parse_args()

    data         = load_preprocessed(args.processed_dir)
    train_matrix = data["train_matrix"]
    val_matrix   = data["val_matrix"]
    test_matrix  = data["test_matrix"]

    if args.resume is not None:
        print(f" Reprise depuis checkpoint : {args.resume}")
        trainer = LightGCNTrainer.load(args.resume, device=args.device)
    else:
        trainer = LightGCNTrainer(
            n_playlists   = train_matrix.shape[0],
            n_tracks      = train_matrix.shape[1],
            embedding_dim = args.embedding_dim,
            n_layers      = args.n_layers,
            lr            = args.lr,
            l2_reg        = args.l2_reg,
            batch_size    = args.batch_size,
            n_epochs      = args.n_epochs,
            device        = args.device,
    )
    trainer.fit(train_matrix, val_matrix,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_every=args.checkpoint_every)
    trainer.save(args.output)

    print("\n=== Evaluation test set ===")
    seed_dict, gt_dict = matrix_to_ground_truths(test_matrix, n_seed=5)
    recs_list, gts_list = [], []
    for p_idx, seeds in tqdm(seed_dict.items(), desc="LightGCN recommend"):
        recs_list.append(trainer.recommend_from_tracks(seeds, n=500))
        gts_list.append(gt_dict[p_idx])
    evaluate_batch(recs_list, gts_list)
