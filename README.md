# Spotify Playlist Recommender - LightGCN vs ALS

Système de recommandation musicale entraîné sur le **Spotify Million Playlist Dataset (MPD)**. Il compare deux approches de filtrage collaboratif : un modèle de graph neural network (**LightGCN**) et une factorisation matricielle (**ALS**), avec une interface web pour explorer les recommandations en temps réel.

---

## Fonctionnalités

- **LightGCN** : propagation de messages sur un graphe biparti playlists × tracks, entraîné par BPR loss avec SparseAdam
- **ALS** (Alternating Least Squares) : factorisation matricielle via la librairie `implicit`, rapide et efficace sur CPU
- **API Flask** servant les deux modèles en parallèle
- **Interface web** (dark mode) permettant de chercher des tracks et comparer les recommandations LightGCN vs ALS côte à côte
- **Script d'évaluation** avec métriques NDCG@K, Recall@K et R-Precision

---

## Structure du projet

```
.
├── data/
│   ├── mpd.slice.X-Y         # Slices MPD brutes (mpd.slice.*.json)
│   └── processed/            # Artefacts préprocessés (.pkl)
├── models/
│   ├── saved/
│   │   ├── lightgcn.pt       # Modèle LightGCN entraîné
│   │   └── als_model.pkl     # Modèle ALS entraîné
│   └── checkpoints/          # Checkpoints LightGCN par epoch
├── static/
│   └── index.html            # Interface web
├── lightgcn_modelv5.py       # Architecture LightGCN + entraîneur
├── als_model.py              # Modèle ALS + baseline popularité
├── data_loader.py            # Preprocessing et chargement du MPD
├── evaluate.py               # Évaluation standalone des modèles
├── app.py                    # Serveur Flask (API + frontend)
└── requirements.txt
```

---

## Prérequis

- Python 3.9+
- CUDA recommandé pour LightGCN (fonctionne aussi sur CPU mais plus lent)

```bash
pip install -r requirements.txt
```

---

## Données

Le projet utilise le **Spotify Million Playlist Dataset**, disponible sur [AICrowd](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge). Les données sont soumises à leur propre licence (voir `license.txt`).

Placez les fichiers `mpd.slice.*.json` dans `data/raw/`.

---

## Utilisation

### 1. Préprocessing

```bash
python data_loader.py --raw_dir data/raw --processed_dir data/processed
```

Options disponibles :

| Argument | Défaut | Description |
|---|---|---|
| `--raw_dir` | `data/raw` | Dossier contenant les slices MPD |
| `--processed_dir` | `data/processed` | Dossier de sortie des artefacts |
| `--max_slices` | _(tous)_ | Limite le nombre de fichiers (pratique pour les tests) |
| `--batch_size` | `50` | Fichiers traités par batch (à réduire en cas de MemoryError) |

Le preprocessing génère les fichiers suivants dans `data/processed/` :

- `pid2idx.pkl`, `idx2pid.pkl` - mappings playlists
- `track2idx.pkl`, `idx2track.pkl` - mappings tracks
- `interaction_matrix.pkl` - matrice playlists × tracks complète
- `train_matrix.pkl`, `val_matrix.pkl`, `test_matrix.pkl` - splits (90% / 5% / 5%)

---

### 2. Entraînement LightGCN

```bash
python lightgcn_modelv5.py \
    --processed_dir data/processed \
    --embedding_dim 128 \
    --n_layers 3 \
    --lr 1e-3 \
    --l2_reg 1e-4 \
    --batch_size 4096 \
    --n_epochs 50 \
    --output models/saved/lightgcn.pt \
    --checkpoint_dir models/checkpoints \
    --checkpoint_every 5
```

Pour reprendre depuis un checkpoint :

```bash
python lightgcn_modelv5.py --resume models/checkpoints/lightgcn_epoch020.pt
```

**Architecture** : graphe biparti normalisé (D⁻¹/² A D⁻¹/²), propagation multi-couche sans transformation linéaire, BPR loss avec régularisation L2, optimiseur SparseAdam.

---

### 3. Entraînement ALS

```bash
python als_model.py \
    --processed_dir data/processed \
    --factors 128 \
    --regularization 0.01 \
    --iterations 20 \
    --output models/saved/als_model.pkl
```

Ajouter `--use_gpu` pour accélérer l'entraînement ALS sur GPU.

---

### 4. Évaluation

```bash
# Les deux modèles
python evaluate.py --model both --processed_dir data/processed

# LightGCN seul
python evaluate.py --model lightgcn --lightgcn_path models/saved/lightgcn.pt

# ALS seul
python evaluate.py --model als --als_path models/saved/als_model.pkl
```

Options :

| Argument | Défaut | Description |
|---|---|---|
| `--n_seed` | `5` | Nombre de tracks seeds par playlist |
| `--n_recs` | `500` | Nombre de recommandations générées |
| `--k` | `100` | K pour NDCG@K et Recall@K |

**Métriques calculées** : NDCG@K, Recall@K, R-Precision

---

### 5. Lancer le serveur web

```bash
python app.py \
    --lightgcn_path models/saved/lightgcn.pt \
    --als_path models/saved/als_model.pkl \
    --track_map data/processed/track_map.pkl \
    --host 127.0.0.1 \
    --port 5000
```

L'interface est accessible sur **http://127.0.0.1:5000**.

---

## API

| Endpoint | Méthode | Description |
|---|---|---|
| `GET /api/status` | GET | État des modèles chargés |
| `GET /api/search?q=<query>` | GET | Recherche de tracks (min. 2 caractères) |
| `POST /api/recommend` | POST | Recommandations à partir de seeds |
| `GET /api/track/<id>` | GET | Métadonnées d'un track |

**Corps de `/api/recommend`** :

```json
{
  "seeds": [42, 137, 891],
  "model": "both",
  "n": 50
}
```

`model` accepte `"lightgcn"`, `"als"` ou `"both"`.

---

## Fonctionnement des recommandations

Les deux modèles utilisent la même stratégie d'inférence à partir de tracks seeds :

1. Calculer l'embedding moyen des tracks seeds → **virtual user**
2. Scorer tous les tracks par similarité cosinus avec ce virtual user
3. Retourner les `n` tracks les plus proches (seeds exclus)

---

## Dépendances principales

| Librairie | Version | Rôle |
|---|---|---|
| `torch` | 2.5.1+cu121 | LightGCN (GPU) |
| `implicit` | 0.7.2 | ALS |
| `scipy` | 1.15.3 | Matrices creuses |
| `numpy` | 2.2.6 | Calcul vectoriel |
| `Flask` | 3.1.3 | Serveur web |
| `tqdm` | 4.67.3 | Barres de progression |

---

## Licence

Le code source de ce projet est libre d'utilisation. Les données MPD sont soumises à la licence Spotify/AICrowd - voir [`license.txt`](license.txt).
