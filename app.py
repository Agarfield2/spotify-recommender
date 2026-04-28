import argparse
import json
import os
import pickle
import sys
import time
import traceback

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")


# État global - modèles chargés au démarrage

STATE = {
    "lightgcn": None,   # LightGCNTrainer instance
    "als":      None,   # ALSRecommender instance
    "id2track": {},     # int -> dict {name, artist, ...}
    "track2id": {},     # "artist - name" -> int   (pour recherche)
    "errors":   {},     # clé -> message d'erreur si chargement raté
}


# Chargement des modèles

def load_models(lightgcn_path, als_path, track_map_path):
    # --- LightGCN ---
    if lightgcn_path and os.path.exists(lightgcn_path):
        try:
            _here = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, _here)
            from lightgcn_model import LightGCNTrainer
            print(f"[LightGCN] Chargement depuis {lightgcn_path}...")
            STATE["lightgcn"] = LightGCNTrainer.load(lightgcn_path)
            print("[LightGCN] OK Chargé")
        except Exception as e:
            STATE["errors"]["lightgcn"] = str(e)
            print(f"[LightGCN] Erreur Erreur : {e}")
    else:
        STATE["errors"]["lightgcn"] = f"Fichier introuvable : {lightgcn_path}"
        print(f"[LightGCN] Erreur Pas de fichier : {lightgcn_path}")

    # --- ALS ---
    if als_path and os.path.exists(als_path):
        try:
            _here = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, _here)
            from als_model import ALSRecommender
            print(f"[ALS] Chargement depuis {als_path}...")
            STATE["als"] = ALSRecommender.load(als_path)
            print("[ALS] OK Chargé")
        except Exception as e:
            STATE["errors"]["als"] = str(e)
            print(f"[ALS] Erreur Erreur : {e}")
    else:
        STATE["errors"]["als"] = f"Fichier introuvable : {als_path}"
        print(f"[ALS] Erreur Pas de fichier : {als_path}")

    # --- Track map ---
    # Cherche d'abord track_map.pkl, sinon utilise idx2track.pkl + track2idx.pkl
    processed_dir = os.path.dirname(track_map_path) if track_map_path else "data/processed"

    idx2track_path = os.path.join(processed_dir, "idx2track.pkl")
    track2idx_path = os.path.join(processed_dir, "track2idx.pkl")

    # Cherche aussi playlists.pkl pour enrichir avec les métadonnées réelles
    playlists_path = os.path.join(processed_dir, "playlists.pkl")

    if track_map_path and os.path.exists(track_map_path):
        candidate = track_map_path
    elif os.path.exists(idx2track_path):
        candidate = idx2track_path
    else:
        candidate = None

    if candidate:
        try:
            print(f"[TrackMap] Chargement depuis {candidate}...")
            with open(candidate, "rb") as f:
                raw = pickle.load(f)

            # idx2track : dict int -> str (uri) ou dict
            if isinstance(raw, dict):
                for k, v in raw.items():
                    idx = int(k)
                    if isinstance(v, dict):
                        name   = v.get("track_name", v.get("name", f"Track {idx}"))
                        artist = v.get("artist_name", "")
                        uri    = v.get("track_uri", "")
                    else:
                        # URI brute spotify:track:xxxx -> on extrait l'id comme nom de fallback
                        uri    = str(v)
                        name   = uri.split(":")[-1] if ":" in uri else uri
                        artist = ""
                    STATE["id2track"][idx] = {"name": name, "artist_name": artist, "track_uri": uri}
                    label = f"{artist} {name}".strip().lower()
                    STATE["track2id"][label] = idx

            print(f"[TrackMap] OK {len(STATE['id2track']):,} tracks chargés (URIs)")

            # Enrichissement optionnel avec playlists.pkl pour récupérer track_name / artist_name
            if os.path.exists(playlists_path):
                print(f"[TrackMap] Enrichissement depuis playlists.pkl...")
                # Charge track2idx pour retrouver les index
                if os.path.exists(track2idx_path):
                    with open(track2idx_path, "rb") as f:
                        track2idx_raw = pickle.load(f)
                else:
                    track2idx_raw = {}

                with open(playlists_path, "rb") as f:
                    playlists_raw = pickle.load(f)

                STATE["track2id"] = {}  # reset pour reconstruire avec les vrais noms
                seen = set()
                for playlist in playlists_raw:
                    for track in playlist.get("tracks", []):
                        uri = track.get("track_uri", "")
                        if uri in seen:
                            continue
                        seen.add(uri)
                        idx = track2idx_raw.get(uri)
                        if idx is None:
                            continue
                        name   = track.get("track_name", "")
                        artist = track.get("artist_name", "")
                        STATE["id2track"][idx] = {
                            "name": name, "artist_name": artist, "track_uri": uri
                        }
                        label = f"{artist} {name}".strip().lower()
                        STATE["track2id"][label] = idx

                print(f"[TrackMap] OK Enrichi : {len(STATE['id2track']):,} tracks avec noms réels")

        except Exception as e:
            STATE["errors"]["track_map"] = str(e)
            print(f"[TrackMap] Erreur Erreur : {e}")
    else:
        print(f"[TrackMap] Pas de fichier trouvé - indices numériques utilisés")


# Helpers

def track_info(idx: int) -> dict:
    if idx in STATE["id2track"]:
        info = STATE["id2track"][idx]
        return {
            "id":     idx,
            "name":   info.get("track_name", info.get("name", f"Track {idx}")),
            "artist": info.get("artist_name", ""),
            "uri":    info.get("track_uri", ""),
        }
    return {"id": idx, "name": f"Track #{idx}", "artist": "", "uri": ""}


def search_tracks(query: str, limit: int = 10) -> list:
    q = query.lower().strip()
    if not q:
        return []
    results = []
    seen = set()
    for label, idx in STATE["track2id"].items():
        if q in label and idx not in seen:
            seen.add(idx)
            results.append(track_info(idx))
            if len(results) >= limit:
                break
    return results


# Routes API

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "lightgcn": STATE["lightgcn"] is not None,
        "als":      STATE["als"] is not None,
        "n_tracks": len(STATE["id2track"]),
        "errors":   STATE["errors"],
    })


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify([])
    results = search_tracks(q, limit=12)
    return jsonify(results)


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    body  = request.get_json(force=True)
    model = body.get("model", "both")
    seeds = [int(x) for x in body.get("seeds", [])]
    n     = int(body.get("n", 50))

    if not seeds:
        return jsonify({"error": "Aucun seed fourni"}), 400

    results = {}

    # LightGCN
    if model in ("lightgcn", "both"):
        if STATE["lightgcn"] is None:
            results["lightgcn"] = {"error": STATE["errors"].get("lightgcn", "Non chargé")}
        else:
            try:
                t0   = time.time()
                recs = STATE["lightgcn"].recommend_from_tracks(seeds, n=n)
                ms   = int((time.time() - t0) * 1000)
                results["lightgcn"] = {
                    "tracks": [track_info(i) for i in recs[:n]],
                    "ms":     ms,
                }
            except Exception as e:
                results["lightgcn"] = {"error": traceback.format_exc()}

    # ALS
    if model in ("als", "both"):
        if STATE["als"] is None:
            results["als"] = {"error": STATE["errors"].get("als", "Non chargé")}
        else:
            try:
                t0   = time.time()
                recs = STATE["als"].recommend_from_tracks(seeds, n=n)
                ms   = int((time.time() - t0) * 1000)
                results["als"] = {
                    "tracks": [track_info(i) for i in recs[:n]],
                    "ms":     ms,
                }
            except Exception as e:
                results["als"] = {"error": traceback.format_exc()}

    return jsonify(results)


@app.route("/api/track/<int:track_id>")
def api_track(track_id):
    return jsonify(track_info(track_id))


# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lightgcn_path", default="models/saved/lightgcn.pt")
    parser.add_argument("--als_path",      default="models/saved/als_model.pkl")
    parser.add_argument("--track_map",     default="data/processed/track_map.pkl")
    parser.add_argument("--host",          default="127.0.0.1")
    parser.add_argument("--port",          type=int, default=5000)
    parser.add_argument("--debug",         action="store_true")
    args = parser.parse_args()

    # Crée le dossier static si besoin
    os.makedirs("static", exist_ok=True)

    load_models(args.lightgcn_path, args.als_path, args.track_map)

    print(f"\n Serveur démarré sur http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
