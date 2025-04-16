import os
import subprocess
import pandas as pd
from pathlib import Path

# Configuration
tokens = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "MATICUSDT"] # Utilisation de symboles standards pour Binance/Kucoin (à vérifier si besoin)
exchange = "kucoin" # Changé pour Kucoin
timeframe = "15m"
start = "2024-01-01" # Date historique valide
end = "2024-04-01"   # Date historique valide
expected_columns = 38 # Nombre de colonnes attendu en sortie

# Dossiers
raw_dir = Path("data/raw")
processed_dir = Path("data/processed")
logs_dir = Path("logs") # Bien que non utilisé directement ici, c'est bien de le définir

raw_dir.mkdir(exist_ok=True)
processed_dir.mkdir(exist_ok=True)
logs_dir.mkdir(exist_ok=True)

success_count = 0
total_tokens = len(tokens)

print(f"--- Démarrage du Test Complet du Pipeline pour {total_tokens} Tokens ---")
print(f"Configuration: Exchange={exchange}, Timeframe={timeframe}, Période={start} à {end}")
print(f"Colonnes attendues en sortie: {expected_columns}")

for token in tokens:
    # Ajuster le nom du fichier pour utiliser le symbole de base (sans suffixes comme _SPBL)
    symbol_base = token.replace("USDT", "").replace("_SPBL", "").lower()
    raw_path = raw_dir / f"{symbol_base}_raw.csv"
    output_path = processed_dir / f"{symbol_base}_final.parquet"

    print(f"\n=== Traitement de {token} ===") # Afficher le token original pour info

    # --- Étape 1 : Téléchargement des données brutes ---
    print(f"[1/3] Téléchargement des données brutes vers {raw_path}...")
    cmd_download = [
        "python", "utils/api_manager.py",
        "--token", token,
        "--exchange", exchange,
        "--start", start,
        "--end", end,
        "--timeframe", timeframe,
        "--output", str(raw_path) # Convertir Path en string pour subprocess
    ]
    result_download = subprocess.run(cmd_download, capture_output=True, text=True, check=False) # check=False pour gérer l'erreur manuellement

    # Vérification du téléchargement
    if result_download.returncode != 0 or not raw_path.exists() or raw_path.stat().st_size < 1000: # Seuil de taille bas pour être sûr qu'il y a des données
        print(f"❌ Échec téléchargement {token}.")
        if result_download.stderr:
            print(f"   Erreur: {result_download.stderr.strip()}")
        elif not raw_path.exists():
            print(f"   Erreur: Fichier {raw_path} non créé.")
        elif raw_path.stat().st_size < 1000:
             print(f"   Erreur: Fichier {raw_path} semble trop petit ({raw_path.stat().st_size} bytes).")
        continue
    else:
        print(f"✅ Données brutes téléchargées ({raw_path.stat().st_size} bytes).")


    # --- Étape 2 : Transformation via pipeline ---
    print(f"[2/3] Lancement du pipeline de transformation vers {output_path}...")
    cmd_pipeline = [
        "python", "data/pipelines/data_pipeline.py",
        "--input", str(raw_path),    # Argument corrigé
        "--output", str(output_path) # Argument corrigé
    ]
    result_pipeline = subprocess.run(cmd_pipeline, capture_output=True, text=True, check=False)

    # Vérification de la sortie du pipeline
    if result_pipeline.returncode != 0 or not output_path.exists():
        print(f"❌ Échec pipeline {token}.")
        if result_pipeline.stderr:
            print(f"   Erreur: {result_pipeline.stderr.strip()}")
        elif not output_path.exists():
             print(f"   Erreur: Fichier {output_path} non créé.")
        continue
    else:
        print(f"✅ Pipeline terminé.")


    # --- Étape 3 : Validation du fichier Parquet ---
    print(f"[3/3] Validation du fichier {output_path}...")
    try:
        df = pd.read_parquet(output_path)
        shape = df.shape
        columns_count = len(df.columns)
        print(f"   📊 Shape: {shape}")
        print(f"   📎 Colonnes trouvées: {columns_count}")

        if columns_count == expected_columns:
            print(f"   🎯 Structure validée ({expected_columns} colonnes) ✅")
            success_count += 1
        else:
            print(f"   ⚠️ Mauvais nombre de colonnes (attendu: {expected_columns}, trouvé: {columns_count}).")
            # print(f"      Colonnes: {df.columns.tolist()}") # Décommenter pour voir les colonnes

    except Exception as e:
        print(f"❌ Erreur lors de la lecture ou validation du fichier {output_path}: {e}")
        continue

# --- Résumé Final ---
print(f"\n--- Test Complet du Pipeline Terminé ---")
if success_count == total_tokens:
    print(f"✅🔥 Tous les {total_tokens}/{total_tokens} jeux de données ont été traités et validés avec succès !")
else:
    print(f"⚠️ {success_count}/{total_tokens} jeux de données validés avec succès.")

print("-----------------------------------------")
