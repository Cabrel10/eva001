#!/bin/bash
# Script de déploiement complet

# Configuration
MODEL_NAME="morningstar_tf"
PAIR="BTC/USDT"
TIMEFRAME="1h"
EPOCHS=100

# 1. Activer l'environnement
echo "Activation de l'environnement conda..."
conda activate trading_env

# 2. Exécuter les tests
echo "Lancement des tests..."
pytest tests/ -v || { echo "Les tests ont échoué"; exit 1; }

# 3. Entraînement du modèle
echo "Lancement de l'entraînement..."
python -m Morningstar.workflows.training_workflow \
    --pair $PAIR \
    --timeframe $TIMEFRAME \
    --epochs $EPOCHS

# 4. Packager le modèle
echo "Packaging du modèle..."
mkdir -p deploy
tar -czvf deploy/${MODEL_NAME}.tar.gz \
    models/ \
    configs/ \
    requirements.txt

echo "Déploiement terminé avec succès"
