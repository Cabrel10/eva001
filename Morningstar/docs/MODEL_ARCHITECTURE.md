# Architecture du Modèle Morningstar

## Architecture du Modèle Morningstar

### Caractéristiques Techniques
- **Type** : Modèle hybride GA + CNN + LSTM
- **Input Shape** : (fenêtre_temporelle, 5) [open, high, low, close, volume]
- **Output** : 3 classes [Achat, Vente, Neutre]

### Paramètres Clés
```python
{
    "time_window": 60,  # 60 périodes
    "cnn_filters": 64,  # Nombre de filtres CNN
    "lstm_units": 64,   # Unités LSTM
    "learning_rate": 0.001,
    "dropout_rate": 0.2
}
```

### Prérequis Données
- Format : Parquet
- Colonnes obligatoires : timestamp, open, high, low, close, volume
- Fréquence : 1m, 15m, 1h, 4h, 1d
- Normalisation : Z-score (automatique)

## Configuration
- **Périodes**: 1m, 15m, 1h, 4h, 1d
- **Styles de Trading**:
  - Scalping (court terme)
  - Day Trading
  - Swing Trading (recommandé)
  - Position Trading

- **Niveaux de Risque**:
  - Prudent (1% du capital)
  - Modéré (3% du capital)
  - Agressif (5% du capital)

## Workflows
1. **Entraînement**:
   - Chargement des données historiques
   - Prétraitement et normalisation
   - Entraînement des trois modules
   - Sauvegarde du modèle

2. **Trading**:
   - Analyse en temps réel
   - Prédiction des signaux
   - Gestion des positions
   - Backtesting intégré

## Tests Unitaires
Couverture des fonctionnalités clés:
- Chargement des données
- Prédictions du modèle
- Gestion du risque
- Exécution des trades

## Dépendances
- TensorFlow 2.x
- Pandas
- NumPy
- CCXT (pour les données)

## Roadmap
1. Intégration des données on-chain
2. Optimisation GPU
3. Interface web de monitoring
