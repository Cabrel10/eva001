# Documentation du Projet Morningstar

## 1. Architecture Globale (Mise à jour)
```mermaid
graph TD
    A[Data Sources (API Binance, Twitter, Reddit, GitHub)] --> B(Data Manager / Social Scraper)
    B --> C[Data Preprocessing / Feature Engineering]
    C --> D(Base Model CNN-LSTM)
    D --> E[Decision Engine / Trading Workflow]
    E --> F[Execution System (via CCXT)]
    F --> G[Performance Monitoring]
    G --> H[Model Tuning]
    H --> C
```

## 2. Cahier des Charges Technique

### 2.1 Spécifications Techniques des Données

#### Types de Données Supportés :
- **Données Marché (Core)**:
  ```python
  {
    "timestamp": "int64",        # Horodatage UNIX en ms
    "open": "float32",           # Prix d'ouverture normalisé [0-1]
    "high": "float32",           # Prix haut normalisé
    "low": "float32",            # Prix bas normalisé  
    "close": "float32",          # Prix clôture normalisé
    "volume": "float32",         # Volume normalisé
    "liquidité": "float32",      # Profondeur de marché (order book)
    "funding_rate": "float32",   # Taux de financement
    "open_interest": "float32"   # Open Interest (futures)
  }
  ```

- **Données On-Chain**:
  ```python
  {
    "hashrate": "float32",       # Puissance de minage
    "difficulty": "float32",     # Difficulté du réseau
    "tx_count": "int32",         # Nombre de transactions
    "fees": "float32",           # Frais moyens
    "whale_flows": "float32"     # Mouvements des gros portefeuilles
  }
  ```

- **Données Sentiment**:
  ```python
  {
    "twitter_sentiment": "float32",  # Score [-1,1]
    "reddit_comments": "int32",      # Volume de discussions
    "bitcointalk": "float32",        # Sentiment analysé
    "news_sentiment": "float32",     # Analyse NLP médias
    "fear_greed": "float32"          # Indice peur/avidité
  }
  ```

- **Métriques Avancées**:
  ```python
  {
    "volatility_index": "float32",   # Indice de volatilité
    "correlation_btc": "float32",    # Corrélation avec BTC
    "macro_indicators": "float32",   # Données macroéconomiques
    "liq_levels": "float32[]",       # Niveaux de liquidation
    "oi_change": "float32"           # Changement d'open interest
  }
  ```

#### Conditions d'Entrée :
1. **Prérequis Utilisateur**:
   - Fournir un historique minimal de 10 000 candles
   - Définir la paire d'actifs et le timeframe
   - Spécifier le mode (spot/futures)

2. **Contraintes Techniques**:
   - Taille max des séquences : 512 candles
   - Latence maximale tolérée : 500ms
   - Mise à jour horaire des données

#### Contraintes Marché :
- **Gestion des Données Hétérogènes** :
  - Synchronisation multi-sources (API, scrapers, nodes)
  - Normalisation cross-platform (0-1)
  - Gestion des latences variables (100ms-5s)

- **Analyse Temps Réel** :
  - NLP des réseaux sociaux <500ms
  - Mise à jour on-chain toutes les 10min
  - Actualisation métriques trading 1Hz
  - Détection anomalies en <1s

- **Sécurité et Compliance** :
  - Chiffrement AES-256 des flux
  - Anonymisation données sensibles
  - Audit trail RGPD complet
  - Rotation des clés API

- **Robustesse Marché** :
  - Gestion krachs (>30% drop)
  - Détection manipulation marché
  - Adaptation aux forks/upgrades

#### Orientations Possibles :
1. **Mode Spot**:
   - Optimisation du timing d'entrée/sortie
   - Gestion des frais de trading

2. **Mode Futures**:
   - Prise en compte du leverage
   - Gestion du risque de liquidation
   - Analyse du funding rate

3. **Multi-Actifs**:
   - Corrélations inter-marchés
   - Allocation dynamique
```

### 2.2 Modularité du Modèle
- **Structure modulaire**:
  ```
  model/
  ├── architecture/
  │   └── base_model.py # Architecture CNN-LSTM de base
  ├── training/         # (Structure à vérifier/implémenter)
  │   # ├── train.py    
  │   # └── data_loader.py
  └── tuning/           # (Structure à vérifier/implémenter)
      # ├── online.py   
      # └── validate.py
  ```

### 2.3 Gestion des Versions et Contraintes

#### Versioning :
- Système DVC + Git avec tags sémantiques
- Signature cryptographique des inputs/outputs
- Historique des performances par version

#### Contraintes Opérationnelles :
1. **Hardware** :
   - GPU minimum : NVIDIA RTX 3080 (12GB VRAM)
   - RAM requise : 32GB minimum
   - Stockage : 1TB SSD NVMe

2. **Temps Réel** :
   - Fréquence de prédiction : 10Hz max
   - Latence cible : <200ms
   - Tolérance aux gaps de données

3. **Sécurité** :
   - Chiffrement AES-256 des modèles
   - Audit trail des décisions
   - Isolation des données sensibles

#### Orientations Futures :
1. **Optimisations** :
   - Quantification INT8 pour l'inférence
   - Pruning des réseaux neuronaux
   - Parallelisation multi-GPU

2. **Évolutivité** :
   - Support multi-langues (Python, Rust)
   - API GraphQL pour l'intégration
   - Containers Docker/Kubernetes

## 3. Spécifications Techniques

### 3.1 Préprocessing
- Normalisation adaptative
- Gestion des NaN
- Augmentation de données

### 3.2 Modèle CNN-LSTM (Implémentation actuelle)
- **CNN**:
  - Couches: 1 Conv1D (64 filtres, kernel 3) + MaxPooling1D + BatchNormalization
- **LSTM**:
  - Couches: 2 LSTM (128 puis 64 unités)
- **Dense**:
  - Couches: 1 Dense (64 unités, ReLU) + Dropout(0.3) + 1 Dense (1 unité, sortie)
- **Input Shape (défaut)**: (32, 4) - Séquence de 32 pas avec 4 features.
- **Configuration**: Chargée depuis `Morningstar.configs.tf_config.TFConfig`.

## 4. Proposition d'Améliorations

1. **Système de Monitoring**:
   - Tracking des features en production
   - Détection de drift

2. **Pipeline CI/CD**:
   - Tests automatiques
   - Validation croisée en continu

3. **Sauvegarde Incrémentale**:
   - Versioning des weights
   - Rollback possible

## 5. Synthèse et Feuille de Route

### Points Forts Actuels :
✅ **Couverture Data Complète** :
   - Données OHLCV (Binance), Sentiment (Twitter, Reddit), Activité (GitHub) via `prepare_full_dataset.py`.
   - `DataManager` amélioré (gestion dates, retries, timeout, progression).
   - Script `prepare_full_dataset.py` corrigé (async/await, format paires).

✅ **Architecture Modulaire** :
   - Modèle CNN-LSTM de base défini dans `model/architecture/base_model.py`.
   - Workflows distincts pour entraînement et trading.
   - Versioning robuste (DVC+Git) - *À confirmer si DVC est utilisé.*
   - Support multi-résolutions (1m-1d) via `DataManager`.

✅ **Gestion des Risques** :
   - Détection anomalies marché
   - Système anti-liquidation
   - Monitoring continu

### Roadmap Prioritaire :

1. **Phase 1 (M1-M3)** :
   - Implémentation du pipeline de données
   - Développement du core modèle
   - Intégration backtesting

2. **Phase 2 (M4-M6)** :
   - Optimisation temps réel
   - Module de fine-tuning
   - Dashboard de monitoring

3. **Phase 3 (M7+)** :
   - Analyse multi-actifs
   - AutoML intégré
   - Gestion de portefeuille

### Recommandations Stratégiques :
🔧 **Intégration Progressive** :
   - Commencer par BTC/ETH spot
   - Puis étendre aux alts/futures

📊 **Validation Continue** :
   - Backtests hebdomadaires
   - Benchmark vs stratégies existantes
   - Audit mensuel des performances

🛡️ **Sécurité Renforcée** :
   - Pentest des APIs
   - Isolation des données sensibles
   - Plan de reprise d'activité
