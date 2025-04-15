# Morningstar V2 - Robot de Trading Crypto Hybride Avancé (Spécifications)

## Introduction du Projet

**Contexte et Objectifs :**
Morningstar V2 est un système de trading algorithmique avancé conçu pour opérer sur les marchés des cryptomonnaies. Son objectif principal est de générer des signaux de trading optimisés en combinant :
1.  **Analyse Technique Quantitative**: Utilisation de modèles de deep learning (CNN, LSTM) pour identifier les patterns dans les données de marché (OHLCV, indicateurs).
2.  **Analyse Contextuelle Qualitative**: Intégration de données alternatives (sentiment des réseaux sociaux, news, données on-chain) et supervision par un Large Language Model (LLM) pour évaluer la pertinence des signaux techniques dans le contexte actuel du marché.
3.  **Optimisation Continue**: Utilisation d'algorithmes d'optimisation (génétiques, Optuna) pour affiner les hyperparamètres du modèle et potentiellement les stratégies de trading.
4.  **Gestion de Risque Adaptative**: Intégration de têtes de prédiction dédiées à la volatilité, aux régimes de marché et à la suggestion dynamique de Stop-Loss/Take-Profit.

**Valeur Ajoutée :**
Le système vise à dépasser les limites des approches purement techniques en ajoutant une couche d'intelligence contextuelle via le LLM, permettant une meilleure adaptation aux conditions changeantes du marché et une gestion des risques plus proactive. L'architecture modulaire facilite l'évolution et la maintenance.

---

## Fonctionnalités Clés

*   **Préparation et Enrichissement des Données**: Pipelines ETL robustes pour collecter, nettoyer, aligner et enrichir les données de marché avec des indicateurs techniques et des données contextuelles (sentiment, news...).
*   **Modèle `EnhancedHybridModel`**: Architecture hybride multi-modale et multi-tâches combinant CNN, LSTM et potentiellement des embeddings textuels/contextuels.
    *   **Têtes de Prédiction Multiples**:
        *   Signal de Trading (Classification 5 classes + score de confiance).
        *   Prédiction de Volatilité (Classification/Régression).
        *   Détection de Régime de Marché (Classification).
        *   Suggestion de SL/TP Dynamiques (Régression/RL).
*   **Optimisation Intégrée**: Modules pour l'optimisation des hyperparamètres et potentiellement des stratégies.
*   **Workflow de Trading Orchestré**: Processus centralisé gérant le flux complet : récupération des données, préparation des features, prédiction du modèle, supervision par LLM, décision et exécution d'ordre.
*   **Backtesting et Simulation Avancés**: Capacités de backtesting multi-stratégies (spot, futures, options) avec différents profils de risque, incluant la simulation des coûts de transaction et du slippage.
*   **Exécution Live Multi-Exchange**: Connexion et exécution d'ordres en temps réel sur les principaux exchanges (Bitget, KuCoin, Binance via `ccxt`).
*   **Monitoring et Supervision LLM en Temps Réel**: Dashboard de suivi des performances, des positions et des alertes. Intégration du LLM pour l'analyse contextuelle continue, la validation des signaux et les recommandations de gestion.

---

## Instructions d’Installation

1.  **Prérequis Système :**
    *   Python 3.9+
    *   Git
    *   (Optionnel mais recommandé) Gestionnaire d'environnement virtuel (venv, conda)
    *   (Si TA-Lib est utilisé) Installation des dépendances système de TA-Lib (voir documentation TA-Lib).

2.  **Cloner le Dépôt :**
    ```bash
    git clone <url_du_depot>
    cd Morningstar
    ```

3.  **Créer et Activer l'Environnement Virtuel (avec Conda) :**
    ```bash
    # Assurez-vous d'avoir Conda (Miniconda ou Anaconda) installé
    conda create --name trading_env python=3.9 -y  # Ou la version de Python souhaitée
    conda activate trading_env
    ```

4.  **Installer les Dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note : Assurez-vous que `requirements.txt` est à jour avec toutes les dépendances listées dans les spécifications).*

5.  **Configuration des Clés API et Secrets :**
    *   Créez une copie du fichier `config/secrets.env.example` (qui doit être créé comme modèle) et nommez-la `config/secrets.env`.
    *   Remplissez `config/secrets.env` avec vos clés API personnelles pour les exchanges (Bitget, KuCoin, Binance), les sources de données (NewsAPI, Twitter si utilisé) et le LLM (OpenAI).
    *   **IMPORTANT :** Ce fichier `secrets.env` **ne doit jamais être versionné** (il est inclus dans `.gitignore`).

6.  **Configuration des Paramètres Généraux :**
    *   Ouvrez le fichier `config/config.yaml`.
    *   Ajustez les paramètres selon vos besoins :
        *   Exchanges à utiliser.
        *   Paires de trading cibles.
        *   Paramètres de stratégie (profil de risque, taille de position par défaut...).
        *   Seuils de décision.
        *   Paramètres du modèle (fenêtre temporelle, etc.).
        *   Configuration du logging.
        *   Paramètres spécifiques au LLM (modèle à utiliser...).

7.  **Initialisation (si nécessaire) :**
    *   Certains modules pourraient nécessiter une initialisation (ex: création de base de données, téléchargement initial de données historiques). Suivez les instructions spécifiques si fournies.

---

## Utilisation et Déploiement

*   **Préparation des Données :**
    *   **Pipeline Complet :** Pour exécuter l'ensemble du pipeline de préparation des données (chargement, nettoyage, feature engineering, labeling) sur un fichier brut et générer le dataset final :
        ```bash
        python data/pipelines/data_pipeline.py --input data/raw/VOTRE_FICHIER_BRUT.csv --output data/processed/final_dataset.parquet
        ```
        *(Remplacez `VOTRE_FICHIER_BRUT.csv` par le nom de votre fichier de données brutes, par exemple `btc_usdt_1h.csv`)*
    *   **(Optionnel) Étapes Individuelles :** Si des scripts pour des étapes spécifiques existent (ex: téléchargement initial) :
        *   Exemple : `python data/pipelines/fetch_ohlcv.py --exchange binance --pair BTC/USDT --start_date 2020-01-01`

*   **Entraînement du Modèle :**
    ```bash
    python model/training/training_script.py --config config/config.yaml
    ```
    *   Le script utilisera la configuration spécifiée et les données préparées dans `data/processed/`. Le modèle entraîné sera sauvegardé dans `models/`.

*   **Backtesting / Simulation :**
    *   Le backtesting peut être intégré au workflow principal ou via un script dédié (à définir).
    *   Exemple (si intégré au workflow en mode backtest) :
        ```bash
        python workflows/trading_workflow.py --config config/config.yaml --mode backtest --start_date YYYY-MM-DD --end_date YYYY-MM-DD
        ```

*   **Exécution du Workflow de Trading (Live) :**
    ```bash
    python workflows/trading_workflow.py --config config/config.yaml --mode live
    ```
    *   Le workflow s'exécutera en continu (ou selon la fréquence définie), récupérant les données, générant des signaux, interagissant avec le LLM et potentiellement passant des ordres via `live/executor.py`.

*   **Monitoring :**
    *   Lancez le dashboard de monitoring (si implémenté, ex: avec Streamlit) :
        ```bash
        streamlit run live/monitoring.py
        ```
    *   Consultez les fichiers de logs dans le dossier `logs/`.

*   **Exécution des Tests :**
    ```bash
    pytest tests/
    ```

*   **Analyse Exploratoire des Labels :**
    *   Pour valider la pertinence et la distribution des labels générés par le pipeline, vous pouvez exécuter le notebook d'analyse :
        ```bash
        # Assurez-vous d'avoir installé les dépendances pour l'environnement explore_env
        # (pandas, matplotlib, seaborn, scipy, jupyter)
        jupyter notebook notebooks/explore_labels.ipynb
        ```
    *   Ce notebook charge `data/processed/final_dataset.parquet` et fournit des visualisations et statistiques sur les différents labels (`signal_trading`, `market_regime`, `level_sl`, `level_tp`, etc.).

---

## Documentation Détaillée

*   **Structure du Projet**: `docs/FILE_STRUCTURE.md`
*   **Checklist d'Implémentation**: `docs/checklist_implementation.md`
*   **Guide des Prompts LLM**: `docs/PROMPTS_GUIDE.md`
*   **Spécifications des Modules**: Fichiers `.md` dédiés dans chaque sous-dossier (`model/architecture/enhanced_hybrid_model.md`, `utils/api_manager.md`, etc.).

---

## Contributions et Roadmap

*   **Contributions**: Les contributions sont encouragées. Veuillez suivre les conventions de codage, écrire des tests et mettre à jour la documentation pertinente.
*   **Roadmap**:
    *   Implémentation complète de toutes les têtes de prédiction.
    *   Intégration de plus de sources de données alternatives.
    *   Amélioration des stratégies d'optimisation.
    *   Développement d'un dashboard de monitoring plus interactif.
    *   Exploration de techniques de Reinforcement Learning pour la décision et le SL/TP.
    *   Support de nouveaux exchanges.

---

## Licence

(À définir - ex: MIT, Apache 2.0) - Créer un fichier `LICENSE`.
