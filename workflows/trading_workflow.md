# Spécification : Workflow de Trading Principal

**Objectif :** Définir le processus complet, les étapes, les interactions entre modules et le comportement attendu du workflow de trading principal (`workflows/trading_workflow.py`).

---

## 1. Vue d'ensemble

Le `trading_workflow.py` est le script central qui orchestre l'ensemble du cycle de trading en temps réel (ou en simulation/backtesting). Il intègre les différents modules du projet pour prendre des décisions de trading informées.

Le workflow s'exécute de manière périodique (ex: toutes les minutes, heures, jours, selon la stratégie et la fréquence des données) ou en continu.

---

## 2. Étapes Principales du Workflow (par cycle d'exécution)

1.  **Initialisation (au démarrage du script)**:
    *   Charger la configuration (`config/config.yaml`).
    *   Initialiser le logging (`utils/logging.py`).
    *   Initialiser le gestionnaire d'API (`utils/api_manager.py`).
    *   Initialiser le module de préparation des données (`utils/data_preparation.py`).
    *   Charger le modèle entraîné via le wrapper `MorningstarModel` (`model/architecture/morningstar_model.py`).
    *   Initialiser l'exécuteur d'ordres (`live/executor.py`).
    *   Initialiser le module de monitoring (`live/monitoring.py`).
    *   Récupérer l'état actuel du portefeuille/positions (via `live/executor.py`).

2.  **Boucle Principale (exécutée périodiquement)**:
    *   **a. Récupération des Données Récentes**:
        *   Utiliser `utils/api_manager.py` pour récupérer les dernières données de marché (OHLCV) pour les paires configurées.
        *   (Optionnel) Récupérer les dernières données alternatives (news, sentiment social, on-chain) via `api_manager`.
    *   **b. Préparation des Caractéristiques (Features)**:
        *   Utiliser `utils/data_preparation.py` pour :
            *   Nettoyer les nouvelles données.
            *   Calculer les indicateurs techniques requis par le modèle.
            *   Intégrer/Aligner les données alternatives.
            *   Normaliser/Scaler les données en utilisant les scalers sauvegardés lors de l'entraînement (chargés par `MorningstarModel`).
            *   Formater les données dans la structure attendue par le modèle (séquences temporelles, dictionnaire d'entrées).
    *   **c. Prédiction du Modèle**:
        *   Passer les features préparées au wrapper `MorningstarModel` (`model_wrapper.predict(input_features)`).
        *   Récupérer le dictionnaire de prédictions des différentes têtes actives (signal, volatilité, régime, SL/TP).
    *   **d. Supervision par LLM (Optionnel mais recommandé)**:
        *   Préparer un prompt pour le LLM basé sur `docs/PROMPTS_GUIDE.md`, incluant :
            *   Les prédictions brutes du modèle technique.
            *   Le contexte récent du marché (peut être demandé au LLM dans une étape précédente ou résumé à partir des données alternatives).
            *   Les indicateurs techniques clés.
        *   Envoyer le prompt au LLM via `utils/api_manager.py`.
        *   Parser la réponse du LLM (ex: évaluation de cohérence, score de risque contextuel, suggestions SL/TP alternatives).
    *   **e. Logique de Décision**:
        *   Combiner les informations :
            *   Prédiction du signal de trading (`EnhancedHybridModel`).
            *   Score de confiance du signal.
            *   Prédiction de volatilité.
            *   Prédiction de régime de marché.
            *   Suggestions SL/TP (`EnhancedHybridModel` et/ou LLM).
            *   Évaluation de cohérence / Score de risque du LLM.
            *   État actuel du portefeuille et positions ouvertes.
            *   Règles de gestion de risque définies dans `config/config.yaml` (ex: taille max de position, drawdown max).
        *   Appliquer la logique de décision pour déterminer l'action finale :
            *   Ouvrir une nouvelle position (Achat/Vente).
            *   Fermer une position existante.
            *   Ne rien faire.
            *   Ajuster un SL/TP sur une position ouverte.
        *   Déterminer la taille de la position en fonction de la volatilité prédite, du score de confiance, des règles de risque et potentiellement des suggestions du LLM.
    *   **f. Exécution de l'Ordre**:
        *   Si une action est décidée, utiliser `live/executor.py` pour :
            *   Passer l'ordre (market, limit) sur l'exchange approprié.
            *   Gérer les erreurs potentielles d'exécution.
            *   Confirmer l'exécution et mettre à jour l'état interne des positions.
    *   **g. Monitoring & Logging**:
        *   Enregistrer toutes les étapes clés, les prédictions, les décisions et les actions dans les logs (`utils/logging.py`).
        *   Mettre à jour le module de monitoring (`live/monitoring.py`) avec les dernières informations (performance, état, alertes LLM...).

3.  **Gestion des Erreurs**:
    *   Implémenter une gestion robuste des erreurs à chaque étape (connexion API, préparation données, prédiction, exécution...).
    *   Définir des stratégies de retry ou d'arrêt en cas d'erreurs critiques.

4.  **Mode Backtesting/Simulation**:
    *   Le workflow doit pouvoir s'exécuter en mode simulation/backtesting.
    *   Dans ce mode :
        *   Les données sont lues depuis un historique (`data/processed/`) au lieu d'une API live.
        *   L'exécution des ordres (`live/executor.py`) est simulée (calcul des P&L, application des frais/slippage).
        *   L'interaction avec le LLM peut être simulée ou désactivée.
    *   Le mode doit être configurable (ex: via argument `--mode` ou `config.yaml`).

---

## 3. Interactions Clés entre Modules

*   **Workflow -> ApiManager**: Pour récupérer les données live et interagir avec le LLM.
*   **Workflow -> DataPreparation**: Pour transformer les données brutes en features pour le modèle.
*   **Workflow -> MorningstarModel**: Pour obtenir les prédictions multi-tâches.
*   **Workflow -> Executor**: Pour passer les ordres et gérer les positions.
*   **Workflow -> Monitoring**: Pour mettre à jour l'état et les performances.
*   **Workflow -> Logging**: Pour enregistrer l'activité.
*   **Workflow -> Config**: Pour lire les paramètres.

---

## 4. Considérations

*   **Latence**: Minimiser la latence entre la récupération des données et l'exécution de l'ordre, surtout pour les stratégies à haute fréquence.
*   **Atomicité**: Assurer la cohérence de l'état, en particulier lors de l'exécution des ordres et de la mise à jour des positions.
*   **Scalabilité**: Concevoir le workflow pour pouvoir potentiellement gérer plusieurs paires ou stratégies en parallèle.
*   **Robustesse**: Gérer les pannes d'API, les données manquantes, les erreurs de modèle, etc.

---

Cette spécification guide l'implémentation de `workflows/trading_workflow.py`.
