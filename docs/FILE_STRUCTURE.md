# Structure des Fichiers - Morningstar V2 (Approche Modulaire)

Cette documentation décrit la structure de dossiers et de fichiers adoptée pour le projet Morningstar V2, basée sur une approche modulaire intégrant la gestion des données, le modèle, le backtesting, l'exécution live et la supervision LLM.

```
/home/morningstar/Desktop/crypto_robot/Morningstar/
├── config/
│   ├── config.yaml                # Fichier de configuration (API keys, paramètres, seuils, etc.)
│   ├── secrets.env                # Variables d’environnement sensibles (NE PAS VERSIONNER)
│   └── README.md                  # Documentation sur la configuration
│
├── data/
│   ├── __init__.py                # Init package data
│   ├── raw/                       # Données brutes (CSV, JSON...)
│   ├── processed/                 # Données prétraitées prêtes pour le modèle
│   └── pipelines/                 # Scripts ETL pour traitement des données
│
├── docs/
│   ├── FILE_STRUCTURE.md          # Documentation de la structure (ce fichier)
│   ├── checklist_implementation.md# Liste de vérification pour l'implémentation
│   └── PROMPTS_GUIDE.md           # Guide des prompts pour interagir avec Cline/LLM
│
├── model/
│   ├── architecture/
│   │   ├── __init__.py            # Init package architecture
│   │   ├── enhanced_hybrid_model.py  # Modèle hybride avancé (fusion, multi-tâches)
│   │   ├── morningstar_model.py   # Wrapper/Interface pour le modèle
│   │   ├── enhanced_hybrid_model.md # Documentation/Spécification du modèle hybride
│   │   └── morningstar_model.md   # Documentation/Spécification du wrapper
│   │
│   ├── optimization/
│   │   ├── __init__.py            # Init package optimisation
│   │   ├── optimization_module.py # Modules pour optimisation (Optuna, GA...)
│   │   └── optimization_module.md # Documentation/Spécification de l'optimisation
│   │
│   └── training/
│       ├── training_script.py     # Script principal d'entraînement
│       ├── evaluation.py          # Scripts/Modules d'évaluation du modèle
│       ├── data_loader.py         # Chargement/Prétraitement données pour entraînement
│       ├── data_loader.md         # Documentation/Spécification du data loader
│       └── training_script.md     # Documentation/Spécification du script d'entraînement
│
├── workflows/
│   ├── trading_workflow.py        # Workflow principal (Data -> Features -> Signal -> Ordre -> Supervision LLM)
│   └── trading_workflow.md      # Documentation/Spécification du workflow
│
├── live/
│   ├── __init__.py                # Init package live
│   ├── executor.py                # Exécution des ordres sur les exchanges (via ccxt)
│   ├── monitoring.py              # Dashboard/Scripts de surveillance temps réel + Supervision LLM
│   ├── executor.md                # Documentation/Spécification de l'exécuteur
│   └── monitoring.md              # Documentation/Spécification du monitoring
│
├── utils/
│   ├── __init__.py                # Init package utils
│   ├── data_preparation.py        # Fonctions de préparation/nettoyage données + indicateurs
│   ├── api_manager.py             # Gestion des connexions API (exchanges, news, social, LLM)
│   ├── logging.py                 # Configuration du logging
│   ├── helpers.py                 # Fonctions utilitaires diverses
│   ├── data_preparation.md      # Documentation/Spécification de la préparation des données
│   └── api_manager.md           # Documentation/Spécification du gestionnaire d'API
│
├── tests/
│   ├── __init__.py                # Init package tests
│   ├── test_model.py              # Tests unitaires pour le modèle
│   ├── test_workflow.py           # Tests pour le workflow de trading
│   └── test_utils.py              # Tests pour les modules utilitaires
│
├── .gitignore                     # Fichier pour ignorer certains fichiers/dossiers de Git
├── requirements.txt               # Dépendances Python du projet
└── README.md                      # Présentation générale, installation, utilisation
```

## Principes Directeurs

*   **Modularité Forte**: Chaque responsabilité majeure (config, data, model, workflows, live, utils, tests) est isolée dans son propre dossier.
*   **Configuration Centralisée**: Le dossier `config/` gère tous les paramètres et secrets.
*   **Pipelines de Données Clairs**: Le dossier `data/` sépare les données brutes, traitées et les scripts de transformation.
*   **Modèle Bien Défini**: Le dossier `model/` contient l'architecture, l'optimisation et l'entraînement de manière structurée.
*   **Workflow Unique (pour l'instant)**: Le fichier `workflows/trading_workflow.py` orchestre le processus principal. Des workflows spécifiques (backtesting, optimisation) pourraient être ajoutés plus tard si nécessaire.
*   **Exécution Live Séparée**: Le dossier `live/` gère l'interaction avec les exchanges et le monitoring temps réel, y compris l'intégration LLM.
*   **Utilitaires Transverses**: Le dossier `utils/` fournit des fonctions réutilisables pour la préparation des données, la gestion des API, le logging, etc.
*   **Tests Organisés**: Le dossier `tests/` contient les tests unitaires et d'intégration pour valider les différents modules.

Cette structure vise à fournir une base solide, évolutive et maintenable pour le projet Morningstar V2.
