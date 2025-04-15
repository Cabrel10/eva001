# Spécification : Module d'Optimisation

**Objectif :** Définir les stratégies, les outils et le processus d'optimisation des hyperparamètres du modèle `EnhancedHybridModel` et potentiellement d'autres aspects de la stratégie de trading.

---

## 1. Vue d'ensemble

Le module d'optimisation vise à trouver les meilleures configurations pour le modèle et la stratégie afin de maximiser la performance (selon une métrique cible) sur les données historiques. L'optimisation est typiquement un processus hors ligne, exécuté périodiquement ou lorsque des changements majeurs sont apportés au modèle ou aux données.

---

## 2. Stratégies d'Optimisation

Plusieurs aspects peuvent être optimisés :

*   **Hyperparamètres du Modèle `EnhancedHybridModel`**:
    *   Architecture : Nombre de couches, nombre de neurones/filtres par couche, types de couches (LSTM vs GRU), fonctions d'activation.
    *   Paramètres d'entraînement : Taux d'apprentissage (learning rate), taille du batch, nombre d'époques (ou critère d'arrêt précoce).
    *   Régularisation : Taux de dropout, poids de la régularisation L1/L2.
    *   Fenêtre temporelle (`time_window`).
    *   Poids des différentes fonctions de perte dans l'approche multi-tâches.
*   **Sélection de Caractéristiques (Features)**: Identifier le sous-ensemble optimal d'indicateurs techniques ou de données alternatives à fournir en entrée du modèle.
*   **Paramètres de la Stratégie de Trading**: Seuils de décision pour les signaux, paramètres de gestion de risque (si non prédits par le modèle), etc. (Moins prioritaire si le modèle gère dynamiquement SL/TP et la décision).

---

## 3. Outils et Bibliothèques

Plusieurs bibliothèques Python peuvent être utilisées pour l'optimisation :

*   **Optuna**: Framework d'optimisation d'hyperparamètres agnostique au framework ML. Facile à utiliser, supporte divers algorithmes d'échantillonnage et d'élagage (pruning). **(Recommandé comme point de départ)**.
*   **KerasTuner / Ray Tune**: Outils spécifiquement conçus pour l'optimisation d'hyperparamètres de modèles Keras/TensorFlow ou distribués.
*   **DEAP (Distributed Evolutionary Algorithms in Python)**: Bibliothèque pour implémenter des algorithmes évolutionnistes (comme les algorithmes génétiques - GA), utile pour l'optimisation de stratégies complexes ou la sélection de caractéristiques.
*   **Scikit-learn**: Contient des outils comme `GridSearchCV` et `RandomizedSearchCV`, plus simples mais potentiellement moins efficaces pour les grands espaces de recherche.

Le choix final dépendra de la complexité de l'espace de recherche et des préférences de l'équipe. Optuna est souvent un bon compromis entre simplicité et efficacité.

---

## 4. Processus d'Optimisation (Exemple avec Optuna)

1.  **Définir la Fonction Objectif (`objective`)**:
    *   Cette fonction prend un `trial` Optuna en argument.
    *   À l'intérieur du `trial`, suggérer les hyperparamètres à tester (ex: `trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)`, `trial.suggest_int('lstm_units', 32, 128)`).
    *   Instancier le `DataLoader` et le `EnhancedHybridModel` avec les hyperparamètres suggérés.
    *   Entraîner le modèle (potentiellement sur un sous-ensemble des données ou avec moins d'époques pour accélérer). L'utilisation de l'élagage (pruning) d'Optuna basé sur les métriques intermédiaires est fortement recommandée.
    *   Évaluer le modèle entraîné sur un jeu de validation dédié (non vu pendant l'entraînement). L'évaluation peut être une simple métrique de validation (ex: `val_loss`, `val_accuracy`) ou, idéalement, le résultat d'un backtest rapide sur la période de validation.
    *   Retourner la métrique de performance à optimiser (ex: Sharpe Ratio du backtest, négatif de la `val_loss`). Optuna cherchera à maximiser (ou minimiser) cette valeur.

2.  **Créer une Étude Optuna (`study`)**:
    *   `study = optuna.create_study(direction='maximize')` (ou `'minimize'`).
    *   Configurer potentiellement un `sampler` (ex: TPESampler) et un `pruner` (ex: MedianPruner).

3.  **Lancer l'Optimisation**:
    *   `study.optimize(objective, n_trials=100)` (ex: pour 100 essais).

4.  **Analyser les Résultats**:
    *   `study.best_params`: Dictionnaire des meilleurs hyperparamètres trouvés.
    *   `study.best_value`: Meilleure valeur de la métrique objectif atteinte.
    *   Optuna fournit aussi des fonctions de visualisation pour analyser l'importance des hyperparamètres et l'historique de l'optimisation.

---

## 5. Implémentation (`optimization_module.py`)

Le fichier `model/optimization/optimization_module.py` contiendra :

*   La définition de la fonction `objective` décrite ci-dessus.
*   Des fonctions utilitaires pour lancer et gérer les études Optuna (ou autre outil).
*   Potentiellement, des classes ou fonctions pour des stratégies d'optimisation plus spécifiques (ex: algorithme génétique pour la sélection de features).

---

## 6. Considérations

*   **Coût Calculatoire**: L'optimisation d'hyperparamètres peut être très coûteuse en temps de calcul, nécessitant potentiellement des ressources GPU importantes et du temps.
*   **Validation Robuste**: Il est crucial d'utiliser un jeu de données de validation distinct et représentatif pour évaluer chaque `trial` afin d'éviter le surapprentissage sur l'ensemble de test final. Une validation croisée temporelle (walk-forward optimization) est souvent recommandée pour les données financières.
*   **Métrique Objectif**: Le choix de la métrique à optimiser est critique et doit refléter les objectifs réels du système de trading (ex: Sharpe Ratio ajusté au risque, Profit Factor, Calmar Ratio...).

---

Cette spécification fournit un cadre pour l'implémentation du module d'optimisation dans `model/optimization/optimization_module.py`.
