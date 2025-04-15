# Spécification : DataLoader pour l'Entraînement

**Objectif :** Définir la structure, les fonctionnalités et le comportement attendu du module `DataLoader` (`model/training/data_loader.py`), responsable de la préparation et du chargement des données pour l'entraînement et l'évaluation du modèle `EnhancedHybridModel`.

---

## 1. Vue d'ensemble

Le `DataLoader` est une classe ou un ensemble de fonctions qui prend les données prétraitées (générées par `utils/data_preparation.py` et stockées dans `data/processed/`) et les transforme dans le format requis par le framework de deep learning (TensorFlow/PyTorch) pour l'entraînement et l'évaluation.

Ses responsabilités principales incluent :

*   Charger les données prétraitées.
*   Créer des séquences temporelles (fenêtres glissantes).
*   Séparer les données en ensembles d'entraînement, de validation et de test.
*   Générer des lots (batches) de données.
*   Gérer les différentes modalités d'entrée (techniques, contextuelles).
*   Préparer les cibles (labels) pour les différentes têtes de prédiction multi-tâches.

---

## 2. Fonctionnalités Clés

*   **Chargement des Données**:
    *   Doit pouvoir charger les données depuis des fichiers Parquet ou CSV stockés dans `data/processed/`.
    *   Doit sélectionner les colonnes pertinentes (features et cibles) basées sur la configuration (`config/config.yaml`).

*   **Création de Séquences**:
    *   Transformer les séries temporelles plates en séquences de longueur fixe (`time_window` défini dans la config).
    *   Pour chaque séquence d'entrée (features), associer la ou les cibles correspondantes (ex: le signal, la volatilité, le régime, les niveaux SL/TP pour la période suivant immédiatement la fenêtre).

*   **Séparation Train/Validation/Test**:
    *   Implémenter une stratégie de séparation temporelle pour éviter la fuite de données futures dans le passé.
    *   Options :
        *   Séparation simple par dates (ex: 70% train, 15% validation, 15% test).
        *   Validation croisée temporelle (Walk-Forward Validation) pour une évaluation plus robuste (peut être complexe à implémenter dans le DataLoader lui-même, parfois géré au niveau du script d'entraînement).
    *   Les proportions ou dates de séparation doivent être configurables.

*   **Gestion Multi-Modale**:
    *   Si des données contextuelles/alternatives sont utilisées, le DataLoader doit les charger et les aligner temporellement avec les données techniques.
    *   Il doit générer des lots contenant toutes les modalités d'entrée requises par `EnhancedHybridModel`.

*   **Préparation des Cibles Multi-Tâches**:
    *   Générer les labels pour chaque tête de prédiction active.
    *   Cela peut impliquer :
        *   Calculer le signal futur (ex: basé sur le changement de prix sur une période future).
        *   Calculer la volatilité future.
        *   Identifier le régime de marché futur.
        *   Calculer les niveaux SL/TP optimaux a posteriori (pour l'entraînement supervisé de la tête SL/TP).
        *   Appliquer l'encodage nécessaire (ex: one-hot encoding pour les cibles de classification).

*   **Batching**:
    *   Regrouper les séquences préparées en lots (batches) de taille configurable (`batch_size` dans la config).
    *   Optionnel : Mélanger (shuffle) les données de l'ensemble d'entraînement à chaque époque.

*   **Intégration Framework DL**:
    *   Fournir une interface compatible avec Keras/TensorFlow (ex: `tf.data.Dataset`) ou PyTorch (ex: `torch.utils.data.Dataset` et `DataLoader`). Cela permet une intégration efficace avec les boucles d'entraînement (`model.fit()` ou boucle manuelle).

---

## 3. Interface Attendue (Exemple Classe)

```python
# Exemple de structure de classe (pseudo-code)

class TrainingDataLoader:
    def __init__(self, config: dict, data_path: str, features: list, targets: list):
        """
        Initialise le DataLoader.
        Args:
            config: Dictionnaire de configuration.
            data_path: Chemin vers les données prétraitées.
            features: Liste des noms de colonnes de features.
            targets: Liste des noms de colonnes cibles.
        """
        self.config = config
        self.data_path = data_path
        self.features = features
        self.targets = targets
        self.time_window = config['model_params']['time_window']
        # ... charger les données ...
        # ... préparer les cibles multi-tâches ...
        # ... séparer train/val/test ...

    def _create_sequences(self, data_features, data_targets):
        """Crée les séquences temporelles."""
        # ... logique de fenêtrage ...
        return sequences, labels

    def get_datasets(self, batch_size: int) -> tuple:
        """
        Retourne les datasets train, validation, test prêts à l'emploi
        (ex: tf.data.Dataset ou PyTorch DataLoader).
        Args:
            batch_size: Taille des lots.
        Returns:
            Tuple contenant (train_dataset, validation_dataset, test_dataset).
        """
        # ... créer séquences pour train, val, test ...
        # ... créer les datasets/dataloaders avec batching et shuffling ...
        return train_ds, val_ds, test_ds

    # ... autres méthodes utilitaires si nécessaire ...
```

---

## 4. Considérations

*   **Performance**: Le chargement et la préparation des données peuvent être un goulot d'étranglement. Optimiser ces étapes est important (ex: utilisation de Parquet, vectorisation avec Numpy/Pandas, parallélisation avec `tf.data` ou `num_workers` de PyTorch).
*   **Gestion de la Mémoire**: Pour les grands datasets, éviter de charger toutes les données en mémoire simultanément. Utiliser des générateurs ou les capacités de chargement différé des frameworks DL.
*   **Reproductibilité**: S'assurer que la séparation train/val/test et le shuffling (si utilisé) sont reproductibles (ex: en fixant les graines aléatoires).

---

Cette spécification guide l'implémentation de `model/training/data_loader.py`.
