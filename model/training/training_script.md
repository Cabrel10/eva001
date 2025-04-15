# Spécification : Script d'Entraînement Principal

**Objectif :** Définir la structure, les étapes et les paramètres du script principal (`model/training/training_script.py`) responsable de l'entraînement du modèle `EnhancedHybridModel`.

---

## 1. Vue d'ensemble

Ce script orchestre le processus d'entraînement complet : chargement de la configuration, préparation des données via le `DataLoader`, instanciation du modèle, exécution de la boucle d'entraînement, évaluation et sauvegarde du modèle entraîné et des artefacts associés.

---

## 2. Étapes Principales du Script

1.  **Parsing des Arguments**:
    *   Accepter des arguments en ligne de commande (ex: via `argparse`) pour spécifier le chemin vers le fichier de configuration (`--config`), et potentiellement surcharger certains paramètres (ex: `--epochs`, `--batch_size`).

2.  **Chargement de la Configuration**:
    *   Charger le fichier de configuration principal (`config/config.yaml`) spécifié.
    *   Extraire les sections pertinentes : paramètres du modèle, paramètres d'entraînement, chemins des données, etc.

3.  **Initialisation du Logging**:
    *   Configurer le module de logging (`utils/logging.py`) pour enregistrer les informations importantes du processus d'entraînement.

4.  **Initialisation du DataLoader**:
    *   Instancier le `DataLoader` (`model/training/data_loader.py`) en lui passant la configuration et les chemins des données prétraitées.
    *   Obtenir les datasets d'entraînement, de validation et de test (ex: `tf.data.Dataset` ou `torch.utils.data.DataLoader`).

5.  **Instanciation du Modèle**:
    *   Instancier le modèle `EnhancedHybridModel` (`model/architecture/enhanced_hybrid_model.py`) en lui passant les paramètres d'architecture définis dans la configuration.

6.  **Compilation du Modèle (si applicable, ex: Keras)**:
    *   Définir l'optimiseur (ex: `tf.keras.optimizers.Adam`) avec le taux d'apprentissage configuré.
    *   Définir la ou les fonctions de perte. Pour le multi-tâches, définir un dictionnaire de pertes (une par tête de sortie) et potentiellement des poids de perte (`loss_weights`).
    *   Définir les métriques à suivre pendant l'entraînement et l'évaluation (spécifiques à chaque tâche).
    *   Compiler le modèle avec l'optimiseur, les pertes et les métriques.

7.  **Configuration des Callbacks (si applicable, ex: Keras)**:
    *   `ModelCheckpoint`: Pour sauvegarder le meilleur modèle (ou les poids) basé sur une métrique de validation (ex: `val_loss` ou une métrique spécifique à une tâche clé comme `val_signal_accuracy`). Configurer le chemin de sauvegarde (`models/v2/`).
    *   `EarlyStopping`: Pour arrêter l'entraînement si la métrique de validation ne s'améliore plus pendant un certain nombre d'époques (`patience`).
    *   `TensorBoard`: Pour enregistrer les logs d'entraînement et de validation visualisables avec TensorBoard (configurer le `log_dir`).
    *   (Optionnel) `ReduceLROnPlateau` ou `LearningRateScheduler`: Pour ajuster dynamiquement le taux d'apprentissage.

8.  **Exécution de l'Entraînement**:
    *   Lancer la boucle d'entraînement :
        *   Avec Keras : `model.fit(train_dataset, validation_data=validation_dataset, epochs=..., callbacks=...)`.
        *   Avec PyTorch ou boucle manuelle TensorFlow : Implémenter la boucle d'époques et de batches, calculer les gradients, mettre à jour les poids, calculer les métriques, et gérer les callbacks manuellement.
    *   Stocker l'historique de l'entraînement (pertes et métriques par époque).

9.  **Évaluation Finale**:
    *   Évaluer le modèle final (ou le meilleur modèle sauvegardé par `ModelCheckpoint`) sur l'ensemble de test (`test_dataset`).
    *   Utiliser `model.evaluate(test_dataset)` (Keras) ou une boucle d'évaluation manuelle.
    *   Enregistrer les métriques finales sur l'ensemble de test.

10. **Sauvegarde Finale**:
    *   Sauvegarder le modèle final entraîné (format `SavedModel` TF ou `.pt` PyTorch) dans `models/v2/`.
    *   Sauvegarder les objets de prétraitement utilisés par le `DataLoader` (ex: scalers de `scikit-learn`) avec le modèle, car ils sont nécessaires pour l'inférence. Utiliser `pickle` ou un format similaire.
    *   Sauvegarder potentiellement l'historique de l'entraînement et les métriques finales.

11. **Logging Final**:
    *   Enregistrer un résumé de l'entraînement : configuration utilisée, chemin du modèle sauvegardé, métriques finales sur l'ensemble de test.

---

## 4. Paramètres Configurables (via `config.yaml`)

Le script doit être piloté par des paramètres dans `config.yaml`, notamment :

*   `paths`: Chemins vers les données (`processed_data_path`), sauvegarde des modèles (`model_save_path`), logs (`log_dir`).
*   `data_loader_params`: Configuration pour le DataLoader (proportions train/val/test, etc.).
*   `model_params`: Paramètres d'architecture (`time_window`, `lstm_units`, `cnn_filters`, têtes actives, dimensions embeddings...).
*   `training_params`: Paramètres d'entraînement (`epochs`, `batch_size`, `learning_rate`, configuration de l'optimiseur, poids des pertes multi-tâches).
*   `callback_params`: Paramètres pour les callbacks (`early_stopping_patience`, `checkpoint_monitor_metric`...).

---

## 5. Considérations

*   **Reproductibilité**: Utiliser des graines aléatoires (`random.seed`, `np.random.seed`, `tf.random.set_seed`) au début du script pour assurer la reproductibilité des résultats.
*   **Gestion des Ressources**: Monitorer l'utilisation CPU/GPU/Mémoire pendant l'entraînement, surtout avec de grands modèles ou datasets.
*   **Gestion des Erreurs**: Implémenter une gestion robuste des erreurs (ex: erreurs de chargement de données, problèmes de convergence).

---

Cette spécification guide l'implémentation de `model/training/training_script.py`.
