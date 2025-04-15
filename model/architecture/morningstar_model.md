# Spécification : Wrapper MorningstarModel

**Objectif :** Définir le rôle, l'interface et le comportement attendu du wrapper `MorningstarModel`, qui sert de point d'entrée principal pour interagir avec le modèle `EnhancedHybridModel` depuis les workflows.

---

## 1. Vue d'ensemble

`MorningstarModel` est une classe Python qui encapsule le modèle `EnhancedHybridModel` (implémenté en Keras/TensorFlow ou PyTorch). Son but est de simplifier l'utilisation du modèle en :

*   Gérant le chargement du modèle entraîné et de ses dépendances (ex: scalers, encodeurs).
*   Fournissant une interface claire et stable pour la prédiction, indépendamment des détails internes de `EnhancedHybridModel`.
*   Abstrayant potentiellement la complexité de la préparation des entrées spécifiques au framework sous-jacent.

---

## 2. Interface Principale (API Exposée)

La classe `MorningstarModel` devrait exposer au minimum les méthodes suivantes :

*   **`__init__(self, model_path: str, config: dict)`**:
    *   **Rôle**: Initialiser le wrapper.
    *   **Arguments**:
        *   `model_path`: Chemin vers le dossier contenant le modèle `EnhancedHybridModel` sauvegardé (ex: format `SavedModel` de TF ou `.pt` de PyTorch) et les fichiers associés (scalers, encodeurs...).
        *   `config`: Dictionnaire de configuration (chargé depuis `config/config.yaml`) contenant les paramètres nécessaires (ex: têtes actives, noms des features attendues).
    *   **Comportement**:
        *   Charge le modèle `EnhancedHybridModel` depuis `model_path`.
        *   Charge les objets de prétraitement nécessaires (scalers, encodeurs) sauvegardés lors de l'entraînement.
        *   Stocke la configuration pertinente.

*   **`predict(self, input_data: dict) -> dict`**:
    *   **Rôle**: Effectuer une prédiction en utilisant le modèle chargé.
    *   **Arguments**:
        *   `input_data`: Un dictionnaire contenant les données d'entrée préparées et alignées temporellement. Les clés du dictionnaire doivent correspondre à ce qu'attend le `EnhancedHybridModel` (ex: `'technical_features'`, `'contextual_features'`). Les données doivent être dans un format compatible (ex: Numpy arrays, Tensors).
    *   **Comportement**:
        *   Applique le prétraitement nécessaire (scaling, encoding) aux données d'entrée en utilisant les objets chargés lors de l'`__init__`.
        *   Formate les données pour correspondre aux shapes d'entrée attendues par `EnhancedHybridModel`.
        *   Appelle la méthode de prédiction du modèle `EnhancedHybridModel` sous-jacent.
        *   Formate potentiellement les sorties brutes du modèle (ex: logits) en un dictionnaire plus interprétable (ex: probabilités, classes prédites), tel que défini dans la spécification de `EnhancedHybridModel`.
    *   **Retour**: Un dictionnaire contenant les prédictions des différentes têtes actives (ex: `{'signal_probs': ..., 'volatility_pred': ...}`).

*   **(Optionnel) `load_model(cls, model_path: str, config: dict)`**:
    *   **Rôle**: Méthode de classe alternative pour charger et retourner une instance de `MorningstarModel`. Peut être utile pour simplifier l'instanciation.

---

## 3. Gestion des Dépendances

*   Le wrapper est responsable de charger et d'utiliser correctement les objets de prétraitement (scalers, encodeurs) qui ont été sauvegardés avec le modèle lors de l'entraînement (`model/training/training_script.py`). Ces objets sont essentiels pour assurer que les données d'inférence sont traitées de la même manière que les données d'entraînement.
*   Le chemin vers ces dépendances doit être inclus ou déductible de `model_path`.

---

## 4. Interaction avec la Configuration

*   Le wrapper utilise le dictionnaire `config` pour connaître les paramètres importants, tels que :
    *   Quelles têtes de prédiction sont actives (pour savoir quelles sorties retourner).
    *   Les noms ou indices des features attendues en entrée.
    *   Potentiellement des seuils de décision par défaut (bien que la logique de décision principale soit plutôt dans `workflows/trading_workflow.py`).

---

## 5. Exemple d'Utilisation (Pseudo-code dans le Workflow)

```python
# Dans workflows/trading_workflow.py

from model.architecture.morningstar_model import MorningstarModel
# ... autres imports ...

# Charger la configuration
config = load_config('config/config.yaml')
model_config = config['model_settings'] # Section pertinente de la config
model_directory = config['paths']['trained_model_dir']

# Initialiser le wrapper
try:
    model_wrapper = MorningstarModel(model_path=model_directory, config=model_config)
except Exception as e:
    log.error(f"Erreur lors du chargement du modèle : {e}")
    # Gérer l'erreur (ex: arrêter le workflow)

# ... dans la boucle principale du workflow ...

# Préparer les données d'entrée (via data_preparation et api_manager)
input_features = prepare_input_data(...) # Doit retourner un dict

# Obtenir les prédictions
try:
    predictions = model_wrapper.predict(input_features)
    signal_probabilities = predictions.get('signal_probs')
    volatility_prediction = predictions.get('volatility_pred')
    # ... récupérer autres prédictions ...
except Exception as e:
    log.error(f"Erreur lors de la prédiction : {e}")
    # Gérer l'erreur

# Utiliser les prédictions pour la logique de décision...
```

---

Cette spécification guide l'implémentation de `model/architecture/morningstar_model.py`, en assurant une interface claire et cohérente pour l'utilisation du modèle complexe sous-jacent.
