# Spécification : Modèle EnhancedHybridModel

**Objectif :** Définir l'architecture, les entrées, les sorties et le comportement attendu du modèle de deep learning principal `EnhancedHybridModel`.

---

## 1. Vue d'ensemble

`EnhancedHybridModel` est un modèle Keras/TensorFlow (ou PyTorch) conçu pour le trading de cryptomonnaies. Il adopte une architecture hybride et multi-tâches pour intégrer diverses sources d'information et prédire plusieurs aspects pertinents du marché.

*   **Type**: Hybride (CNN, LSTM, potentiellement Attention/Transformers) + Multi-Modal + Multi-Tâches.
*   **Framework**: TensorFlow (ou PyTorch, à préciser lors de l'implémentation).

---

## 2. Architecture Modulaire

Le modèle est composé de plusieurs modules interconnectés :

### 2.1. Modules d'Extraction de Caractéristiques

*   **Module d'Extraction Temporelle (CNN/LSTM)**:
    *   **Entrée**: Séquences de données OHLCV et indicateurs techniques normalisées (Shape: `(batch_size, time_window, num_technical_features)`).
    *   **Architecture**: Combinaison de couches Convolutionnelles 1D (pour capturer les motifs locaux/spatiaux dans la fenêtre temporelle) suivies de couches LSTM ou GRU (pour modéliser les dépendances temporelles longues). Des couches de Dropout et BatchNormalization peuvent être ajoutées pour la régularisation.
    *   **Sortie**: Vecteur de caractéristiques temporelles (Shape: `(batch_size, temporal_embedding_dim)`).

*   **Module d'Extraction Contextuelle (NLP/Embedding)**:
    *   **Entrée**: Données textuelles prétraitées (news, tweets) ou scores de sentiment dérivés. Peut aussi accepter d'autres données alternatives structurées (on-chain, économiques). (Shape variable selon le type d'entrée, ex: `(batch_size, sequence_length)` pour texte, `(batch_size, num_alternative_features)` pour structuré).
    *   **Architecture**:
        *   Pour le texte : Utilisation d'embeddings pré-entraînés (ex: Word2Vec, GloVe, BERT via `TFHub` ou `Transformers`) suivis potentiellement de couches RNN ou Attention pour obtenir une représentation contextuelle.
        *   Pour les données structurées : Couches Dense.
    *   **Sortie**: Vecteur de caractéristiques contextuelles/alternatives (Shape: `(batch_size, contextual_embedding_dim)`).

### 2.2. Module de Fusion Multi-Modale

*   **Entrée**: Les vecteurs de caractéristiques issus des modules d'extraction temporelle et contextuelle.
*   **Architecture**: Mécanisme pour combiner les informations des différentes modalités. Options :
    *   **Concaténation simple**: Concaténer les vecteurs de sortie.
    *   **Attention pondérée**: Utiliser un mécanisme d'attention (ex: Additive Attention, Multi-Head Attention) pour pondérer l'importance relative de chaque modalité avant ou après concaténation.
    *   **Couches Dense**: Appliquer des couches Dense sur les caractéristiques concaténées/pondérées.
*   **Sortie**: Vecteur de caractéristiques fusionnées représentant l'état combiné du marché (Shape: `(batch_size, fused_embedding_dim)`).

### 2.3. Têtes de Prédiction Multi-Tâches

Chaque tête est une petite sous-architecture (généralement quelques couches Dense) prenant en entrée le vecteur de caractéristiques fusionnées et produisant une prédiction spécifique.

*   **Tête 1: Signal de Trading**
    *   **Objectif**: Prédire le signal de trading directionnel.
    *   **Type de Sortie**: Classification (multi-classe).
    *   **Architecture**: Couches Dense avec une couche finale ayant `N` neurones (ex: N=5 pour Achat Fort, Achat, Neutre, Vente, Vente Forte) et une activation `softmax`.
    *   **Sortie**: Probabilités pour chaque classe de signal (Shape: `(batch_size, num_signal_classes)`). Un score de confiance peut être dérivé de ces probabilités (ex: probabilité max, différence entre probabilités...).
    *   **Fonction de Perte**: Categorical Cross-Entropy.

*   **Tête 2: Prédiction de Volatilité**
    *   **Objectif**: Anticiper le niveau de volatilité futur proche.
    *   **Type de Sortie**: Régression (prédire une valeur de volatilité, ex: ATR normalisé) OU Classification (prédire des niveaux : Basse, Moyenne, Haute).
    *   **Architecture**: Couches Dense. Activation `linear` pour la régression, `softmax` pour la classification.
    *   **Sortie**: Valeur de volatilité prédite (Shape: `(batch_size, 1)`) ou probabilités de classe (Shape: `(batch_size, num_volatility_classes)`).
    *   **Fonction de Perte**: Mean Squared Error (MSE) pour la régression, Categorical Cross-Entropy pour la classification.

*   **Tête 3: Détection de Régime de Marché**
    *   **Objectif**: Identifier le régime dominant actuel (ex: Tendance Haussière, Tendance Baissière, Range/Consolidation).
    *   **Type de Sortie**: Classification (multi-classe).
    *   **Architecture**: Couches Dense avec une couche finale ayant `M` neurones (nombre de régimes) et une activation `softmax`.
    *   **Sortie**: Probabilités pour chaque régime (Shape: `(batch_size, num_regime_classes)`).
    *   **Fonction de Perte**: Categorical Cross-Entropy.

*   **Tête 4: Suggestion de SL/TP Dynamiques**
    *   **Objectif**: Suggérer des niveaux de Stop-Loss et Take-Profit adaptés aux conditions actuelles.
    *   **Type de Sortie**: Régression (prédire les niveaux de prix SL et TP, ou des distances/multiples d'ATR).
    *   **Architecture**: Couches Dense avec activation `linear`. Peut nécessiter une normalisation spécifique des cibles. L'approche par Reinforcement Learning (RL) est une alternative complexe mais potentiellement plus performante (à considérer pour une évolution future).
    *   **Sortie**: Niveaux SL et TP prédits (Shape: `(batch_size, 2)`).
    *   **Fonction de Perte**: MSE ou une perte personnalisée (ex: MAE).

---

## 3. Entrées Attendues

*   **Données Techniques**: DataFrame/Numpy array contenant OHLCV et indicateurs techniques, normalisés (ex: Z-score), sous forme de séquences temporelles.
*   **Données Contextuelles/Alternatives**: Données textuelles tokenisées/embeddées, scores de sentiment, données on-chain structurées, etc., normalisées de manière appropriée.
*   **Alignement Temporel**: Toutes les entrées doivent être correctement alignées dans le temps.

Le `model/training/data_loader.py` est responsable de la préparation de ces entrées dans le format attendu par le modèle.

---

## 4. Sorties Attendues

Le modèle retourne un dictionnaire (ou une structure similaire) contenant les sorties de chaque tête de prédiction active.

```python
# Exemple de sortie attendue (pseudo-code)
output = {
    "signal_probs": tensor_signal_probabilities,      # Shape: (batch_size, num_signal_classes)
    "volatility_pred": tensor_volatility_prediction, # Shape: (batch_size, 1 ou num_volatility_classes)
    "regime_probs": tensor_regime_probabilities,    # Shape: (batch_size, num_regime_classes)
    "sl_tp_pred": tensor_sl_tp_levels              # Shape: (batch_size, 2)
}
```

---

## 5. Entraînement

*   **Fonction de Perte Combinée**: L'entraînement utilisera une fonction de perte combinée, somme pondérée des pertes de chaque tête active. Les poids peuvent être ajustés (hyperparamètre) pour équilibrer l'apprentissage des différentes tâches.
    `total_loss = w1 * loss_signal + w2 * loss_volatility + w3 * loss_regime + w4 * loss_sl_tp`
*   **Optimiseur**: Adam ou AdamW sont de bons choix par défaut.
*   **Métriques**: Suivi des métriques spécifiques à chaque tâche (accuracy, MSE, MAE) en plus de la perte totale.

---

## 6. Flexibilité

*   L'architecture doit permettre d'activer/désactiver facilement les différentes têtes de prédiction via la configuration (`config/config.yaml`).
*   Le module d'extraction contextuelle doit pouvoir être désactivé si seules les données techniques sont utilisées.

---

Cette spécification sert de guide pour l'implémentation de `model/architecture/enhanced_hybrid_model.py`. Les détails précis des couches, du nombre de neurones, des fonctions d'activation (autres que la couche finale) et des hyperparamètres seront déterminés lors de l'implémentation et de l'optimisation.
