import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import yaml

# Importer les modules locaux
from data_loader import load_and_split_data
# Supposer que l'architecture est définie dans ce module par Agent 1
from model.architecture.enhanced_hybrid_model import build_enhanced_hybrid_model 
# Supposer que evaluation.py contient les métriques nécessaires (ex: RMSE)
# from evaluation import rmse # Décommenter si une métrique RMSE personnalisée est nécessaire

# Configuration (pourrait être chargée depuis config.yaml)
CONFIG_PATH = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'processed'
MODEL_SAVE_DIR = Path(__file__).parent
DEFAULT_MODEL_FILENAME = 'saved_model.h5'
DEFAULT_ASSET = 'btc' # Actif par défaut pour l'entraînement

# Charger la configuration globale (si nécessaire pour les hyperparamètres)
# try:
#     with open(CONFIG_PATH, 'r') as f:
#         config = yaml.safe_load(f)
#     training_params = config.get('training', {})
# except FileNotFoundError:
#     print(f"Warning: {CONFIG_PATH} not found. Using default training parameters.")
#     training_params = {}

# Hyperparamètres par défaut (peuvent être surchargés par config.yaml ou args)
EPOCHS = training_params.get('epochs', 50)
BATCH_SIZE = training_params.get('batch_size', 32)
VALIDATION_SPLIT = training_params.get('validation_split', 0.2) # Pourcentage des données pour la validation
LEARNING_RATE = training_params.get('learning_rate', 0.001)
LABEL_COLUMNS = ['trading_signal', 'volatility', 'market_regime'] # Doit correspondre aux sorties du modèle

def train_model(asset_name=DEFAULT_ASSET, 
                data_dir=DATA_DIR, 
                model_save_path=MODEL_SAVE_DIR / DEFAULT_MODEL_FILENAME,
                epochs=EPOCHS, 
                batch_size=BATCH_SIZE, 
                validation_split=VALIDATION_SPLIT,
                learning_rate=LEARNING_RATE,
                label_columns=LABEL_COLUMNS):
    """
    Fonction principale pour entraîner le modèle Morningstar.
    """
    print(f"--- Début de l'entraînement pour l'actif : {asset_name} ---")
    
    # 1. Charger et préparer les données
    print("1. Chargement et préparation des données...")
    file_path = data_dir / f"{asset_name}_final.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Le fichier de données {file_path} n'a pas été trouvé.")
        
    # Charger en tant que Tensors TensorFlow
    X, y_dict = load_and_split_data(file_path, label_columns=label_columns, as_tensor=True)
    print(f"  - Données chargées : X shape={X.shape}, Labels={list(y_dict.keys())}")

    # Séparer en ensembles d'entraînement et de validation (split temporel simple)
    num_samples = X.shape[0]
    num_val_samples = int(num_samples * validation_split)
    num_train_samples = num_samples - num_val_samples

    X_train, X_val = X[:num_train_samples], X[num_train_samples:]
    y_train_dict = {name: tensor[:num_train_samples] for name, tensor in y_dict.items()}
    y_val_dict = {name: tensor[num_train_samples:] for name, tensor in y_dict.items()}

    print(f"  - Séparation Train/Validation : Train={num_train_samples}, Val={num_val_samples}")

    # 2. Construire le modèle
    print("2. Construction du modèle...")
    # Déterminer la forme d'entrée à partir des données chargées
    input_shape = (X_train.shape[1],) 
    # Supposer que le modèle a besoin de connaître le nombre de classes pour les sorties de classification
    # Ces valeurs devraient idéalement venir de la config ou être déterminées à partir des données
    num_trading_classes = 5 # Strong Sell, Sell, Hold, Buy, Strong Buy
    num_regime_classes = 3 # Exemple: Bull, Bear, Sideways (à adapter)
    
    model = build_enhanced_hybrid_model(input_shape=input_shape, 
                                        num_trading_classes=num_trading_classes, 
                                        num_regime_classes=num_regime_classes)
    model.summary() # Afficher l'architecture du modèle

    # 3. Compiler le modèle
    print("3. Compilation du modèle...")
    # Définir les fonctions de perte et les métriques pour chaque tête de sortie
    # Les noms ('trading_signal_output', etc.) doivent correspondre aux noms des couches de sortie dans build_enhanced_hybrid_model
    losses = {
        'trading_signal_output': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # ou True si logits
        'volatility_output': tf.keras.losses.MeanSquaredError(),
        'market_regime_output': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) # ou True si logits
    }
    metrics = {
        'trading_signal_output': ['accuracy'],
        'volatility_output': [tf.keras.metrics.RootMeanSquaredError(name='rmse'), 'mae'], # Utiliser RMSE intégré
        'market_regime_output': ['accuracy']
    }
    # Optionnel: Pondération des pertes si certaines tâches sont plus importantes
    # loss_weights = {'trading_signal_output': 1.0, 'volatility_output': 0.5, 'market_regime_output': 0.8}

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, 
                  loss=losses, 
                  metrics=metrics)
                  # loss_weights=loss_weights) # Décommenter si besoin

    print("  - Modèle compilé avec succès.")

    # 4. Entraîner le modèle
    print(f"4. Entraînement du modèle pour {epochs} epochs...")
    
    # Callbacks
    # Sauvegarder le meilleur modèle basé sur la perte de validation totale
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        save_weights_only=False, # Sauvegarder le modèle complet
        monitor='val_loss',     # Ou une métrique spécifique comme 'val_trading_signal_output_accuracy'
        mode='min',             # 'min' pour la perte, 'max' pour l'accuracy
        save_best_only=True,
        verbose=1
    )
    # Arrêter l'entraînement si la performance ne s'améliore plus
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, # Nombre d'epochs sans amélioration avant d'arrêter
        verbose=1,
        restore_best_weights=True # Restaurer les poids du meilleur epoch
    )

    history = model.fit(
        X_train,
        y_train_dict,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val_dict),
        callbacks=[checkpoint_callback, early_stopping_callback],
        verbose=1
    )

    print("  - Entraînement terminé.")

    # 5. Sauvegarde (optionnel, car ModelCheckpoint sauvegarde déjà le meilleur)
    # Si on veut sauvegarder le modèle final indépendamment du meilleur:
    # final_model_path = str(model_save_path).replace('.h5', '_final.h5')
    # model.save(final_model_path)
    # print(f"  - Modèle final sauvegardé dans : {final_model_path}")
    print(f"  - Meilleur modèle sauvegardé dans : {model_save_path}")

    # 6. Évaluation (optionnelle ici, peut être faite dans un script séparé ou notebook)
    print("6. Évaluation sur l'ensemble de validation...")
    results = model.evaluate(X_val, y_val_dict, batch_size=batch_size, verbose=0)
    print("  - Résultats de l'évaluation (loss & metrics):")
    # Afficher les résultats de manière lisible
    metric_names = ['loss'] + [f'{output}_{metric}' for output in metrics for metric in metrics[output]] \
                   + [f'{output}_loss' for output in losses] # Keras ajoute les pertes individuelles
    
    # Créer un dictionnaire pour un affichage plus clair
    results_dict = {}
    # Les noms retournés par evaluate() peuvent être complexes, essayons de les mapper
    current_metrics_names = model.metrics_names # Noms réels utilisés par Keras
    for name, value in zip(current_metrics_names, results):
         results_dict[name] = value
         print(f"    - {name}: {value:.4f}")


    print(f"--- Entraînement pour {asset_name} terminé ---")
    return history, results_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'entraînement du modèle Morningstar")
    parser.add_argument('--asset', type=str, default=DEFAULT_ASSET, help=f"Nom de l'actif à utiliser pour l'entraînement (défaut: {DEFAULT_ASSET})")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help=f"Nombre d'epochs (défaut: {EPOCHS})")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help=f"Taille du batch (défaut: {BATCH_SIZE})")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help=f"Taux d'apprentissage (défaut: {LEARNING_RATE})")
    parser.add_argument('--val_split', type=float, default=VALIDATION_SPLIT, help=f"Proportion pour validation (défaut: {VALIDATION_SPLIT})")
    parser.add_argument('--save_path', type=str, default=str(MODEL_SAVE_DIR / DEFAULT_MODEL_FILENAME), help=f"Chemin pour sauvegarder le modèle (défaut: {MODEL_SAVE_DIR / DEFAULT_MODEL_FILENAME})")
    
    args = parser.parse_args()

    # Créer le répertoire parent si nécessaire
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_model(asset_name=args.asset,
                model_save_path=save_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                validation_split=args.val_split,
                learning_rate=args.lr,
                label_columns=LABEL_COLUMNS)
