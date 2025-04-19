import argparse
from pathlib import Path
from model.training.data_loader import load_and_split_data
from model.architecture.morningstar_model import MorningstarModel

def train_model(data_path: str, model_save_path: str):
    """Exécute le workflow complet d'entraînement du modèle."""
    # Charger les données
    (x_train, x_llm_train), y_train = load_and_split_data(
        data_path,
        label_columns=['signal', 'volatility_quantiles', 'volatility_regime', 'market_regime', 'sl_tp'],
        as_tensor=True
    )

    # Initialiser le modèle
    model = MorningstarModel()
    model.initialize_model()
    
    # Configuration de compilation
    losses = {
        'signal': 'sparse_categorical_crossentropy',
        'volatility_quantiles': 'mse', 
        'volatility_regime': 'sparse_categorical_crossentropy',
        'market_regime': 'sparse_categorical_crossentropy',
        'sl_tp': 'mse'
    }
    
    metrics = {
        'signal': ['accuracy'],
        'volatility_quantiles': ['mae'],
        'volatility_regime': ['accuracy'],
        'market_regime': ['accuracy'],
        'sl_tp': ['mae']
    }
    
    model.model.compile(
        optimizer='adam',
        loss=losses,
        metrics=metrics
    )

    # Entraînement
    model.model.fit(
        {'technical_input': x_train, 'llm_input': x_llm_train},
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2
    )

    # Sauvegarde
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    model.model.save(model_save_path)
    print(f"Modèle sauvegardé à {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Chemin vers les données d'entraînement")
    parser.add_argument("--output", type=str, required=True, help="Chemin de sauvegarde du modèle")
    args = parser.parse_args()
    
    train_model(args.data, args.output)
