import pandas as pd
from sklearn.metrics import mean_squared_error
from data_loader import load_and_split_data
import tensorflow as tf

def evaluate_model(model, X_test, y_test, is_classification=False):
    """
    Évalue le modèle sur un jeu de test.

    Args:
        model (tf.keras.Model): Modèle à évaluer.
        X_test (pd.DataFrame): Features de test.
        y_test (pd.Series): Labels de test.
        is_classification (bool): Indique si le problème est de classification.

    Returns:
        dict: Dictionnaire contenant les métriques d'évaluation.
    """
    if is_classification:
        # Pour la classification, nous devrions utiliser des métriques spécifiques
        # comme l'accuracy, le classification report, et la matrice de confusion.
        # Cependant, dans cet exemple, nous supposons une régression.
        raise NotImplementedError("Classification n'est pas encore implémentée pour ce modèle.")
    else:
        # Évaluation pour la régression
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        metrics = {
            'rmse': rmse
        }

    return metrics

def load_test_data(file_path, label_column='label'):
    """
    Charge les données de test à partir d'un fichier Parquet.

    Args:
        file_path (str): Chemin vers le fichier Parquet.
        label_column (str): Nom de la colonne contenant les labels.

    Returns:
        tuple: Tuple contenant les features (X) et les labels (y).
    """
    return load_and_split_data(file_path, label_column)

def generate_evaluation_report(model, test_file_path, is_classification=False):
    """
    Génère un rapport d'évaluation pour le modèle.

    Args:
        model (tf.keras.Model): Modèle à évaluer.
        test_file_path (str): Chemin vers le fichier Parquet de test.
        is_classification (bool): Indique si le problème est de classification.

    Returns:
        str: Rapport d'évaluation.
    """
    X_test, y_test = load_test_data(test_file_path, label_column='label')

    metrics = evaluate_model(model, X_test, y_test, is_classification)

    if is_classification:
        report = (
            f"Accuracy: {metrics['accuracy']:.4f}\n"
            f"Classification Report:\n{metrics['classification_report']}\n"
            f"Confusion Matrix:\n{metrics['confusion_matrix']}"
        )
    else:
        report = f"RMSE: {metrics['rmse']:.4f}"

    return report
