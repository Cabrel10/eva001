import tensorflow as tf
# Correction de l'import selon les standards Keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 
from Morningstar.model.architecture.morningstar_model import MorningstarTradingModel
from Morningstar.configs.morningstar_config import MorningstarConfig, TradingStyle
from Morningstar.utils.data_manager import ExchangeDataManager
import pandas as pd
import numpy as np
import logging

class MorningstarTrainingWorkflow:
    """Workflow complet pour l'entraînement du modèle Morningstar"""
    
    def __init__(self, config: MorningstarConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def prepare_data(self) -> tuple:
        """Charge et prépare les données pour l'entraînement"""
        # Chargement des données
        data = pd.read_parquet(self.config.DATA_PATH)
        
        # Vérification et sélection des colonnes disponibles
        available_cols = data.columns.tolist()
        features = []
        # Fonction pour ajouter et vérifier les colonnes
        def add_features(column_list):
            added = [col for col in column_list if col in available_cols]
            if len(added) != len(column_list):
                 missing = set(column_list) - set(available_cols)
                 self.logger.warning(f"Colonnes manquantes dans les données: {missing}")
            return added

        features += add_features(self.config.base_columns)
        features += add_features(self.config.technical_columns)
        features += add_features(self.config.social_columns)
        features += add_features(self.config.correlation_columns) # Ajout des colonnes de corrélation
        
        # Assurer l'unicité et l'ordre (optionnel mais propre)
        features = sorted(list(set(features))) 
        
        self.logger.info(f"Colonnes sélectionnées ({len(features)}): {features}")
        if not self.config.base_columns: # Au moins les colonnes de base doivent être présentes
            raise ValueError("Aucune colonne valide trouvée - vérifiez la configuration")
            
        # Gestion des NaN
        data = data[features]
        
        # Gestion des NaN pour toutes les features sauf base
        # Remplissage par 0 pour techniques, sociales, corrélations (à ajuster si besoin)
        fill_cols = self.config.technical_columns + self.config.social_columns + self.config.correlation_columns
        for col in features:
            if col in fill_cols:
                 nan_pct = data[col].isna().mean()
                 if nan_pct > 0:
                     self.logger.info(f"Remplissage des NaN pour {col} ({nan_pct:.2%} NaN) avec 0.")
                     data[col] = data[col].fillna(0) # Remplissage simple par 0
        
        # Suppression des lignes avec NaN restantes (devrait concerner surtout base_columns)
        initial_rows = len(data)
        data = data.dropna()
        rows_dropped = initial_rows - len(data)
        if rows_dropped > 0:
            self.logger.warning(f"{rows_dropped} lignes supprimées à cause de NaN restants.")
        
        self.logger.info(f"Données finales: {len(data)} lignes, {len(features)} colonnes")
        
        # Normalisation par groupe (StandardScaler)
        def scale_group(df, cols):
            valid_cols = [col for col in cols if col in df.columns]
            if valid_cols:
                 mean = df[valid_cols].mean()
                 std = df[valid_cols].std()
                 # Éviter la division par zéro pour les colonnes constantes
                 std[std == 0] = 1 
                 df[valid_cols] = (df[valid_cols] - mean) / std
                 self.logger.info(f"Normalisation appliquée aux colonnes: {valid_cols}")
            return df

        data = scale_group(data, self.config.base_columns)
        data = scale_group(data, self.config.technical_columns)
        data = scale_group(data, self.config.social_columns)
        data = scale_group(data, self.config.correlation_columns) # Ajout normalisation corrélations
        
        # Conversion en séquences avec gestion mémoire
        final_features = [col for col in features if col in data.columns]
        self.logger.info(f"Création des séquences avec {len(final_features)} features: {final_features}")
        
        # Paramètres de batch
        chunk_size = 100000  # Nombre de séquences par batch
        total_sequences = len(data) - self.config.time_window
        n_chunks = int(np.ceil(total_sequences / chunk_size))
        
        # Pré-allocation mémoire (float32 pour économiser de l'espace)
        sequences = np.zeros((total_sequences, self.config.time_window, len(final_features)), dtype=np.float32)
        
        # Remplissage par chunks
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, total_sequences)
            
            # Création des séquences pour ce chunk
            chunk_data = data.iloc[start:start + self.config.time_window + chunk_size]
            chunk_values = chunk_data[final_features].values.astype(np.float32)
            
            for i in range(end - start):
                sequences[start + i] = chunk_values[i:i+self.config.time_window]
            
            self.logger.info(f"Traitement du chunk {chunk_idx+1}/{n_chunks} ({end-start} séquences)")
        
        # Génération des labels par batch pour économiser la mémoire
        labels = np.zeros((len(sequences), 3), dtype=np.float32)
        label_chunk_size = 50000
        
        for i in range(0, len(sequences), label_chunk_size):
            chunk_end = min(i + label_chunk_size, len(sequences))
            labels[i:chunk_end] = self._generate_labels(
                sequences[i:chunk_end],
                data=data.iloc[i:i + chunk_end + self.config.time_window]
            )
        
        return self._split_data(sequences, labels) # Passer aussi les labels au split
    
    def _split_data(self, sequences: np.ndarray, labels: np.ndarray, test_size=0.2):
        """Division des séquences et labels en train/test"""
        split_idx = int(len(sequences) * (1 - test_size))
        train_seq, test_seq = sequences[:split_idx], sequences[split_idx:]
        train_labels, test_labels = labels[:split_idx], labels[split_idx:]
        self.logger.info(f"Split data: Train ({train_seq.shape}, {train_labels.shape}), Test ({test_seq.shape}, {test_labels.shape})")
        return (train_seq, train_labels), (test_seq, test_labels)
    
    def train_model(self, train_data: tuple, val_data: tuple): # Accepte tuples (seq, labels)
        """Entraînement du modèle avec les données préparées"""
        train_seq, train_labels = train_data
        # Remplacer les variables non utilisées par _
        _, _ = val_data # Utiliser les données de validation splittées via val_data directement dans fit
        
        # S'assurer que input_shape correspond au nombre final de features
        input_shape = train_seq.shape[1:] 
        self.logger.info(f"Initialisation du modèle avec input_shape: {input_shape}")
        
        model_instance = MorningstarTradingModel(
            input_shape=input_shape, 
            num_classes=train_labels.shape[1] # Dynamique basé sur les labels
        )
        
        model_instance.compile_model(self.config.learning_rate)
        
        self.logger.info("Début de l'entraînement...")
        history = model_instance.model.fit(
            x=train_seq,
            y=train_labels,
            validation_data=val_data, # Utiliser le set de validation explicite
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            callbacks=[
                ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'), # Monitorer val_loss
                EarlyStopping(patience=15, monitor='val_loss', restore_best_weights=True) # Patience augmentée
            ]
        )
        self.logger.info("Entraînement terminé.")
        
        return model_instance, history # Retourner l'instance du modèle
    
    def _generate_labels(self, sequences: np.ndarray, data: pd.DataFrame = None, look_forward_steps: int = 5, pt_sl_ratio: float = 1.5) -> np.ndarray:
        """
        Génération de labels basée sur la méthode Triple Barrier.
        Prédit si le prix atteindra une barrière de profit (pt) avant une barrière de perte (sl),
        ou s'il touchera la barrière temporelle (look_forward_steps).

        Args:
            sequences: Numpy array des séquences de features (shape: [n_samples, time_window, n_features]).
            look_forward_steps: Nombre de pas de temps à regarder dans le futur pour les barrières.
            pt_sl_ratio: Ratio Profit Target / Stop Loss.

        Returns:
            Numpy array de labels one-hot encoded [Buy, Sell, Hold/Timeout].
            - Buy (1, 0, 0): Barrière de profit supérieure atteinte avant la perte.
            - Sell (0, 1, 0): Barrière de perte inférieure atteinte avant le profit.
            - Hold (0, 0, 1): Aucune barrière touchée avant la fin de look_forward_steps.
        """
        self.logger.info(f"Génération des labels (Triple Barrier): look_forward={look_forward_steps}, pt_sl_ratio={pt_sl_ratio}")
        
        n_samples = sequences.shape[0]
        n_timesteps = sequences.shape[1]
        close_col_index = self.config.base_columns.index('close') # Index de la colonne 'close'
        
        # Récupérer les prix de clôture à la fin de chaque séquence (t)
        entry_prices = sequences[:, -1, close_col_index]
        
        # Initialiser les labels à Hold/Timeout (0, 0, 1)
        labels = np.zeros((n_samples, 3))
        labels[:, 2] = 1 

        # Calculer les changements de prix futurs pour chaque séquence
        # Note: Nécessite d'avoir accès aux données *après* la séquence.
        # Ici, on simule en regardant les pas suivants DANS les données fournies.
        # ATTENTION: Cela crée une fuite de données si les séquences se chevauchent trop.
        # Une implémentation correcte nécessiterait le DataFrame original.
        # Pour la démo, on utilise les données disponibles, mais ce n'est pas idéal.
        
        # Initialiser le tableau des prix futurs
        future_prices = np.full((n_samples, look_forward_steps), np.nan)
        
        if data is None:
            self.logger.error("Impossible d'accéder aux données originales pour les prix futurs")
            return labels

        close_col_name = self.config.base_columns[close_col_index] # Nom de la colonne
        for i in range(n_samples):
            # Obtenir l'index dans le DataFrame original correspondant à la fin de la séquence
            original_idx = len(data) - len(sequences) + i + n_timesteps - 1
            
            # Vérifier qu'on ne dépasse pas la taille du DataFrame
            if original_idx + look_forward_steps >= len(data):
                continue  # On ne peut pas regarder dans le futur plus loin que les données existantes
                
            # Récupérer les prix futurs
            future_prices[i] = data.loc[
                original_idx + 1 : original_idx + look_forward_steps, 
                close_col_name
            ].values

        # Calcul des barrières dynamiques (basé sur la volatilité serait mieux)
        # Exemple simple: pourcentage fixe
        volatility = np.std(sequences[:, :, close_col_index], axis=1) * 0.01 # Approximation très simple
        volatility[volatility == 0] = 0.001 # Eviter division par zero

        upper_barrier = entry_prices + entry_prices * volatility * pt_sl_ratio
        lower_barrier = entry_prices - entry_prices * volatility

        # Parcourir les pas futurs simulés
        for t in range(look_forward_steps):
            current_future_prices = future_prices[:, t]
            
            # Indices où le prix futur est disponible
            valid_indices = ~np.isnan(current_future_prices)
            
            # Indices où le label n'est pas encore défini (encore Hold/Timeout)
            hold_indices = labels[:, 2] == 1
            
            active_indices = valid_indices & hold_indices

            # Vérifier si la barrière supérieure est touchée
            hit_upper = current_future_prices[active_indices] >= upper_barrier[active_indices]
            labels[active_indices][hit_upper] = [1, 0, 0] # Label Buy

            # Mettre à jour les indices actifs (ceux qui n'ont pas touché la barrière sup)
            active_indices_after_upper = active_indices & (labels[:, 2] == 1)

            # Vérifier si la barrière inférieure est touchée
            hit_lower = current_future_prices[active_indices_after_upper] <= lower_barrier[active_indices_after_upper]
            labels[active_indices_after_upper][hit_lower] = [0, 1, 0] # Label Sell

        n_buy = np.sum(labels[:, 0])
        n_sell = np.sum(labels[:, 1])
        n_hold = np.sum(labels[:, 2])
        self.logger.info(f"Labels générés: Buy={n_buy}, Sell={n_sell}, Hold={n_hold}")
        
        # Vérifier si des labels ont pu être générés (si future_prices est resté NaN)
        if n_buy == 0 and n_sell == 0 and n_hold == n_samples:
             self.logger.warning("Aucun label Buy/Sell généré. Vérifiez la logique d'accès aux prix futurs.")

        return labels
