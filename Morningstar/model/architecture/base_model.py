import tensorflow as tf
from tensorflow.keras import layers
from Morningstar.configs.tf_config import TFConfig

class BaseTradingModel:
    """Architecture de base pour les modèles de trading avec TensorFlow"""
    
    def __init__(self, input_shape=(32, 4)):
        self.config = TFConfig.get_optimized_config()
        self.model = self.build_model(input_shape)
        self.compile_model()

    def build_model(self, input_shape):
        """Construit une architecture hybride CNN-LSTM"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # Partie CNN
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.BatchNormalization()(x)
        
        # Partie LSTM 
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.LSTM(64)(x)
        
        # Couches Denses
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        # Normalisation finale et sortie
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(1)(x)  # Sortie linéaire non bornée
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def compile_model(self):
        """Configure l'entraînement pour la régression robuste"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',  # Moins sensible aux outliers
            metrics=['mae', 'mse'],
            weighted_metrics=['mae']
        )

    def train(self, dataset, epochs=10):
        """Entraîne le modèle sur un Dataset TensorFlow"""
        return self.model.fit(
            dataset,
            epochs=epochs,
            batch_size=self.config['batch_size']
        )
