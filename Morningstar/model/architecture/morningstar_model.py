import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple

class MorningstarTradingModel:
    """Modèle composite GA+CNN+LSTM pour le trading crypto"""
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_composite_model()
        
    def _build_ga_module(self) -> models.Model:
        """Module Genetic Algorithm pour l'optimisation des hyperparamètres"""
        inputs = layers.Input(shape=self.input_shape)
        # Architecture à optimiser génétiquement
        x = layers.Flatten()(inputs)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Reshape((self.input_shape[0], 64))(x)
        return models.Model(inputs, x, name='ga_module')
    
    def _build_cnn_module(self) -> models.Model:
        """Module CNN pour la détection de patterns"""
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        return models.Model(inputs, x, name='cnn_module')
    
    def _build_lstm_module(self) -> models.Model:
        """Module LSTM pour l'analyse temporelle"""
        inputs = layers.Input(shape=self.input_shape)
        x = layers.LSTM(64, return_sequences=True)(inputs)
        return models.Model(inputs, x, name='lstm_module')
    
    def _build_composite_model(self) -> models.Model:
        """Assemblage des modules"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Branches parallèles
        ga_output = self._build_ga_module()(inputs)
        cnn_output = self._build_cnn_module()(inputs)
        lstm_output = self._build_lstm_module()(inputs)
        
        # Ajustement des dimensions pour la concaténation
        cnn_pooled = layers.GlobalAveragePooling1D()(cnn_output)
        cnn_repeated = layers.RepeatVector(self.input_shape[0])(cnn_pooled)
        
        # Fusion des features
        merged = layers.Concatenate()([ga_output, cnn_repeated, lstm_output])
        
        # Couche de décision
        outputs = layers.Dense(self.num_classes, activation='softmax')(merged)
        
        return models.Model(inputs, outputs, name='morningstar_model')

    def compile_model(self, learning_rate=0.001):
        """Configuration de l'entraînement"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
