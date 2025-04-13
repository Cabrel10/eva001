import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy.random import Generator, PCG64
import pandas as pd
from pathlib import Path

# Vérification des versions
assert tf.__version__ == '2.12.0', f"TensorFlow 2.12.0 requis, version actuelle: {tf.__version__}"
assert np.__version__ == '1.23.5', f"NumPy 1.23.5 requis, version actuelle: {np.__version__}"

class TrainingWorkflow:
    def __init__(self, config):
        tf.keras.backend.clear_session()
        self.config = config
        self.model = self._build_model()
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.config.time_window, len(self.config.features)))
        x = layers.Conv1D(64, 3, activation='relu')(inputs)
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.LSTM(64)(x)
        outputs = layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', run_eagerly=True)
        return model

    def _prepare_dataset(self, data):
        features = (data[self.config.features] - data[self.config.features].mean()) / data[self.config.features].std()
        features = features.values.astype(np.float32)
        
        # Génération des labels avec le nouveau générateur aléatoire
        rng = Generator(PCG64())
        labels = rng.uniform(-0.1, 0.1, size=len(data)).astype(np.float32)
        
        def generator():
            for i in range(len(features) - self.config.time_window):
                yield features[i:i+self.config.time_window], labels[i+self.config.time_window]
                
        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(self.config.time_window, len(self.config.features)), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32)
            )
        ).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

class TestConfig:
    def __init__(self):
        self.time_window = 30
        self.features = ["open", "high", "low", "close"]
        self.epochs = 2
        self.batch_size = 32
        self.dataset_path = "data/test_subset.parquet"

if __name__ == "__main__":
    print("=== TEST MODE ===")
    config = TestConfig()
    data = pd.read_parquet(config.dataset_path)
    
    workflow = TrainingWorkflow(config)
    history = workflow.model.fit(
        workflow._prepare_dataset(data),
        epochs=config.epochs,
        batch_size=config.batch_size
    )
    print("Training completed successfully!")
    print("Training history:", history.history)
