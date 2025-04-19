import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
from utils.llm_integration import LLMIntegration
from tensorflow.keras.models import Model
from typing import Tuple

def build_enhanced_hybrid_model(input_shape: Tuple[int] = (38,),
                              llm_embedding_dim: int = 768,
                              num_trading_classes: int = 5,
                              num_regime_classes: int = 3) -> Model:
    """Construit le modèle avec 2 entrées et 5 sorties."""
    # Initialisation du service LLM
    llm_service = LLMIntegration()
    
    # Entrées
    tech_input = Input(shape=input_shape, name="technical_input")
    llm_input = Input(shape=(llm_embedding_dim,), name="llm_input")
    
    # Description des indicateurs techniques pour générer les embeddings
    tech_description = "Indicateurs techniques: " + ", ".join([
        "RSI", "MACD", "Bollinger Bands", 
        "Volume", "Moyennes mobiles"
    ])

    # Traitement des features techniques
    x = Dense(256, activation='relu')(tech_input)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Traitement des embeddings LLM (avec fallback sur zéros si échec)
    llm_embeddings = llm_service.get_embeddings(tech_description)
    if llm_embeddings is None:
        llm_embeddings = [0.0] * llm_embedding_dim
        
    # Utilisation des embeddings comme entrée
    y = Dense(128, activation='relu')(llm_input)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)

    # Fusion
    fused = Concatenate()([x, y])
    z = Dense(128, activation='relu')(fused)

    # Têtes de sortie
    signal_output = Dense(num_trading_classes, activation='softmax', name='signal')(z)
    volatility_quantiles = Dense(3, activation='linear', name='volatility_quantiles')(z)
    volatility_regime = Dense(3, activation='softmax', name='volatility_regime')(z)
    market_regime = Dense(num_regime_classes, activation='softmax', name='market_regime')(z)
    sl_tp = Dense(2, activation='linear', name='sl_tp')(z)

    # Modèle final
    model = Model(
        inputs=[tech_input, llm_input],
        outputs=[signal_output, volatility_quantiles, volatility_regime, market_regime, sl_tp],
        name='enhanced_hybrid_model'
    )

    # Compilation
    model.compile(
        optimizer='adam',
        loss={
            'signal': 'sparse_categorical_crossentropy',
            'volatility_quantiles': 'mse',
            'volatility_regime': 'sparse_categorical_crossentropy',
            'market_regime': 'sparse_categorical_crossentropy',
            'sl_tp': 'mse'
        },
        metrics=['accuracy']
    )

    return model
