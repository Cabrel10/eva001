# Guide de Déploiement Complet

## Étape 1: Préparation des Données
1. Le notebook Colab crée automatiquement le dataset
2. Il gère 20+ colonnes (OHLCV + indicateurs techniques)
3. Le modèle est conçu pour cette dimension

## Étape 2: Entraînement
1. Exécuter toutes les cellules du notebook
2. Vérifier que la loss finale est inférieure à 0.01
3. Télécharger `morningstar_final.h5`

## Étape 3: Déploiement Live
```python
# trading_bot.py
import ccxt
import tensorflow as tf
import numpy as np

# 1. Chargement du modèle
model = tf.keras.models.load_model('morningstar_final.h5')

# 2. Configuration
exchange = ccxt.binance({
    'apiKey': 'VOTRE_CLE',
    'secret': 'VOTRE_SECRET'
})

# 3. Fonction de prédiction
def predict_signal(data):
    data = (data - data.mean()) / data.std()  # Normalisation
    prediction = model.predict(np.array([data]))
    return float(prediction[0][0])

# 4. Boucle de trading
while True:
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d', limit=50)
    data = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    processed_data = prepare_dataset(data)  # Utilisez votre fonction existante
    signal = predict_signal(processed_data)
    
    if signal > 0.5:
        exchange.create_market_buy_order('BTC/USDT', 0.01)
    elif signal < -0.5:
        exchange.create_market_sell_order('BTC/USDT', 0.01)
```

## Monitoring
- Vérifier les logs toutes les 24h
- Recalibrer le modèle toutes les 2 semaines
- Surveiller le drawdown (<20%)

## Sécurité
1. Ne pas stocker les clés API dans le code
2. Utiliser un compte dédié
3. Limiter les fonds exposés
