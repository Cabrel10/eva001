import tensorflow as tf
import tf2onnx
from model.architecture.morningstar_model import MorningstarTradingModel
from configs.morningstar_config import MorningstarConfig

def convert_to_onnx():
    # Chargement du modèle entraîné
    model = tf.keras.models.load_model('best_model.h5')
    
    # Conversion en ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(model)
    
    # Sauvegarde
    with open("morningstar_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("Modèle converti avec succès en ONNX")

if __name__ == "__main__":
    convert_to_onnx()
