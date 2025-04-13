from Morningstar.workflows.training_workflow import TrainingWorkflow
import pandas as pd

class TestConfig:
    def __init__(self):
        self.time_window = 30
        self.features = ['open', 'high', 'low', 'close']
        self.epochs = 2
        self.batch_size = 32
        self.dataset_path = 'data/test_subset.parquet'

print("=== TEST DU WORKFLOW COMPLET ===")
print("1. Chargement de la configuration...")
config = TestConfig()

print("2. Initialisation du workflow...")
workflow = TrainingWorkflow(config)

print("3. Chargement des données...")
data = pd.read_parquet(config.dataset_path)

print("4. Lancement de l'entraînement...")
history = workflow.run(data)

print("✅ Workflow testé avec succès!")
print(f"Résultats: {history.history if hasattr(history, 'history') else 'N/A'}")
