from Morningstar.workflows.training_workflow import TrainingWorkflow

class Cfg:
    def __init__(self):
        self.time_window = 30
        self.features = ['open', 'high', 'low', 'close']
        self.epochs = 2
        self.batch_size = 32

print("✅ Création de la configuration...")
config = Cfg()
print("✅ Initialisation du workflow...")
workflow = TrainingWorkflow(config)
print("✅ Workflow initialisé avec succès!")
