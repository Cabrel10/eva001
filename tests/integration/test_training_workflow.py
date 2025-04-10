import pytest
import os
from Morningstar.workflows.training_workflow import TrainingWorkflow

class TestTrainingWorkflow:
    @pytest.fixture
    def workflow(self):
        return TrainingWorkflow(pair="BTC/USDT", timeframe="1h")

    # Mark test as async
    async def test_workflow_execution(self, workflow, tmp_path):
        """Test l'exécution complète du workflow"""
        # Crée un dossier temporaire pour les modèles
        os.makedirs(tmp_path/"models", exist_ok=True)
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Add await to the async run method
            history = await workflow.run(epochs=1)  
            assert 'loss' in history.history
            # Check if the model file exists (adjust filename if needed based on actual saving logic)
            assert os.path.exists(f"models/BTC_USDT_1h.h5") 
        finally:
            os.chdir(original_dir)
