from dataclasses import dataclass

@dataclass
class HFConfig:
    repo_id: str = "your_username/morningstar-model"
    commit_message: str = "Update model v1.0"
    model_card: str = """
---
language: en
tags:
- trading
- neural-evolution
- cryptocurrency
---

# Morningstar Trading Model

Hybrid AI model combining:
- CNN-LSTM architecture 
- Genetic algorithm optimization
- Crypto market analysis
"""
