# Gestion des Erreurs - Morningstar

## Erreurs Courantes et Solutions

### 1. Problèmes de Données
**Erreur**: `MissingDataError`  
**Cause**: Données historiques incomplètes  
**Solution**:
- Vérifier la connexion API
- Utiliser un fallback local
- Implémenter des données mock pour les tests

**Erreur**: `DataShapeError`  
**Cause**: Format des données incorrect  
**Solution**:
- Valider le shape avant traitement
- Logger les données problématiques

### 2. Problèmes de Modèle
**Erreur**: `ModelPredictionError`  
**Cause**: Shape des inputs incompatible  
**Solution**:
- Vérifier le preprocessing
- Ajouter des assertions dans le code

**Erreur**: `TrainingDivergenceError`  
**Cause**: Apprentissage divergent  
**Solution**:
- Ajuster le learning rate
- Normaliser les données
- Utiliser des callbacks (EarlyStopping)

### 3. Problèmes de Trading
**Erreur**: `OrderExecutionError`  
**Cause**: API exchange indisponible  
**Solution**:
- Implémenter des retries
- Journaliser les erreurs
- Mode dégradé sans execution

**Erreur**: `RiskLimitExceeded`  
**Cause**: Dépassement des limites de risque  
**Solution**:
- Forcer la fermeture des positions
- Envoyer une alerte
- Suspension temporaire

## Journalisation des Erreurs
Toutes les erreurs sont journalisées dans:
- `logs/error.log` (détails techniques)
- `logs/trading.log` (activité de trading)

## Politique de Reessai
- 3 tentatives maximum
- Backoff exponentiel entre les tentatives
- Marquer les erreurs persistantes

```mermaid
graph TD
    A[Erreur] --> B{Tentatives < 3?}
    B -->|Oui| C[Reessayer après délai]
    B -->|Non| D[Journaliser erreur]
    D --> E[Notifier l'admin]
    C --> F[Succès?]
    F -->|Oui| G[Continuer]
    F -->|Non| B
