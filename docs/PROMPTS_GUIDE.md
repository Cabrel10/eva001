# Guide des Prompts LLM - Morningstar V2

**Objectif :**
Fournir un guide complet des prompts utilisés pour interroger le LLM dans le cadre de la surveillance et l’analyse contextuelle du modèle de trading Morningstar V2.

---

## Contexte d’Utilisation

Le LLM (Large Language Model) intégré dans Morningstar V2 joue un rôle de **supervision et d'aide à la décision**, et non de décideur final. Ses principales fonctions sont :

1.  **Analyse de Contexte du Marché**: Fournir une évaluation qualitative de l'état actuel du marché (sentiment général, événements majeurs, risques potentiels) basée sur les données récentes (news, social media, indicateurs macro).
2.  **Vérification de Cohérence des Signaux**: Évaluer si un signal de trading généré par le modèle technique (`EnhancedHybridModel`) est cohérent avec le contexte actuel du marché et les données alternatives.
3.  **Recommandations de Gestion de Risque/Capital**: Suggérer des ajustements de taille de position, des niveaux de Stop-Loss/Take-Profit ou des stratégies de diversification basées sur le contexte, la volatilité prédite et le profil de risque configuré.
4.  **Diagnostic d'Anomalies**: Aider à identifier les raisons potentielles derrière des performances inattendues ou des erreurs dans le workflow.

**Important**: Le LLM complète l'analyse quantitative du modèle principal par une couche d'interprétation qualitative et contextuelle. Les décisions finales d'exécution restent sous le contrôle de la logique définie dans `workflows/trading_workflow.py` et des paramètres de configuration.

---

## Exemples de Prompts et Comportements Attendus

Les prompts doivent être clairs, fournir suffisamment de contexte et spécifier le format de réponse attendu.

### 1. Analyse de Contexte

*   **Prompt Type 1 (Général)**:
    ```
    Analyse le contexte global du marché crypto pour la période [DATE_DEBUT] à [DATE_FIN/MAINTENANT]. Quels sont les événements majeurs, le sentiment dominant (basé sur [SOURCES: news, twitter...]) et les risques potentiels à surveiller ? Fournis un résumé concis et un score de risque global (0-10).
    ```
    *   **Attente**: Résumé textuel (ex: "Marché nerveux suite aux annonces réglementaires US, sentiment Twitter majoritairement négatif, risque de volatilité accrue.") et un score (ex: `Risque: 7/10`).

*   **Prompt Type 2 (Spécifique à une paire)**:
    ```
    Analyse le contexte spécifique pour la paire [PAIRE] (ex: BTC/USDT) à [DATETIME]. Intègre les dernières news, les données on-chain pertinentes (si disponibles) et le sentiment social. Y a-t-il des signaux d'alerte spécifiques à cet actif ? Score de sentiment (-1 à 1).
    ```
    *   **Attente**: Analyse ciblée (ex: "Forte activité on-chain pour BTC, news positives sur adoption institutionnelle, mais sentiment social mitigé.") et un score (ex: `Sentiment: 0.2`).

### 2. Vérification de Cohérence de Signal

*   **Prompt**:
    ```
    Le modèle EnhancedHybridModel a généré un signal '[SIGNAL_TYPE: ACHAT/VENTE/NEUTRE]' pour [PAIRE] à [DATETIME] avec un score de confiance de [SCORE_CONFIANCE].
    Contexte actuel du marché : [RESUME_CONTEXTE_MARCHE_DU_LLM_PRECEDENT ou DONNEES_BRUTES].
    Indicateurs techniques clés : [LISTE_INDICATEURS_ET_VALEURS].
    Ce signal te semble-t-il cohérent avec le contexte global et les indicateurs ? Fournis une évaluation (Cohérent / Peu Cohérent / Incohérent) et une brève justification.
    ```
    *   **Attente**: Évaluation + Justification (ex: `Évaluation: Peu Cohérent. Justification: Signal d'achat technique faible dans un contexte de marché globalement baissier et sentiment négatif. Risque de faux signal élevé.`).

### 3. Recommandation de Gestion de Risque/Capital

*   **Prompt Type 1 (Allocation)**:
    ```
    Profil de risque configuré : [PROFIL: Prudent/Modéré/Agressif].
    Contexte marché : [RESUME_CONTEXTE].
    Volatilité prédite pour [PAIRE] : [NIVEAU: Basse/Moyenne/Haute].
    Quelle taille de position (en % du capital alloué au trading) suggères-tu pour un potentiel trade sur [PAIRE] ? Justifie brièvement.
    ```
    *   **Attente**: Suggestion + Justification (ex: `Suggestion: 1.5%. Justification: Profil Prudent et volatilité prédite Moyenne justifient une exposition limitée.`).

*   **Prompt Type 2 (Ajustement SL/TP)**:
    ```
    Un trade [TYPE: Long/Short] est envisagé sur [PAIRE].
    Niveau d'entrée potentiel : [PRIX_ENTREE].
    Volatilité prédite : [NIVEAU].
    Régime de marché prédit : [REGIME: Tendance/Range].
    Basé sur ces éléments et le contexte [RESUME_CONTEXTE], suggère des niveaux de Stop-Loss (SL) et Take-Profit (TP) raisonnables.
    ```
    *   **Attente**: Niveaux suggérés (ex: `SL suggéré: [PRIX_SL]. TP suggéré: [PRIX_TP]. Justification: Niveaux basés sur [METHODE: ex: ATR, niveaux de support/résistance] ajustés à la volatilité et au régime de marché.`).

### 4. Diagnostic d'Anomalies

*   **Prompt**:
    ```
    Le système a subi une série de [NOMBRE] trades perdants sur [PAIRE] entre [DATE_DEBUT] et [DATE_FIN].
    Voici les logs des signaux et décisions : [EXTRAIT_LOGS].
    Le contexte marché durant cette période était : [RESUME_CONTEXTE].
    Peux-tu analyser ces informations et suggérer des causes possibles à cette sous-performance (ex: changement de régime non détecté, problème de données, pertinence des indicateurs, signaux LLM ignorés...) ?
    ```
    *   **Attente**: Analyse diagnostique (ex: "Causes possibles : 1. Le modèle semble avoir mal interprété un changement rapide de régime vers une forte volatilité. 2. Les signaux de vente ont été générés tardivement. 3. La supervision LLM avait signalé une incohérence sur 2 des 5 trades, qui n'a pas été prise en compte.").

---

## Conseils de Format et Structure des Réponses

*   **Clarté et Concision**: Les réponses doivent être directes et faciles à interpréter par le système.
*   **Scores et Indicateurs**: Utiliser des scores numériques standardisés (ex: -1 à 1 pour sentiment, 0-10 pour risque) lorsque c'est pertinent.
*   **Justification**: Toujours demander une brève justification pour comprendre le raisonnement du LLM.
*   **Format Structuré**: Préférer des réponses structurées (ex: JSON ou points clés) pour faciliter le parsing automatique par `workflows/trading_workflow.py` ou `live/monitoring.py`.
    *   *Exemple JSON pour Cohérence Signal*:
        ```json
        {
          "evaluation": "Peu Cohérent",
          "justification": "Signal d'achat technique faible dans un contexte de marché globalement baissier et sentiment négatif. Risque de faux signal élevé.",
          "score_confiance_llm": 0.3
        }
        ```

---

## Mise à Jour et Adaptabilité

*   **Itération**: Les prompts devront être testés et affinés pendant les phases de backtesting et de paper trading.
*   **Versioning**: Envisager un versioning simple des prompts si des changements majeurs sont apportés.
*   **Priorisation**: Se concentrer initialement sur les prompts d'analyse de contexte et de vérification de cohérence, qui apportent le plus de valeur ajoutée pour la supervision.

Ce guide sert de point de départ. La qualité et la pertinence des interactions avec le LLM dépendront fortement de la précision des prompts et de la qualité des données fournies en contexte.
