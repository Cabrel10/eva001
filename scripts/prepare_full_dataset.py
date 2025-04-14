#!/usr/bin/env python3
"""
Script simplifié pour préparer un dataset d'entraînement avec :
- Données OHLCV 1h (SHIB, DOGE, SOL, YFI)
- Indicateurs techniques
- Normalisation des données
"""
import os
import pandas as pd
import numpy as np
from typing import List, Dict
import asyncio
from Morningstar.utils.data_manager import ExchangeDataManager
from Morningstar.utils.custom_indicators import add_technical_indicators
from sklearn.preprocessing import RobustScaler

class SimpleDatasetBuilder:
    def __init__(self):
        self.data_manager = ExchangeDataManager('kucoin')
        print("Utilisation de KuCoin comme exchange")
        
    async def build_dataset(self, pairs: List[str], 
                          start_date: str, end_date: str) -> pd.DataFrame:
        """
        Construit un dataset simplifié avec seulement les données OHLCV et indicateurs
        
        Args:
            pairs: Liste des paires (ex: ['SHIB/USDT', 'DOGE/USDT'])
            start_date: Date de début (format: 'YYYY-MM-DD')
            end_date: Date de fin (format: 'YYYY-MM-DD')
            
        Returns:
            DataFrame contenant les données OHLCV et indicateurs techniques
        """
        # 1. Télécharger les données OHLCV 1h (plus stable que 1m)
        ohlcv_data = await self._get_ohlcv_data(pairs, '1h', start_date, end_date)
        
        # 2. Ajouter les indicateurs techniques seulement
        full_data = self._add_technical_indicators(ohlcv_data)
        
        # 3. Fusionner les données (sans données sociales)
        merged = []
        for pair, df in full_data.items():
            df['pair'] = pair
            merged.append(df)
            
        if not merged:
            raise ValueError("Aucune donnée valide n'a été téléchargée")
            
        full_data = pd.concat(merged)
        
        # 4. Nettoyage et normalisation
        full_data = self._clean_and_normalize(full_data)
        
        # Ajout des colonnes sociales manquantes pour correspondre à la structure demandée
        for col in ["commits", "stars", "forks", "issues_opened", "issues_closed"]:
            if col not in full_data.columns:
                full_data[col] = None

        return full_data

    # Added async keyword
    async def _get_ohlcv_data(self, pairs: List[str], interval: str,
                       start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Récupère les données OHLCV pour toutes les paires (async)"""
        ohlcv = {}
        tasks = []
        # Prepare tasks for concurrent fetching
        for pair in pairs:
            print(f"Préparation du téléchargement pour {pair} {interval}...")
            # Create a coroutine task for each pair
            task = asyncio.create_task(self.data_manager.load_data(
                pair=pair,
                timeframe=interval,
                start_date=start_date,
                end_date=end_date
            ))
            tasks.append((pair, task)) # Store pair along with task

        # Execute tasks concurrently and gather results
        results = await asyncio.gather(*(task for _, task in tasks))

        # Populate the ohlcv dictionary with results
        for i, (pair, _) in enumerate(tasks):
             df = results[i]
             if df is not None and not df.empty:
                 print(f"Données {pair} téléchargées avec succès. Shape: {df.shape}")
                 ohlcv[pair] = df
             else:
                 print(f"Échec du téléchargement ou données vides pour {pair}.")
                 # Optionally handle the failure, e.g., skip the pair or raise an error
                 # For now, just print a message and don't add it to the dict
        
        # Original sequential code (replaced by concurrent version above):
        # for pair in pairs:
        #     print(f"Téléchargement des données {pair} {interval}...")
        #     # Added await keyword
        #     df = await self.data_manager.load_data(
        #         pair=pair,
        #         timeframe=interval,
        #         start_date=start_date,
        #         end_date=end_date # Corrected indentation
        #     ) # Corrected indentation
        #     ohlcv[pair] = df # Corrected indentation
        return ohlcv

    def _add_technical_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Ajoute les indicateurs techniques à chaque paire"""
        for pair, df in data.items():
            data[pair] = add_technical_indicators(df)
        return data


    def _clean_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoyage et normalisation simplifiés des données"""
        
        # 1. Forward fill pour combler les NaN
        df = df.fillna(method='ffill')
        
        # 2. Identifier les colonnes à normaliser (toutes sauf OHLC et pair)
        cols_to_scale = df.columns.difference(['open', 'high', 'low', 'close', 'volume', 'pair']).tolist()
        
        # 3. Appliquer RobustScaler aux colonnes sélectionnées
        if cols_to_scale:
            scaler = RobustScaler()
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale].fillna(0))
        
        # 4. Supprimer les NaN restants
        df = df.dropna()
        
        return df

async def main():
    # Configuration
    # Corrected pair format to BASE/QUOTE
    pairs = ['SHIB/USDT', 'DOGE/USDT', 'SOL/USDT', 'YFI/USDT'] 
    start_date = '2022-12-12'
    end_date = '2025-02-10'
    
    # Construction du dataset
    builder = SimpleDatasetBuilder()
    try:
        # Load markets before fetching data
        await builder.data_manager.load_markets_async()     
        
        dataset = await builder.build_dataset(pairs, start_date, end_date)
        
        # Sauvegarde
        # Ensure the directory exists
        output_dir = os.path.join('Morningstar', 'data')
        os.makedirs(output_dir, exist_ok=True) 
        output_path = os.path.join(output_dir, 'full_dataset.parquet')
        # Check if dataset is not empty before saving
        if dataset is not None and not dataset.empty:
            dataset.to_parquet(output_path)
            print(f"Dataset sauvegardé dans {output_path}")
        else:
            print("Le dataset est vide ou None, aucune sauvegarde effectuée.")

    except Exception as e:
        import traceback
        print(f"Une erreur est survenue lors de la construction du dataset: {e}")
        traceback.print_exc()
    finally:
        # Ensure the exchange connection is closed
        if builder and builder.data_manager:
            print("Fermeture de la connexion à l'échange...")
            await builder.data_manager.close()


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
