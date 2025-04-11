#!/usr/bin/env python3
"""
Script complet pour préparer un dataset d'entraînement avec :
- Données OHLCV 1m (SHIB, DOGE, SOL, YFI)
- Données réseaux sociaux (Twitter/Reddit)
- Données on-chain (hashrate, transactions)
- Données de liquidités
- Normalisation zero mapping
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import asyncio # Ensure asyncio is imported
from Morningstar.utils.data_manager import ExchangeDataManager
from Morningstar.utils.social_scraper import SocialMediaScraper as SocialDataScraper
from Morningstar.utils.custom_indicators import add_technical_indicators
import pytz
from sklearn.preprocessing import RobustScaler # Import RobustScaler

class FullDatasetBuilder:
    def __init__(self):
        self.data_manager = ExchangeDataManager('binance')
        from Morningstar.configs.social_config import TWITTER_KEYS, REDDIT_KEYS
        self.social_scraper = SocialDataScraper(TWITTER_KEYS, REDDIT_KEYS)
        self.timezone = pytz.UTC
        
    async def build_dataset(self, pairs: List[str], 
                          start_date: str, end_date: str) -> pd.DataFrame:
        """
        Construit le dataset complet avec toutes les données demandées (async)
        
        Args:
            pairs: Liste des paires (ex: ['SHIBUSDT', 'DOGEUSDT'])
            start_date: Date de début (format: 'YYYY-MM-DD')
            end_date: Date de fin (format: 'YYYY-MM-DD')
            
        Returns:
            DataFrame contenant toutes les données fusionnées et normalisées
        """
        # 1. Télécharger les données OHLCV 1m (await added)
        ohlcv_data = await self._get_ohlcv_data(pairs, '1m', start_date, end_date)
        
        # 2. Ajouter les indicateurs techniques
        ohlcv_data = self._add_technical_indicators(ohlcv_data)
        
        # 3. Récupérer les données sociales
        social_data = await self._get_social_data(pairs, start_date, end_date)
        
        # 4. Fusionner toutes les données
        full_data = self._merge_all_data(ohlcv_data, social_data)
        
        # 5. Nettoyage et normalisation
        full_data = self._clean_and_normalize(full_data)
        
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

    async def _get_social_data(self, pairs: List[str], 
                             start_date: str, end_date: str) -> pd.DataFrame:
        """Récupère les données sociales pour toutes les paires"""
        social_data = {}
        
        # Mapping des paires (format BASE/QUOTE) aux repos GitHub pertinents
        github_repos = {
            'SHIB/USDT': 'Shiba-Inu-Developers/shiba-inu',
            'DOGE/USDT': 'dogecoin/dogecoin',
            'SOL/USDT': 'solana-labs/solana',
            'YFI/USDT': 'yearn/yearn-finance'
        }
        
        # Use asyncio.gather for concurrent social data fetching
        social_tasks = []
        for pair in pairs:
            print(f"Préparation de la récupération des données sociales pour {pair}...")
            
            # Create tasks for each social source
            twitter_task = asyncio.create_task(self.social_scraper.get_twitter_sentiment(
                query=pair, # Use the full pair string for query
                start_date=start_date,
                end_date=end_date
            ))
            
            # Use the base currency for subreddit lookup
            base_currency = pair.split('/')[0] 
            reddit_task = asyncio.create_task(self.social_scraper.get_reddit_sentiment(
                subreddit=base_currency, 
                limit=100
            ))
            
            github_task = None
            if pair in github_repos:
                 github_task = asyncio.create_task(self.social_scraper.get_github_activity(
                     repo=github_repos[pair],
                     start_date=start_date,
                     end_date=end_date
                 ))
            
            social_tasks.append({
                'pair': pair,
                'twitter': twitter_task,
                'reddit': reddit_task,
                'github': github_task
            })

        # Execute all social tasks concurrently
        # We need to await each task individually after gather returns results
        # because gather itself returns the results (or exceptions)
        
        # Gather results (coroutines or exceptions)
        all_task_results = await asyncio.gather(
            *[task['twitter'] for task in social_tasks],
            *[task['reddit'] for task in social_tasks],
            *[task['github'] for task in social_tasks if task['github']],
            return_exceptions=True # Return exceptions instead of raising them immediately
        )

        # Process results
        num_pairs = len(pairs)
        twitter_results = all_task_results[0:num_pairs]
        reddit_results = all_task_results[num_pairs:2*num_pairs]
        github_results_list = all_task_results[2*num_pairs:]
        
        github_results_map = {}
        github_idx = 0
        for task_info in social_tasks:
             if task_info['github']:
                 github_results_map[task_info['pair']] = github_results_list[github_idx]
                 github_idx += 1

        for i, task_info in enumerate(social_tasks):
            pair = task_info['pair']
            print(f"Traitement des données sociales pour {pair}...")
            
            twitter_data = twitter_results[i]
            reddit_data = reddit_results[i]
            github_data = github_results_map.get(pair, None) # Get result from map

            # Handle potential exceptions returned by gather
            if isinstance(twitter_data, Exception):
                print(f"Erreur Twitter pour {pair}: {twitter_data}")
                twitter_data = pd.DataFrame() # Use empty DataFrame on error
            if isinstance(reddit_data, Exception):
                print(f"Erreur Reddit pour {pair}: {reddit_data}")
                reddit_data = pd.DataFrame()
            if isinstance(github_data, Exception):
                print(f"Erreur GitHub pour {pair}: {github_data}")
                github_data = pd.DataFrame()
            elif github_data is None: # Handle case where no GitHub task was created
                 github_data = pd.DataFrame()

            # Ensure results are DataFrames before concat
            twitter_data = twitter_data if isinstance(twitter_data, pd.DataFrame) else pd.DataFrame()
            reddit_data = reddit_data if isinstance(reddit_data, pd.DataFrame) else pd.DataFrame()
            github_data = github_data if isinstance(github_data, pd.DataFrame) else pd.DataFrame()

            # Fusion des données sociales pour la paire actuelle
            # Use outer join to keep all timestamps and fill NaNs later if needed
            current_social = pd.concat([twitter_data, reddit_data, github_data], axis=1) 
            social_data[pair] = current_social

        # Original sequential code (replaced by concurrent version above):
        # for pair in pairs:
        #     print(f"Récupération des données sociales pour {pair}...")
            
        #     # Twitter (await added)
        #     twitter_data = await self.social_scraper.get_twitter_sentiment(
        #         query=pair,
        #         start_date=start_date,
        #         end_date=end_date
        #     )
            
        #     # Reddit (await added)
        #     reddit_data = await self.social_scraper.get_reddit_sentiment(
        #         subreddit=pair.replace('/USDT', ''), # Adjusted replace for new format
        #         limit=100 # Corrected indentation
        #     ) # Corrected indentation
            
        #     # GitHub (si disponible) # Corrected indentation
        #     github_data = pd.DataFrame() # Corrected indentation
        #     if pair in github_repos: # Corrected indentation
        #         # await added # Corrected indentation
        #         github_data = await self.social_scraper.get_github_activity( # Corrected indentation
        #             repo=github_repos[pair], # Corrected indentation
        #             start_date=start_date, # Corrected indentation
        #             end_date=end_date # Corrected indentation
        #         ) # Corrected indentation
            
        #     # Fusion # Corrected indentation
        #     social_data[pair] = pd.concat([twitter_data, reddit_data, github_data], axis=1) # Corrected indentation
            
        return social_data

    def _merge_all_data(self, ohlcv_data: Dict, social_data: Dict) -> pd.DataFrame:
        """Fusionne toutes les sources de données"""
        merged_data = []
        
        for pair in ohlcv_data.keys():
            # Fusion OHLCV + Social
            df = pd.merge(
                ohlcv_data[pair],
                social_data[pair],
                left_index=True,
                right_index=True,
                how='left'
            )
            df['pair'] = pair  # Ajout de l'identifiant de paire
            merged_data.append(df)
        
        return pd.concat(merged_data)

    def _clean_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoyage et normalisation améliorés des données"""
        
        # 1. Forward fill général pour combler les NaN (OHLCV, indicateurs, social si possible)
        # Exclure la colonne 'pair' si elle existe
        cols_to_fill = df.columns.difference(['pair'])
        df[cols_to_fill] = df[cols_to_fill].fillna(method='ffill')
        
        # 2. Identifier les colonnes sociales et de volume/indicateurs pour la mise à l'échelle
        social_cols = [c for c in df.columns if 'twitter' in c or 'reddit' in c or 'github' in c]
        # Identifier d'autres colonnes numériques à normaliser (exclure OHLC pour éviter de fausser les prix bruts si besoin)
        # Pour l'instant, normalisons toutes les features ajoutées (indicateurs, social)
        indicator_cols = df.columns.difference(['open', 'high', 'low', 'close', 'volume', 'pair'] + social_cols).tolist()
        cols_to_scale = social_cols + indicator_cols

        # 3. Remplir les NaN restants dans les colonnes sociales avec 0 (Zero mapping)
        df[social_cols] = df[social_cols].fillna(0)
        
        # 4. Appliquer RobustScaler aux colonnes sélectionnées
        # Vérifier qu'il reste des colonnes à scaler et qu'elles existent dans le df
        valid_cols_to_scale = [col for col in cols_to_scale if col in df.columns]
        if valid_cols_to_scale:
            # Gérer les colonnes potentiellement non numériques ou entièrement NaN
            numeric_cols_to_scale = df[valid_cols_to_scale].select_dtypes(include=np.number).columns.tolist()
            
            # Supprimer les colonnes avec une variance nulle (constantes) avant de scaler
            cols_with_variance = [col for col in numeric_cols_to_scale if df[col].nunique() > 1]

            if cols_with_variance:
                scaler = RobustScaler()
                # Fit and transform sur les données valides (non-NaN après ffill/fillna(0))
                df[cols_with_variance] = scaler.fit_transform(df[cols_with_variance])
            else:
                print("Avertissement : Aucune colonne avec variance trouvée pour la mise à l'échelle.")
        else:
             print("Avertissement : Aucune colonne valide identifiée pour la mise à l'échelle.")

        # 5. Supprimer les lignes initiales qui pourraient encore contenir des NaN 
        # (dues aux fenêtres des indicateurs/ffill au début)
        df = df.dropna() 
        
        return df

async def main():
    # Configuration
    # Corrected pair format to BASE/QUOTE
    pairs = ['SHIB/USDT', 'DOGE/USDT', 'SOL/USDT', 'YFI/USDT'] 
    start_date = '2022-12-12'
    end_date = '2025-02-10'
    
    # Construction du dataset
    builder = FullDatasetBuilder()
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

async def main():
    # Configuration
    # Corrected pair format to BASE/QUOTE
    pairs = ['SHIB/USDT', 'DOGE/USDT', 'SOL/USDT', 'YFI/USDT'] 
    start_date = '2022-12-12'
    end_date = '2025-02-10'
    
    # Construction du dataset
    builder = FullDatasetBuilder()
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
