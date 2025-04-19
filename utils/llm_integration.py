import os
import json
import time
import hashlib
import redis
import subprocess
import google.generativeai as genai
from pathlib import Path
from typing import Optional, Dict, Any
from config import config

def run_in_llm_env(python_code):
    """Execute code in isolated LLM environment"""
    try:
        result = subprocess.run(
            ['conda', 'run', '-n', 'llm_env', 'python', '-c', python_code],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception as e:
        print(f"Error running in llm_env: {str(e)}")
        return None

# Initialisation Redis
try:
    redis_client = redis.Redis(
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db,
        decode_responses=True
    )
    redis_client.ping()  # Test connection
    redis_available = True
except Exception as e:
    print(f"Redis initialization failed: {str(e)}")
    redis_available = False
    redis_client = None

class LLMIntegration:
    def __init__(self):
        # Configuration
        self.cache_dir = Path("data/llm_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialisation des services
        self.init_google_ai()
        self.init_openrouter()
        
    def init_google_ai(self):
        """Initialise le service Google AI"""
        try:
            genai.configure(api_key=config.GOOGLE_API_KEY)
            self.google_model = genai.GenerativeModel('gemini-1.5-pro-latest')
            self.google_available = True
        except Exception as e:
            print(f"Google AI initialization failed: {str(e)}")
            self.google_available = False

    def init_openrouter(self):
        """Initialise le service OpenRouter"""
        self.openrouter_headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://morningstar-trading.com",
            "X-Title": "Morningstar Trading"
        }
        self.openrouter_available = bool(config.OPENROUTER_API_KEY)

    def get_cache_path(self, text: str) -> Path:
        """Génère un chemin de cache unique pour le texte"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.cache_dir / f"{text_hash}.json"

    def get_embeddings(self, text: str) -> Optional[list[float]]:
        """Obtient les embeddings en utilisant l'environnement isolé"""
        # Vérifier le cache Redis d'abord si disponible
        if redis_available and redis_client:
            try:
                cache_key = f"llm_emb:{hashlib.md5(text.encode()).hexdigest()}"
                cached = redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                print(f"Redis cache access failed: {str(e)}")
                globals()['redis_available'] = False
        
        # Vérifier le cache fichier
        cache_path = self.get_cache_path(text)
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    embeddings = json.load(f)
                    # Mettre à jour le cache Redis si disponible
                    if redis_available and redis_client:
                        try:
                            cache_key = f"llm_emb:{hashlib.md5(text.encode()).hexdigest()}"
                            redis_client.setex(cache_key, config.redis.cache_ttl, json.dumps(embeddings))
                        except Exception as e:
                            print(f"Failed to update Redis cache: {str(e)}")
                    return embeddings
            except Exception as e:
                        print(f"Failed to read file cache: {str(e)}")

        # Vérifier la disponibilité du service AVANT d'exécuter le code isolé
        if not self.google_available:
             print("Google AI service not available, cannot get embeddings.")
             return None
             
        # Exécuter dans l'environnement isolé
        python_code = f"""
import google.generativeai as genai
import json

genai.configure(api_key='{config.GOOGLE_API_KEY}')
model = genai.GenerativeModel('gemini-pro')

try:
    # Nouvelle méthode pour Gemini 1.5+
    response = genai.embed_content(
        model='models/embedding-001',
        content='{text}',
        task_type='retrieval_document'
    )
    print(json.dumps(response['embedding']))
except Exception as e:
    print(str(e))
"""
        result = run_in_llm_env(python_code)
        if result:
            try:
                embeddings = json.loads(result)
                self._save_to_cache(cache_path, embeddings)
                return embeddings
            except json.JSONDecodeError:
                print(f"LLM service error: {result}")
        
        return None

    def _save_to_cache(self, path: Path, data: Any):
        """Sauvegarde les données dans le cache Redis et fichier"""
        try:
            # Sauvegarde Redis si disponible
            if redis_available and redis_client:
                cache_key = f"llm_emb:{hashlib.md5(str(data).encode()).hexdigest()}"
                try:
                    redis_client.setex(cache_key, config.redis.cache_ttl, json.dumps(data))
                except Exception as e:
                    print(f"Redis cache save failed: {str(e)}")
                    globals()['redis_available'] = False
        
            # Sauvegarde fichier
            with open(path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Failed to save cache: {str(e)}")

    def get_llm_analysis(self, text: str) -> Dict[str, Any]:
        """Obtient une analyse complète du texte avec sentiment et catégories"""
        # Vérifier le cache
        cache_key = f"llm_analysis:{hashlib.md5(text.encode()).hexdigest()}"
        if redis_available and redis_client:
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                print(f"Redis cache access failed: {str(e)}")
                globals()['redis_available'] = False
        
        # Vérifier le cache fichier
        cache_path = self.get_cache_path(f"analysis_{hashlib.md5(text.encode()).hexdigest()}")
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    analysis = json.load(f)
                    if redis_available and redis_client:
                        try:
                            redis_client.setex(cache_key, config.redis.cache_ttl, json.dumps(analysis))
                        except Exception as e:
                            print(f"Failed to update Redis cache: {str(e)}")
                    return analysis
            except Exception as e:
                print(f"Failed to read file cache: {str(e)}")
        
        # Exécuter dans l'environnement isolé
        python_code = f"""
import google.generativeai as genai
import json

genai.configure(api_key='{config.GOOGLE_API_KEY}')
model = genai.GenerativeModel('gemini-1.5-pro-latest')

prompt = \"\"\"Analyze this trading-related text and return JSON with:
- sentiment (positive/neutral/negative)
- confidence (0-1)
- key_phrases (list)
- potential_actions (list)
- relevance_to_trading (0-1)

Text: {text}\"\"\"

try:
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            top_p=0.8
        )
    )
    print(response.text)
except Exception as e:
    print(str(e))
"""
        result = run_in_llm_env(python_code)
        if result:
            try:
                # Nettoyer la réponse pour enlever les marqueurs de bloc de code
                cleaned_result = result.strip().removeprefix('```json').removesuffix('```').strip()
                analysis = json.loads(cleaned_result)
                self._save_to_cache(cache_path, analysis)
                if redis_available and redis_client:
                    try:
                        redis_client.setex(cache_key, config.redis.cache_ttl, json.dumps(analysis))
                    except Exception as e:
                        print(f"Failed to save to Redis: {str(e)}")
                return analysis
            except json.JSONDecodeError:
                print(f"LLM analysis error: {result}")
                return {
                    "error": "Failed to parse LLM response",
                    "raw_response": result
                }
        
        return {
            "error": "LLM analysis failed",
            "text": text
        }
