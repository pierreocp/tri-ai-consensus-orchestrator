#!/usr/bin/env python3
"""
Tri-AI Orchestrator
==================
Un orchestrateur permettant de faire converser trois modèles d'IA (ChatGPT, Claude, Gemini)
les uns avec les autres via leurs APIs respectives.

Auteur: Assistant IA
Version: 2.0
"""

from __future__ import annotations
import os
import re
import json
import argparse
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import httpx
import logging
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import self-improvement system
try:
    from self_improvement import run_self_improvement_mode
    SELF_IMPROVEMENT_AVAILABLE = True
except ImportError:
    SELF_IMPROVEMENT_AVAILABLE = False

# Constants
OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/messages"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
GEMINI_VERSION = "v1"

MAX_RETRIES = 3
INITIAL_BACKOFF = 1
MAX_TOKENS_LIMIT = 4096
MIN_TURNS = 1
MAX_TURNS = 50

@dataclass
class ChatMessage:
    """Représente un message dans la conversation."""
    role: str  # 'system', 'user', 'assistant'
    name: str  # Nom de l'agent ou 'moderator'
    content: str  # Contenu du message

@dataclass
class ModelClient:
    """Classe de base pour les clients de modèles d'IA."""
    name: str  # Nom de l'agent (ChatGPT, Claude, Gemini)
    model: str  # Nom du modèle à utiliser

    async def send(self, messages: List[ChatMessage], *, max_tokens: int, timeout: int, verbose: bool) -> str:
        """Envoie les messages au modèle et retourne la réponse."""
        raise NotImplementedError

async def _post_with_retry(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    """Effectue un appel POST avec retry automatique en cas d'échec."""
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.post(url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.TimeoutException as e:
            logging.warning(f"Timeout attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            if attempt + 1 == MAX_RETRIES:
                raise
            await asyncio.sleep(INITIAL_BACKOFF * (2 ** attempt))
        except httpx.HTTPStatusError as e:
            logging.warning(f"HTTP error attempt {attempt + 1}/{MAX_RETRIES}: {e.response.status_code}")
            if attempt + 1 == MAX_RETRIES:
                raise
            await asyncio.sleep(INITIAL_BACKOFF * (2 ** attempt))
        except httpx.RequestError as e:
            logging.warning(f"Request error attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            if attempt + 1 == MAX_RETRIES:
                raise
            await asyncio.sleep(INITIAL_BACKOFF * (2 ** attempt))

class OpenAIClient(ModelClient):
    """Client pour l'API OpenAI (ChatGPT)."""
    base_url: str = OPENAI_BASE_URL

    async def send(self, messages: List[ChatMessage], *, max_tokens: int, timeout: int, verbose: bool) -> str:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY manquant")
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content, **({"name": m.name} if m.name else {})} for m in messages],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await _post_with_retry(client, self.base_url, headers=headers, json=payload)
            if verbose:
                logging.info(f"[OpenAI raw]\n{r.text}\n")
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()

class AnthropicClient(ModelClient):
    """Client pour l'API Anthropic (Claude)."""
    base_url: str = ANTHROPIC_BASE_URL

    async def send(self, messages: List[ChatMessage], *, max_tokens: int, timeout: int, verbose: bool) -> str:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY manquant")
        # Anthropic API requires system prompt as separate parameter
        system_prompts = [m.content for m in messages if m.role == "system"]
        system_prompt = "\n\n".join(system_prompts) if system_prompts else None
        
        # Convert messages (excluding system messages)
        converted = [
            {"role": "user" if m.role == "user" else "assistant", "content": m.content}
            for m in messages if m.role != "system"
        ]
        
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "messages": converted,
        }
        if system_prompt:
            payload["system"] = system_prompt
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            if verbose:
                print(f"[Claude Debug] Payload: {json.dumps(payload, indent=2)}")
            r = await _post_with_retry(client, self.base_url, headers=headers, json=payload)
            if verbose:
                print(f"[Claude Debug] Response: {r.text}")
            data = r.json()
            blocks = data.get("content", [])
            text_parts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
            result = "\n".join(text_parts).strip()
            if verbose:
                print(f"[Claude Debug] Extracted text: '{result}'")
            # Handle empty responses
            if not result:
                if verbose:
                    print("[Claude Debug] Empty response, using fallback")
                return "Je préfère ne pas répondre à cette question pour le moment."
            return result

MENTION_REGEX = re.compile(r"\b@([A-Za-z0-9_-]+)\b", re.IGNORECASE)

class GeminiClient(ModelClient):
    """Client pour l'API Google Gemini."""
    BASE = GEMINI_BASE_URL
    VERSION = GEMINI_VERSION

    def _url(self) -> str:
        """Construit l'URL pour l'API Gemini."""
        return f"{self.BASE}/{self.VERSION}/models/{self.model}:generateContent"

    async def send(self, messages: List[ChatMessage], *, max_tokens: int, timeout: int, verbose: bool) -> str:
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY manquant")
        url = self._url()
        converted = []
        for m in messages:
            role = "user" if m.role in ("user", "system") else "model"
            converted.append({"role": role, "parts": [{"text": m.content}]})
        payload = {
            "contents": converted,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.7,
            },
        }
        params = {"key": key}
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await _post_with_retry(client, url, params=params, json=payload)
            if verbose:
                logging.info(f"[Gemini raw]\n{r.text}\n")
            data = r.json()
            candidates = data.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                return "\n".join(p.get("text", "") for p in parts).strip() or "(réponse vide)"
            raise RuntimeError(f"Réponse Gemini inattendue: {json.dumps(data, indent=2)[:500]}")

@dataclass
class OrchestratorConfig:
    turns: int = 8
    strategy: str = "mention"
    max_tokens: int = 512
    timeout: int = 60
    verbose: bool = False

@dataclass
class Orchestrator:
    agents: List[ModelClient]
    config: OrchestratorConfig
    system_prompt: str
    transcript: List[ChatMessage] = field(default_factory=list)

    def __post_init__(self):
        if self.config.verbose:
            logging.basicConfig(filename='tri_ai_orchestrator.log', level=logging.INFO, format='%(asctime)s %(message)s')

    def _find_agent_by_name(self, name: str) -> Optional[ModelClient]:
        lname = name.lower()
        for a in self.agents:
            if a.name.lower() == lname:
                return a
        logging.warning(f"Mention to unknown agent: {name}")
        return None

    def _next_round_robin(self, current: Optional[ModelClient]) -> ModelClient:
        if current is None:
            return self.agents[0]
        idx = self.agents.index(current)
        return self.agents[(idx + 1) % len(self.agents)]

    def _detect_mention(self, text: str, current: ModelClient) -> Optional[ModelClient]:
        for m in MENTION_REGEX.findall(text):
            candidate = self._find_agent_by_name(m)
            if candidate and candidate is not current:
                return candidate
        return None

    async def run(self, user_prompt: str):
        self.transcript.append(ChatMessage(role="system", name="moderator", content=self.system_prompt))
        self.transcript.append(ChatMessage(role="user", name="moderator", content=user_prompt))
        current: Optional[ModelClient] = None
        for i in range(self.config.turns):
            if self.config.strategy == "round":
                current = self._next_round_robin(current)
            else:
                if current is None:
                    current = self.agents[0]
                else:
                    last_content = self.transcript[-1].content if self.transcript else ""
                    mentioned = self._detect_mention(last_content, current)
                    current = mentioned or self._next_round_robin(current)
            print(f"\n===== Tour {i+1}: {current.name} parle =====")
            try:
                reply = await current.send(self.transcript, max_tokens=self.config.max_tokens, timeout=self.config.timeout, verbose=self.config.verbose)
                # Remove self-mentions
                reply = re.sub(rf'@{re.escape(current.name)}\b', '', reply, flags=re.IGNORECASE).strip()
                # Clean up extra spaces
                reply = ' '.join(reply.split())
            except Exception as e:
                reply = f"[ERROR {current.name}] {e}"
            print(reply)
            self.transcript.append(ChatMessage(role="assistant", name=current.name, content=reply))
        print("\n===== Transcript final =====\n")
        for m in self.transcript:
            tag = f"{m.role.upper()}({m.name})"
            print(f"[{tag}]\n{m.content}\n")

def validate_args(args: argparse.Namespace) -> None:
    """Valide les arguments de ligne de commande."""
    if args.turns < MIN_TURNS or args.turns > MAX_TURNS:
        raise ValueError(f"Le nombre de tours doit être entre {MIN_TURNS} et {MAX_TURNS}")
    
    if args.max_tokens <= 0 or args.max_tokens > MAX_TOKENS_LIMIT:
        raise ValueError(f"Le nombre de tokens doit être entre 1 et {MAX_TOKENS_LIMIT}")
    
    if args.timeout <= 0:
        raise ValueError("Le timeout doit être positif")
    
    if not args.prompt.strip():
        raise ValueError("Le prompt ne peut pas être vide")

def check_api_keys() -> None:
    """Vérifie que toutes les clés API nécessaires sont présentes."""
    required_keys = {
        "OPENAI_API_KEY": "OpenAI (ChatGPT)",
        "ANTHROPIC_API_KEY": "Anthropic (Claude)", 
        "GOOGLE_API_KEY": "Google (Gemini)"
    }
    
    missing_keys = []
    for key, service in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"{key} pour {service}")
    
    if missing_keys:
        raise RuntimeError(
            f"Clés API manquantes: {', '.join(missing_keys)}. "
            "Veuillez les définir dans votre fichier .env"
        )

def parse_args() -> argparse.Namespace:
    """Parse et valide les arguments de ligne de commande."""
    p = argparse.ArgumentParser(
        description="Fait discuter ChatGPT, Claude, Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s --prompt "Débattons de l'AGI" --turns 6 --strategy mention
  %(prog)s --prompt "Analysez ce code" --turns 4 --verbose
"""
    )
    p.add_argument("--prompt", required=True, help="Sujet ou question de départ")
    p.add_argument("--turns", type=int, default=8, 
                   help=f"Nombre de tours au total ({MIN_TURNS}-{MAX_TURNS})")
    p.add_argument("--strategy", choices=["mention", "round"], default="mention",
                   help="Stratégie de passage de parole")
    p.add_argument("--max_tokens", type=int, default=512,
                   help=f"Nombre maximum de tokens par réponse (1-{MAX_TOKENS_LIMIT})")
    p.add_argument("--timeout", type=int, default=60,
                   help="Timeout en secondes pour les appels API")
    p.add_argument("--verbose", action="store_true",
                   help="Active le mode verbeux avec logs détaillés")
    p.add_argument("--self-improve", action="store_true",
                   help="Active le mode d'auto-amélioration (EXPERIMENTAL)")
    p.add_argument("--openai_model", default="gpt-4o",
                   help="Modèle OpenAI à utiliser")
    p.add_argument("--anthropic_model", default="claude-3-5-sonnet-20240620",
                   help="Modèle Anthropic à utiliser")
    p.add_argument("--gemini_model", default="gemini-1.5-pro",
                   help="Modèle Gemini à utiliser")
    
    args = p.parse_args()
    validate_args(args)
    return args

async def main():
    """Fonction principale du programme."""
    try:
        args = parse_args()
        check_api_keys()
        
        # Configuration du logging
        if args.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('tri_ai_orchestrator.log'),
                    logging.StreamHandler()
                ]
            )
            logging.info("Mode verbeux activé")
        
        agents: List[ModelClient] = [
            OpenAIClient(name="ChatGPT", model=args.openai_model),
            AnthropicClient(name="Claude", model=args.anthropic_model),
            GeminiClient(name="Gemini", model=args.gemini_model),
        ]
        
        system_prompt = (
            "Tu participes à une discussion entre IA. Réponds naturellement et de manière constructive. "
            "Si tu veux qu'un autre agent réponde, mentionne @ChatGPT, @Claude ou @Gemini. "
            "Sois concis mais informatif."
        )
        
        orch = Orchestrator(
            agents=agents,
            config=OrchestratorConfig(
                turns=args.turns,
                strategy=args.strategy,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                verbose=args.verbose,
            ),
            system_prompt=system_prompt,
        )
        
        # Mode d'auto-amélioration
        if getattr(args, 'self_improve', False):
            if not SELF_IMPROVEMENT_AVAILABLE:
                raise RuntimeError("Le module d'auto-amélioration n'est pas disponible")
            
            print("🤖 Mode d'auto-amélioration activé!")
            print("⚠️  ATTENTION: Mode expérimental - utilisez à vos risques et périls")
            
            # Demander confirmation
            response = input("Voulez-vous vraiment continuer? (oui/non): ")
            if response.lower() not in ['oui', 'o', 'yes', 'y']:
                print("🚫 Opération annulée")
                return
            
            await run_self_improvement_mode(__file__, orch)
            return
        
        logging.info(f"Démarrage de la conversation avec {args.turns} tours")
        await orch.run(args.prompt)
        logging.info("Conversation terminée avec succès")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interruption par l'utilisateur")
    except (ValueError, RuntimeError) as e:
        print(f"❌ Erreur: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"Erreur inattendue: {e}")
        print(f"❌ Erreur inattendue: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
