#!/usr/bin/env python3
"""
Système de consensus et de décision collaborative entre IA
Permet aux IA de voter, débattre et collaborer pour améliorer leur propre code
"""

import asyncio
import json
import re
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum
import logging

# Import des clients IA existants
from tri_ai_orchestrator import OpenAIClient, AnthropicClient, GeminiClient, ChatMessage

def safe_json_parse(json_str):
    """Parse JSON de manière sécurisée avec nettoyage préalable"""
    try:
        # Nettoie les caractères spéciaux
        cleaned_str = json_str.strip()
        cleaned_str = re.sub(r'[\n\r\t]', ' ', cleaned_str)
        
        # Vérifier les guillemets non fermés
        quote_count = cleaned_str.count('"')
        if quote_count % 2 != 0:
            print(f"⚠️ Nombre impair de guillemets détecté: {quote_count}")
            return None
            
        # Parser le JSON
        return json.loads(cleaned_str)
        
    except json.JSONDecodeError as e:
        print(f"❌ Erreur de parsing JSON: {str(e)}")
        print(f"📝 Chaîne problématique: {json_str[:200]}...")
        return None

class VoteType(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    PROPOSE_MODIFICATION = "propose_modification"

class ProposalStatus(Enum):
    PROPOSED = "proposed"
    UNDER_DISCUSSION = "under_discussion"
    VOTING = "voting"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"

@dataclass
class Vote:
    agent_name: str
    vote_type: VoteType
    reasoning: str
    modification_suggestion: Optional[str] = None

@dataclass
class Proposal:
    id: str
    title: str
    description: str
    proposed_code: str
    proposer: str
    status: ProposalStatus
    votes: List[Vote]
    discussion_messages: List[ChatMessage]
    consensus_threshold: float = 0.67  # 2/3 majority

    def get_approval_rate(self) -> float:
        """Calcule le taux d'approbation"""
        if not self.votes:
            return 0.0
        approvals = sum(1 for v in self.votes if v.vote_type == VoteType.APPROVE)
        total_votes = len([v for v in self.votes if v.vote_type != VoteType.ABSTAIN])
        return approvals / total_votes if total_votes > 0 else 0.0

    def has_consensus(self) -> bool:
        """Vérifie si le consensus est atteint"""
        return self.get_approval_rate() >= self.consensus_threshold

    def get_modifications_suggested(self) -> List[str]:
        """Récupère toutes les modifications suggérées"""
        return [v.modification_suggestion for v in self.votes 
                if v.vote_type == VoteType.PROPOSE_MODIFICATION and v.modification_suggestion]

class ConsensusOrchestrator:
    def __init__(self, agents: List[Any]):
        self.agents = agents
        self.proposals: Dict[str, Proposal] = {}
        self.discussion_history: List[ChatMessage] = []
        
    async def initiate_proposal_discussion(self, topic: str) -> Optional[Proposal]:
        """Initie une discussion pour générer une proposition"""
        print(f"\n🎯 === INITIATION DE DISCUSSION: {topic} ===")
        
        # Phase 1: Brainstorming collaboratif
        brainstorm_prompt = f"""
MISSION COLLABORATIVE: {topic}

Vous devez maintenant collaborer pour proposer une amélioration concrète au script tri_ai_orchestrator.py.

RÈGLES:
1. Discutez de l'idée et de sa faisabilité
2. Proposez du code Python concret
3. Débattez des avantages/inconvénients
4. Arrivez à un consensus sur une proposition finale

Commencez la discussion ! @ChatGPT, lance le brainstorming.
"""
        
        # Discussion collaborative de 6 tours
        discussion_messages = await self._run_collaborative_discussion(brainstorm_prompt, 6)
        
        # Phase 2: Extraction de la proposition finale
        proposal = await self._extract_proposal_from_discussion(discussion_messages, topic)
        
        if proposal:
            self.proposals[proposal.id] = proposal
            print(f"✅ Proposition créée: {proposal.title}")
            
        return proposal
    
    async def _run_collaborative_discussion(self, prompt: str, turns: int) -> List[ChatMessage]:
        """Exécute une discussion collaborative entre les IA"""
        messages = [
            ChatMessage(role="system", name="moderator", 
                       content="Vous collaborez pour améliorer votre propre code. Soyez constructifs et précis."),
            ChatMessage(role="user", name="moderator", content=prompt)
        ]
        
        current_agent = 0
        for i in range(turns):
            agent = self.agents[current_agent % len(self.agents)]
            print(f"\n--- Tour {i+1}: {agent.name} ---")
            
            try:
                response = await agent.send(messages, max_tokens=300, timeout=30, verbose=False)
                print(f"{agent.name}: {response}")
                
                messages.append(ChatMessage(role="assistant", name=agent.name, content=response))
                
                # Détection de mention pour passer à l'agent suivant
                mentioned_agent = self._detect_mention(response)
                if mentioned_agent:
                    current_agent = self._get_agent_index(mentioned_agent)
                else:
                    current_agent += 1
                    
            except Exception as e:
                print(f"❌ Erreur {agent.name}: {e}")
                current_agent += 1
        
        return messages
    
    async def _extract_proposal_from_discussion(self, messages: List[ChatMessage], topic: str) -> Optional[Proposal]:
        """Extrait une proposition concrète de la discussion"""
        print("\n🔍 === EXTRACTION DE PROPOSITION ===")
        
        # Utilise ChatGPT pour synthétiser la discussion
        synthesis_prompt = f"""
Analysez cette discussion collaborative et extrayez UNE proposition concrète d'amélioration.

DISCUSSION:
{self._format_messages_for_analysis(messages)}

VOTRE MISSION:
Répondez UNIQUEMENT au format JSON suivant:
{{
    "title": "Titre court de la proposition",
    "description": "Description détaillée de l'amélioration",
    "proposed_code": "Code Python complet à implémenter",
    "feasible": true/false,
    "benefits": ["avantage1", "avantage2"],
    "risks": ["risque1", "risque2"]
}}

Si aucune proposition claire n'émerge, répondez: {{"feasible": false}}
"""
        
        try:
            chatgpt = self.agents[0]  # Utilise ChatGPT pour la synthèse
            response = await chatgpt.send([
                ChatMessage(role="system", name="moderator", content="Tu es un analyseur de discussions techniques."),
                ChatMessage(role="user", name="moderator", content=synthesis_prompt)
            ], max_tokens=1000, timeout=30, verbose=False)
            
            # Extraction du JSON avec parsing sécurisé
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = safe_json_parse(json_match.group(0))
                if data is None:
                    return None
                
                if data.get("feasible", False):
                    proposal_id = f"prop_{len(self.proposals) + 1}"
                    return Proposal(
                        id=proposal_id,
                        title=data["title"],
                        description=data["description"],
                        proposed_code=data["proposed_code"],
                        proposer="Consensus Collaboratif",
                        status=ProposalStatus.PROPOSED,
                        votes=[],
                        discussion_messages=messages
                    )
            
        except Exception as e:
            print(f"❌ Erreur extraction: {e}")
        
        return None
    
    async def conduct_voting(self, proposal: Proposal) -> bool:
        """Conduit un vote formel sur une proposition"""
        print(f"\n🗳️ === VOTE SUR: {proposal.title} ===")
        proposal.status = ProposalStatus.VOTING
        
        voting_prompt = f"""
VOTE FORMEL SUR PROPOSITION:

TITRE: {proposal.title}
DESCRIPTION: {proposal.description}

CODE PROPOSÉ:
```python
{proposal.proposed_code}
```

VOTRE MISSION: Votez sur cette proposition.

Répondez EXACTEMENT au format JSON:
{{
    "vote": "approve|reject|abstain|propose_modification",
    "reasoning": "Votre justification détaillée",
    "modification": "Si propose_modification, décrivez la modification"
}}

Soyez précis et constructifs dans votre analyse !
"""
        
        # Chaque IA vote
        for agent in self.agents:
            try:
                print(f"\n--- Vote de {agent.name} ---")
                response = await agent.send([
                    ChatMessage(role="system", name="moderator", 
                               content="Vous votez sur une proposition d'amélioration. Soyez objectifs."),
                    ChatMessage(role="user", name="moderator", content=voting_prompt)
                ], max_tokens=200, timeout=30, verbose=False)
                
                # Extraction du vote
                vote = self._extract_vote(response, agent.name)
                if vote:
                    proposal.votes.append(vote)
                    print(f"✅ {agent.name}: {vote.vote_type.value} - {vote.reasoning[:100]}...")
                
            except Exception as e:
                print(f"❌ Erreur vote {agent.name}: {e}")
        
        # Analyse des résultats
        approval_rate = proposal.get_approval_rate()
        has_consensus = proposal.has_consensus()
        
        print(f"\n📊 RÉSULTATS DU VOTE:")
        print(f"Taux d'approbation: {approval_rate:.1%}")
        print(f"Consensus atteint: {'✅ OUI' if has_consensus else '❌ NON'}")
        
        if has_consensus:
            proposal.status = ProposalStatus.APPROVED
        else:
            proposal.status = ProposalStatus.REJECTED
            
        return has_consensus
    
    async def implement_approved_proposal(self, proposal: Proposal) -> bool:
        """Implémente une proposition approuvée"""
        if proposal.status != ProposalStatus.APPROVED:
            print("❌ Proposition non approuvée")
            return False
            
        print(f"\n🚀 === IMPLÉMENTATION: {proposal.title} ===")
        
        try:
            # Crée un fichier temporaire avec le nouveau code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(proposal.proposed_code)
                temp_file = f.name
            
            # Test de syntaxe
            result = subprocess.run(['python', '-m', 'py_compile', temp_file], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Erreur de syntaxe: {result.stderr}")
                os.unlink(temp_file)
                return False
            
            # Backup de l'ancien fichier
            backup_file = f"tri_ai_orchestrator_backup_{len(self.proposals)}.py"
            if os.path.exists("tri_ai_orchestrator.py"):
                os.rename("tri_ai_orchestrator.py", backup_file)
                print(f"📦 Backup créé: {backup_file}")
            
            # Déploie le nouveau code
            os.rename(temp_file, "tri_ai_orchestrator.py")
            print("✅ Nouveau code déployé!")
            
            proposal.status = ProposalStatus.IMPLEMENTED
            
            # Test rapide du nouveau code
            test_result = subprocess.run(['python', 'tri_ai_orchestrator.py', '--help'], 
                                       capture_output=True, text=True, timeout=10)
            
            if test_result.returncode == 0:
                print("✅ Test de base réussi!")
                return True
            else:
                print(f"⚠️ Test de base échoué: {test_result.stderr}")
                # Rollback
                os.rename("tri_ai_orchestrator.py", f"tri_ai_orchestrator_failed_{len(self.proposals)}.py")
                os.rename(backup_file, "tri_ai_orchestrator.py")
                print("🔄 Rollback effectué")
                return False
                
        except Exception as e:
            print(f"❌ Erreur implémentation: {e}")
            return False
    
    def _extract_vote(self, response: str, agent_name: str) -> Optional[Vote]:
        """Extrait un vote de la réponse d'une IA"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = safe_json_parse(json_match.group(0))
                if data is None:
                    return None
                
                vote_type_str = data.get("vote", "abstain")
                vote_type = VoteType(vote_type_str)
                
                return Vote(
                    agent_name=agent_name,
                    vote_type=vote_type,
                    reasoning=data.get("reasoning", "Pas de justification"),
                    modification_suggestion=data.get("modification")
                )
        except Exception as e:
            print(f"❌ Erreur extraction vote: {e}")
        
        return None
    
    def _detect_mention(self, text: str) -> Optional[str]:
        """Détecte les mentions d'agents dans le texte"""
        mentions = re.findall(r'@(\w+)', text)
        agent_names = [agent.name for agent in self.agents]
        
        for mention in mentions:
            if mention in agent_names:
                return mention
        return None
    
    def _get_agent_index(self, agent_name: str) -> int:
        """Récupère l'index d'un agent par son nom"""
        for i, agent in enumerate(self.agents):
            if agent.name == agent_name:
                return i
        return 0
    
    def _format_messages_for_analysis(self, messages: List[ChatMessage]) -> str:
        """Formate les messages pour l'analyse"""
        formatted = []
        for msg in messages:
            if msg.role != "system":
                formatted.append(f"{msg.name}: {msg.content}")
        return "\n".join(formatted)

async def main():
    """Fonction principale pour tester le système de consensus"""
    print("🤖 === SYSTÈME DE CONSENSUS INTER-IA ===\n")
    
    # Initialisation des agents
    agents = [
        OpenAIClient("ChatGPT", "gpt-4o"),
        AnthropicClient("Claude", "claude-3-5-sonnet-20241022"),
        GeminiClient("Gemini", "gemini-1.5-pro"),
    ]
    
    orchestrator = ConsensusOrchestrator(agents)
    
    # Test du système
    topic = "Améliorer la gestion des erreurs et ajouter un système de retry intelligent"
    
    # Phase 1: Discussion et proposition
    proposal = await orchestrator.initiate_proposal_discussion(topic)
    
    if proposal:
        # Phase 2: Vote
        consensus_reached = await orchestrator.conduct_voting(proposal)
        
        if consensus_reached:
            # Phase 3: Implémentation
            print("\n🎉 Consensus atteint ! Implémentation en cours...")
            success = await orchestrator.implement_approved_proposal(proposal)
            
            if success:
                print("🚀 SUCCÈS! Les IA ont collaboré pour améliorer leur propre code!")
            else:
                print("❌ Échec de l'implémentation")
        else:
            print("❌ Pas de consensus. Proposition rejetée.")
            
            # Analyse des modifications suggérées
            modifications = proposal.get_modifications_suggested()
            if modifications:
                print("\n💡 Modifications suggérées:")
                for i, mod in enumerate(modifications, 1):
                    print(f"{i}. {mod}")
    else:
        print("❌ Aucune proposition viable n'a émergé de la discussion")

if __name__ == "__main__":
    asyncio.run(main())
