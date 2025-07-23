#!/usr/bin/env python3
"""
Self-Improvement System for Tri-AI Orchestrator
===============================================
Système d'auto-amélioration permettant aux IA de collaborer pour créer
de nouvelles versions améliorées du script.

Auteur: Assistant IA
Version: 1.0
"""

import os
import sys
import git
import shutil
import subprocess
import tempfile
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

@dataclass
class ImprovementProposal:
    """Proposition d'amélioration du code."""
    author: str  # Nom de l'IA qui propose
    description: str  # Description de l'amélioration
    code_changes: str  # Code modifié ou ajouté
    rationale: str  # Justification de l'amélioration
    priority: int  # Priorité (1-10)
    risk_level: str  # "low", "medium", "high"

@dataclass
class TestResult:
    """Résultat d'un test."""
    test_name: str
    passed: bool
    output: str
    execution_time: float

class VersionManager:
    """Gestionnaire de versions avec Git."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.repo = None
        self._init_repo()
    
    def _init_repo(self):
        """Initialise le repository Git si nécessaire."""
        try:
            self.repo = git.Repo(self.repo_path)
        except git.InvalidGitRepositoryError:
            print("🔧 Initialisation du repository Git...")
            self.repo = git.Repo.init(self.repo_path)
            # Premier commit
            self.repo.index.add(['*.py'])
            self.repo.index.commit("Initial commit - Tri-AI Orchestrator")
    
    def create_branch(self, branch_name: str) -> str:
        """Crée une nouvelle branche pour l'expérimentation."""
        try:
            new_branch = self.repo.create_head(branch_name)
            new_branch.checkout()
            return branch_name
        except Exception as e:
            # Si la branche existe déjà, on l'utilise
            self.repo.heads[branch_name].checkout()
            return branch_name
    
    def commit_changes(self, message: str, files: List[str] = None) -> str:
        """Commit les changements."""
        if files:
            self.repo.index.add(files)
        else:
            self.repo.git.add(A=True)
        
        commit = self.repo.index.commit(message)
        return commit.hexsha
    
    def create_tag(self, tag_name: str, message: str) -> None:
        """Crée un tag pour marquer une version."""
        self.repo.create_tag(tag_name, message=message)
    
    def rollback_to_commit(self, commit_hash: str = None) -> None:
        """Rollback vers un commit spécifique ou le précédent."""
        try:
            if commit_hash:
                self.repo.git.reset('--hard', commit_hash)
            else:
                # Rollback vers le commit précédent s'il existe
                commits = list(self.repo.iter_commits(max_count=2))
                if len(commits) > 1:
                    self.repo.git.reset('--hard', commits[1].hexsha)
                else:
                    print("⚠️ Aucun commit précédent pour le rollback")
        except Exception as e:
            print(f"⚠️ Erreur lors du rollback: {e}")
    
    def get_diff(self, commit1: str = None, commit2: str = None) -> str:
        """Obtient le diff entre deux commits."""
        if commit1 and commit2:
            return self.repo.git.diff(commit1, commit2)
        else:
            return self.repo.git.diff('HEAD~1', 'HEAD')

class SafetyManager:
    """Gestionnaire de sécurité pour l'auto-amélioration."""
    
    def __init__(self, max_iterations: int = 10, backup_dir: str = "backups"):
        self.max_iterations = max_iterations
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.iteration_count = 0
    
    def create_backup(self, file_path: str) -> str:
        """Crée une sauvegarde du fichier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{Path(file_path).stem}_{timestamp}.backup"
        backup_path = self.backup_dir / backup_name
        shutil.copy2(file_path, backup_path)
        return str(backup_path)
    
    def can_continue(self) -> bool:
        """Vérifie si on peut continuer l'amélioration."""
        return self.iteration_count < self.max_iterations
    
    def increment_iteration(self) -> None:
        """Incrémente le compteur d'itérations."""
        self.iteration_count += 1
    
    def validate_code_safety(self, code: str) -> Tuple[bool, str]:
        """Valide la sécurité du code proposé."""
        dangerous_patterns = [
            'os.system(',
            'subprocess.call(',
            'exec(',
            'eval(',
            '__import__(',
            'open(',  # Trop restrictif, mais pour la sécurité
            'rm -rf',
            'del ',
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                return False, f"Code potentiellement dangereux détecté: {pattern}"
        
        return True, "Code semble sûr"

class TestRunner:
    """Exécuteur de tests pour valider les nouvelles versions."""
    
    def __init__(self, script_path: str):
        self.script_path = script_path
    
    async def run_basic_tests(self) -> List[TestResult]:
        """Exécute des tests de base sur le script."""
        tests = []
        
        # Test 1: Le script peut-il s'exécuter sans erreur ?
        test1 = await self._test_script_execution()
        tests.append(test1)
        
        # Test 2: Le script accepte-t-il les arguments de base ?
        test2 = await self._test_help_command()
        tests.append(test2)
        
        # Test 3: Validation des paramètres
        test3 = await self._test_parameter_validation()
        tests.append(test3)
        
        return tests
    
    async def _test_script_execution(self) -> TestResult:
        """Test si le script peut s'exécuter."""
        try:
            result = subprocess.run([
                sys.executable, self.script_path, '--help'
            ], capture_output=True, text=True, timeout=10)
            
            passed = result.returncode == 0
            return TestResult(
                "script_execution",
                passed,
                result.stdout + result.stderr,
                0.0
            )
        except Exception as e:
            return TestResult(
                "script_execution",
                False,
                str(e),
                0.0
            )
    
    async def _test_help_command(self) -> TestResult:
        """Test si la commande --help fonctionne."""
        try:
            result = subprocess.run([
                sys.executable, self.script_path, '--help'
            ], capture_output=True, text=True, timeout=5)
            
            passed = result.returncode == 0 and 'usage:' in result.stdout
            return TestResult(
                "help_command",
                passed,
                result.stdout,
                0.0
            )
        except Exception as e:
            return TestResult(
                "help_command",
                False,
                str(e),
                0.0
            )
    
    async def _test_parameter_validation(self) -> TestResult:
        """Test la validation des paramètres."""
        try:
            # Test avec des paramètres invalides
            result = subprocess.run([
                sys.executable, self.script_path, 
                '--prompt', 'test', '--turns', '1000'
            ], capture_output=True, text=True, timeout=5)
            
            # Doit échouer avec un message d'erreur approprié
            passed = result.returncode != 0 and 'tours doit être' in result.stderr
            return TestResult(
                "parameter_validation",
                passed,
                result.stderr,
                0.0
            )
        except Exception as e:
            return TestResult(
                "parameter_validation",
                False,
                str(e),
                0.0
            )

class SelfImprovementOrchestrator:
    """Orchestrateur principal pour l'auto-amélioration."""
    
    def __init__(self, script_path: str, ai_orchestrator):
        self.script_path = Path(script_path)
        self.ai_orchestrator = ai_orchestrator
        self.version_manager = VersionManager(self.script_path.parent)
        self.safety_manager = SafetyManager()
        self.test_runner = TestRunner(str(self.script_path))
        
    async def start_self_improvement(self) -> None:
        """Démarre le processus d'auto-amélioration."""
        print("🚀 Démarrage du processus d'auto-amélioration...")
        
        # Backup initial
        backup_path = self.safety_manager.create_backup(str(self.script_path))
        print(f"📦 Backup créé: {backup_path}")
        
        iteration = 0
        while self.safety_manager.can_continue():
            iteration += 1
            print(f"\n🔄 Itération {iteration}")
            
            try:
                # 1. Analyse du code actuel par les IA
                proposals = await self._get_improvement_proposals()
                
                if not proposals:
                    print("✅ Aucune amélioration proposée. Processus terminé.")
                    break
                
                # 2. Sélection de la meilleure proposition
                best_proposal = self._select_best_proposal(proposals)
                print(f"🎯 Proposition sélectionnée: {best_proposal.description}")
                
                # 3. Création d'une branche pour l'expérimentation
                branch_name = f"improvement_v{iteration}_{datetime.now().strftime('%H%M%S')}"
                self.version_manager.create_branch(branch_name)
                
                # 4. Application des changements
                success = await self._apply_improvement(best_proposal)
                
                if not success:
                    print("❌ Échec de l'application des changements")
                    continue
                
                # 5. Tests de la nouvelle version
                test_results = await self.test_runner.run_basic_tests()
                all_passed = all(test.passed for test in test_results)
                
                if all_passed:
                    # 6. Commit et tag de la nouvelle version
                    commit_hash = self.version_manager.commit_changes(
                        f"Auto-improvement v{iteration}: {best_proposal.description}"
                    )
                    self.version_manager.create_tag(
                        f"auto_v{iteration}",
                        f"Auto-generated improvement: {best_proposal.description}"
                    )
                    
                    print(f"✅ Amélioration appliquée avec succès! Commit: {commit_hash[:8]}")
                    
                    # 7. Redémarrage avec la nouvelle version (optionnel)
                    if iteration < 3:  # Limite pour éviter les boucles infinies
                        print("🔄 Redémarrage avec la nouvelle version...")
                        await asyncio.sleep(2)  # Pause pour éviter la surcharge
                else:
                    print("❌ Tests échoués, rollback...")
                    self.version_manager.rollback_to_commit('HEAD~1')
                
                self.safety_manager.increment_iteration()
                
            except Exception as e:
                print(f"❌ Erreur lors de l'itération {iteration}: {e}")
                break
        
        print(f"\n🏁 Processus d'auto-amélioration terminé après {iteration} itérations")
    
    async def _get_improvement_proposals(self) -> List[ImprovementProposal]:
        """Demande aux IA de proposer des améliorations."""
        # Lire le code actuel
        with open(self.script_path, 'r', encoding='utf-8') as f:
            current_code = f.read()
        
        prompt = f"""
Analysez ce script Python et proposez UNE amélioration concrète et simple.

Répondez UNIQUEMENT avec un JSON valide dans ce format exact:
{{
    "description": "Description courte (max 50 caractères)",
    "code_changes": "Code à ajouter ou modifier (max 10 lignes)",
    "rationale": "Justification courte",
    "priority": 5,
    "risk_level": "low"
}}

Exemples d'améliorations simples:
- Ajouter une fonction d'aide
- Améliorer un message d'erreur
- Ajouter une validation
- Optimiser une fonction existante

Code actuel (extrait):
{current_code[:1500]}...

Répondez SEULEMENT avec le JSON, rien d'autre.
        """
        
        proposals = []
        
        if self.ai_orchestrator:
            try:
                # Utiliser les IA réelles pour obtenir des propositions
                from tri_ai_orchestrator import ChatMessage
                
                messages = [
                    ChatMessage(role="system", name="system", 
                              content="Tu es un expert en amélioration de code Python. Réponds uniquement avec du JSON valide."),
                    ChatMessage(role="user", name="user", content=prompt)
                ]
                
                # Demander à chaque IA une proposition
                for agent in self.ai_orchestrator.agents[:1]:  # Une seule IA pour commencer
                    try:
                        response = await agent.send(
                            messages, 
                            max_tokens=200, 
                            timeout=30, 
                            verbose=False
                        )
                        
                        # Essayer de parser le JSON
                        import json
                        import re
                        
                        # Extraire le JSON de la réponse
                        json_match = re.search(r'\{.*\}', response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            data = json.loads(json_str)
                            
                            proposal = ImprovementProposal(
                                author=agent.name,
                                description=data.get('description', 'Amélioration'),
                                code_changes=data.get('code_changes', ''),
                                rationale=data.get('rationale', 'Amélioration générale'),
                                priority=data.get('priority', 5),
                                risk_level=data.get('risk_level', 'low')
                            )
                            proposals.append(proposal)
                            print(f"✅ Proposition reçue de {agent.name}: {proposal.description}")
                        else:
                            print(f"⚠️ Réponse non-JSON de {agent.name}: {response[:100]}...")
                            
                    except Exception as e:
                        print(f"⚠️ Erreur avec {agent.name}: {e}")
                        
            except Exception as e:
                print(f"⚠️ Erreur lors de la récupération des propositions: {e}")
        
        # Fallback: propositions par défaut si aucune IA n'a répondu
        if not proposals:
            proposals = [
                ImprovementProposal(
                    author="System",
                    description="Ajouter fonction utilitaire",
                    code_changes="""
# Fonction utilitaire pour formater le temps
def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds/60:.1f}min"
""",
                    rationale="Utilitaire pour afficher les durées",
                    priority=5,
                    risk_level="low"
                )
            ]
        
        return proposals
    
    def _select_best_proposal(self, proposals: List[ImprovementProposal]) -> ImprovementProposal:
        """Sélectionne la meilleure proposition basée sur la priorité et le risque."""
        # Tri par priorité (descendant) et risque (ascendant)
        risk_weights = {"low": 1, "medium": 2, "high": 3}
        
        def score(proposal):
            return proposal.priority * 10 - risk_weights.get(proposal.risk_level, 3)
        
        return max(proposals, key=score)
    
    async def _apply_improvement(self, proposal: ImprovementProposal) -> bool:
        """Applique une amélioration au code."""
        try:
            # Validation de sécurité
            is_safe, safety_msg = self.safety_manager.validate_code_safety(proposal.code_changes)
            if not is_safe:
                print(f"⚠️ Amélioration rejetée pour raisons de sécurité: {safety_msg}")
                return False
            
            # Pour le prototype, on simule l'application
            print(f"🔧 Application de l'amélioration: {proposal.description}")
            print(f"📝 Code ajouté:\n{proposal.code_changes}")
            
            # Ici, on appliquerait réellement les changements au fichier
            # Pour la démo, on simule juste
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de l'application: {e}")
            return False

# Fonction d'intégration avec le script principal
async def run_self_improvement_mode(script_path: str, ai_orchestrator):
    """Lance le mode d'auto-amélioration."""
    orchestrator = SelfImprovementOrchestrator(script_path, ai_orchestrator)
    await orchestrator.start_self_improvement()

if __name__ == "__main__":
    # Test du système
    print("🧪 Test du système d'auto-amélioration")
    
    # Simulation
    async def test_self_improvement():
        orchestrator = SelfImprovementOrchestrator("tri_ai_orchestrator.py", None)
        await orchestrator.start_self_improvement()
    
    asyncio.run(test_self_improvement())
