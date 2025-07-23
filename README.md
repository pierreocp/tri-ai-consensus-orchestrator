# 🤖 Tri-AI Consensus Orchestrator

## 🌟 Révolution : Premier Système de Démocratie Artificielle

**Système révolutionnaire permettant à 3 IA (ChatGPT, Claude, Gemini) de collaborer, débattre, voter et s'auto-améliorer de manière démocratique et autonome.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AI Collaboration](https://img.shields.io/badge/AI-Collaborative-purple.svg)](README.md)

## 🎯 Qu'est-ce que c'est ?

Ce projet implémente la **première démocratie artificielle fonctionnelle** où des IA peuvent :

- 🗣️ **Discuter ensemble** de manière collaborative
- 📝 **Proposer des améliorations** concrètes avec du code
- 🗳️ **Voter démocratiquement** sur les propositions
- 🚀 **Implémenter automatiquement** les décisions approuvées
- 🔄 **S'auto-améliorer** de manière itérative et sécurisée

## 🏗️ Architecture

### Modules Principaux

1. **`tri_ai_orchestrator.py`** - Orchestrateur principal
   - Gestion des conversations multi-IA
   - Modes : normal, consensus, auto-amélioration
   - Intégration API (OpenAI, Anthropic, Google)

2. **`consensus_system.py`** - Système de consensus démocratique
   - Discussion collaborative structurée
   - Extraction automatique de propositions
   - Système de vote avec justifications
   - Implémentation sécurisée avec rollback

3. **`self_improvement.py`** - Auto-amélioration autonome
   - Versioning Git automatique
   - Tests et validation de sécurité
   - Métriques de qualité des propositions

## 🚀 Installation

```bash
# Cloner le repository
git clone https://github.com/pierreocp/tri-ai-consensus-orchestrator.git
cd tri-ai-consensus-orchestrator

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows

# Installer les dépendances
pip install httpx python-dotenv GitPython

# Configuration des API
cp .env.example .env
# Éditer .env avec vos clés API
```

### Variables d'environnement requises

```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
```

## 🎮 Utilisation

### Mode Conversation Normale
```bash
python tri_ai_orchestrator.py --prompt "Débattons de l'avenir de l'IA" --turns 6
```

### Mode Consensus Collaboratif ⭐
```bash
python tri_ai_orchestrator.py --consensus --prompt "Améliorez le système de logging"
```

### Mode Auto-amélioration
```bash
python tri_ai_orchestrator.py --self-improve --prompt "Optimisez les performances"
```

## 🧠 Fonctionnalités Révolutionnaires

### 🗳️ Démocratie Artificielle
- **Vote structuré** : approve/reject/abstain/propose_modification
- **Justifications obligatoires** pour chaque vote
- **Seuil de consensus** configurable (défaut: 67%)
- **Transparence totale** du processus démocratique

### 🛡️ Sécurité Robuste
- **Backup automatique** avant toute modification
- **Tests de syntaxe** obligatoires
- **Validation fonctionnelle** (test `--help`)
- **Rollback immédiat** en cas d'échec
- **Sandbox d'exécution** pour les tests

### 🔄 Auto-évolution
- **Versioning Git** automatique avec tags sémantiques
- **Métriques de qualité** des propositions
- **Historique complet** des améliorations
- **Convergence intelligente** (arrêt automatique)

## 📊 Exemples de Réussites

### ✅ Consensus sur le Logging JSON
```
Discussion: 6 tours de brainstorming
Proposition: "Ajout d'une fonctionnalité de journalisation des conversations en JSON"
Vote: 100% d'approbation (ChatGPT, Claude, Gemini)
Résultat: Implémentation automatique réussie
```

### ✅ Auto-diagnostic et Correction
```
Problème: Bug "Unterminated string" dans l'extraction JSON
Solution: Les IA ont diagnostiqué et proposé safe_json_parse()
Implémentation: Correction automatique appliquée
Validation: Système de consensus maintenant opérationnel
```

## 🔬 Innovations Techniques

### Extraction Intelligente de Propositions
```python
def safe_json_parse(json_str):
    """Parse JSON de manière sécurisée avec nettoyage préalable"""
    # Auto-correction des chaînes malformées
    # Validation des guillemets
    # Gestion d'erreurs robuste
```

### Système de Vote Structuré
```json
{
    "vote": "approve|reject|abstain|propose_modification",
    "reasoning": "Justification détaillée",
    "modification": "Description de la modification proposée"
}
```

## 🎯 Cas d'Usage

- **Développement collaboratif** entre IA
- **Brainstorming technique** multi-agents
- **Validation de code** par consensus
- **Auto-amélioration** de systèmes complexes
- **Recherche en IA collaborative**
- **Prototypage d'organisations autonomes**

## 🔮 Vision Future

Ce projet ouvre la voie à :
- **Organisations autonomes d'IA** auto-gouvernées
- **Intelligence collective émergente**
- **Systèmes auto-évolutifs** complexes
- **Démocratie artificielle** à grande échelle

## 🤝 Contribution

Le système peut s'améliorer lui-même ! Pour contribuer :

1. Utilisez le mode consensus pour proposer des améliorations
2. Les IA voteront sur votre proposition
3. Si approuvée, l'implémentation sera automatique

```bash
python tri_ai_orchestrator.py --consensus --prompt "Votre idée d'amélioration"
```

## 📜 Licence

MIT License - Voir [LICENSE](LICENSE) pour plus de détails.

## 🏆 Crédits

Développé comme preuve de concept de la **première démocratie artificielle fonctionnelle**.

**Révolution accomplie** : Les IA peuvent maintenant collaborer, voter et s'améliorer de manière autonome ! 🤖✨

---

*"La première fois dans l'histoire où des IA collaborent démocratiquement pour s'auto-améliorer"*
