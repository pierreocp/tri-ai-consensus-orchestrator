import pytest
from consensus_system import Proposal, Vote, VoteType, ProposalStatus, ChatMessage

class TestProposalLogic:
    """Tests for proposal voting logic and calculations."""
    
    @pytest.fixture
    def sample_proposal(self):
        """Fixture providing a sample proposal."""
        return Proposal(
            id="test_prop_1",
            title="Test Proposal",
            description="A test proposal for unit testing",
            proposed_code="print('Hello, World!')",
            proposer="TestAgent",
            status=ProposalStatus.PROPOSED,
            votes=[],
            discussion_messages=[]
        )
    
    def test_empty_proposal_has_zero_approval_rate(self, sample_proposal):
        """Test that proposal with no votes has 0% approval rate."""
        approval_rate = sample_proposal.get_approval_rate()
        assert approval_rate == 0.0
    
    def test_single_approval_vote(self, sample_proposal):
        """Test proposal with single approval vote."""
        vote = Vote(
            agent_name="Agent1",
            vote_type=VoteType.APPROVE,
            reasoning="Good idea"
        )
        sample_proposal.votes.append(vote)
        
        approval_rate = sample_proposal.get_approval_rate()
        assert approval_rate == 1.0
    
    def test_single_rejection_vote(self, sample_proposal):
        """Test proposal with single rejection vote."""
        vote = Vote(
            agent_name="Agent1",
            vote_type=VoteType.REJECT,
            reasoning="Bad idea"
        )
        sample_proposal.votes.append(vote)
        
        approval_rate = sample_proposal.get_approval_rate()
        assert approval_rate == 0.0
    
    def test_mixed_votes_approval_rate(self, sample_proposal):
        """Test proposal with mixed approval and rejection votes."""
        votes = [
            Vote("Agent1", VoteType.APPROVE, "Good"),
            Vote("Agent2", VoteType.APPROVE, "Great"),
            Vote("Agent3", VoteType.REJECT, "Bad")
        ]
        sample_proposal.votes.extend(votes)
        
        approval_rate = sample_proposal.get_approval_rate()
        assert approval_rate == 2/3  # 2 approvals out of 3 total votes
    
    def test_abstain_votes_ignored_in_calculation(self, sample_proposal):
        """Test that abstain votes are ignored in approval rate calculation."""
        votes = [
            Vote("Agent1", VoteType.APPROVE, "Good"),
            Vote("Agent2", VoteType.ABSTAIN, "Neutral"),
            Vote("Agent3", VoteType.REJECT, "Bad")
        ]
        sample_proposal.votes.extend(votes)
        
        approval_rate = sample_proposal.get_approval_rate()
        assert approval_rate == 0.5  # 1 approval out of 2 non-abstain votes
    
    def test_only_abstain_votes_returns_zero(self, sample_proposal):
        """Test that proposal with only abstain votes returns 0% approval."""
        votes = [
            Vote("Agent1", VoteType.ABSTAIN, "Neutral"),
            Vote("Agent2", VoteType.ABSTAIN, "Unsure")
        ]
        sample_proposal.votes.extend(votes)
        
        approval_rate = sample_proposal.get_approval_rate()
        assert approval_rate == 0.0
    
    def test_consensus_threshold_default(self, sample_proposal):
        """Test that default consensus threshold is 67%."""
        assert sample_proposal.consensus_threshold == 0.67
    
    def test_has_consensus_with_sufficient_approval(self, sample_proposal):
        """Test consensus detection with sufficient approval rate."""
        votes = [
            Vote("Agent1", VoteType.APPROVE, "Good"),
            Vote("Agent2", VoteType.APPROVE, "Great"),
            Vote("Agent3", VoteType.REJECT, "Bad")
        ]
        sample_proposal.votes.extend(votes)
        
        assert sample_proposal.has_consensus() is False
        
        sample_proposal.votes.append(Vote("Agent4", VoteType.APPROVE, "Excellent"))
        assert sample_proposal.has_consensus() is True
    
    def test_has_consensus_with_insufficient_approval(self, sample_proposal):
        """Test consensus detection with insufficient approval rate."""
        votes = [
            Vote("Agent1", VoteType.APPROVE, "Good"),
            Vote("Agent2", VoteType.REJECT, "Bad"),
            Vote("Agent3", VoteType.REJECT, "Terrible")
        ]
        sample_proposal.votes.extend(votes)
        
        assert sample_proposal.has_consensus() is False
    
    def test_get_modifications_suggested(self, sample_proposal):
        """Test extraction of modification suggestions."""
        votes = [
            Vote("Agent1", VoteType.APPROVE, "Good"),
            Vote("Agent2", VoteType.PROPOSE_MODIFICATION, "Needs change", "Add error handling"),
            Vote("Agent3", VoteType.PROPOSE_MODIFICATION, "Another change", "Improve performance"),
            Vote("Agent4", VoteType.REJECT, "Bad")
        ]
        sample_proposal.votes.extend(votes)
        
        modifications = sample_proposal.get_modifications_suggested()
        assert len(modifications) == 2
        assert "Add error handling" in modifications
        assert "Improve performance" in modifications
    
    def test_get_modifications_suggested_empty(self, sample_proposal):
        """Test modification suggestions when none exist."""
        votes = [
            Vote("Agent1", VoteType.APPROVE, "Good"),
            Vote("Agent2", VoteType.REJECT, "Bad")
        ]
        sample_proposal.votes.extend(votes)
        
        modifications = sample_proposal.get_modifications_suggested()
        assert len(modifications) == 0
    
    def test_custom_consensus_threshold(self):
        """Test proposal with custom consensus threshold."""
        proposal = Proposal(
            id="custom_prop",
            title="Custom Threshold Proposal",
            description="Test with custom threshold",
            proposed_code="pass",
            proposer="TestAgent",
            status=ProposalStatus.PROPOSED,
            votes=[],
            discussion_messages=[],
            consensus_threshold=0.8  # 80% threshold
        )
        
        votes = [
            Vote("Agent1", VoteType.APPROVE, "Good"),
            Vote("Agent2", VoteType.APPROVE, "Great"),
            Vote("Agent3", VoteType.REJECT, "Bad")
        ]
        proposal.votes.extend(votes)
        
        assert proposal.has_consensus() is False
        
        proposal.votes.extend([
            Vote("Agent4", VoteType.APPROVE, "Excellent"),
            Vote("Agent5", VoteType.APPROVE, "Perfect")
        ])
        
        assert proposal.has_consensus() is True
