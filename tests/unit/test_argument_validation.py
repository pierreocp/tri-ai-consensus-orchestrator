import pytest
import argparse
from unittest.mock import patch
from tri_ai_orchestrator import validate_args, check_api_keys, parse_args

class TestArgumentValidation:
    """Tests for command line argument validation."""
    
    def test_valid_arguments_pass_validation(self):
        """Test that valid arguments pass validation."""
        args = argparse.Namespace(
            prompt="Test prompt",
            turns=5,
            max_tokens=512,
            timeout=30
        )
        
        validate_args(args)
    
    def test_turns_below_minimum_raises_error(self):
        """Test that turns below minimum raises ValueError."""
        args = argparse.Namespace(
            prompt="Test prompt",
            turns=0,  # Below minimum of 1
            max_tokens=512,
            timeout=30
        )
        
        with pytest.raises(ValueError, match="Le nombre de tours doit être entre"):
            validate_args(args)
    
    def test_turns_above_maximum_raises_error(self):
        """Test that turns above maximum raises ValueError."""
        args = argparse.Namespace(
            prompt="Test prompt",
            turns=100,  # Above maximum of 50
            max_tokens=512,
            timeout=30
        )
        
        with pytest.raises(ValueError, match="Le nombre de tours doit être entre"):
            validate_args(args)
    
    def test_max_tokens_zero_raises_error(self):
        """Test that zero max_tokens raises ValueError."""
        args = argparse.Namespace(
            prompt="Test prompt",
            turns=5,
            max_tokens=0,
            timeout=30
        )
        
        with pytest.raises(ValueError, match="Le nombre de tokens doit être entre"):
            validate_args(args)
    
    def test_max_tokens_above_limit_raises_error(self):
        """Test that max_tokens above limit raises ValueError."""
        args = argparse.Namespace(
            prompt="Test prompt",
            turns=5,
            max_tokens=5000,  # Above limit of 4096
            timeout=30
        )
        
        with pytest.raises(ValueError, match="Le nombre de tokens doit être entre"):
            validate_args(args)
    
    def test_negative_timeout_raises_error(self):
        """Test that negative timeout raises ValueError."""
        args = argparse.Namespace(
            prompt="Test prompt",
            turns=5,
            max_tokens=512,
            timeout=-1
        )
        
        with pytest.raises(ValueError, match="Le timeout doit être positif"):
            validate_args(args)
    
    def test_empty_prompt_raises_error(self):
        """Test that empty prompt raises ValueError."""
        args = argparse.Namespace(
            prompt="   ",  # Only whitespace
            turns=5,
            max_tokens=512,
            timeout=30
        )
        
        with pytest.raises(ValueError, match="Le prompt ne peut pas être vide"):
            validate_args(args)

class TestApiKeyValidation:
    """Tests for API key validation."""
    
    @patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test_openai_key',
        'ANTHROPIC_API_KEY': 'test_anthropic_key',
        'GOOGLE_API_KEY': 'test_google_key'
    })
    def test_all_api_keys_present_passes(self):
        """Test that having all API keys passes validation."""
        check_api_keys()
    
    @patch.dict('os.environ', {
        'ANTHROPIC_API_KEY': 'test_anthropic_key',
        'GOOGLE_API_KEY': 'test_google_key'
    }, clear=True)
    def test_missing_openai_key_raises_error(self):
        """Test that missing OpenAI key raises RuntimeError."""
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY pour OpenAI"):
            check_api_keys()
    
    @patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test_openai_key',
        'GOOGLE_API_KEY': 'test_google_key'
    }, clear=True)
    def test_missing_anthropic_key_raises_error(self):
        """Test that missing Anthropic key raises RuntimeError."""
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY pour Anthropic"):
            check_api_keys()
    
    @patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test_openai_key',
        'ANTHROPIC_API_KEY': 'test_anthropic_key'
    }, clear=True)
    def test_missing_google_key_raises_error(self):
        """Test that missing Google key raises RuntimeError."""
        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY pour Google"):
            check_api_keys()
    
    @patch.dict('os.environ', {}, clear=True)
    def test_all_keys_missing_raises_error(self):
        """Test that missing all keys raises RuntimeError with all mentioned."""
        with pytest.raises(RuntimeError) as exc_info:
            check_api_keys()
        
        error_message = str(exc_info.value)
        assert "OPENAI_API_KEY" in error_message
        assert "ANTHROPIC_API_KEY" in error_message
        assert "GOOGLE_API_KEY" in error_message
    
    @patch.dict('os.environ', {
        'OPENAI_API_KEY': '',  # Empty string
        'ANTHROPIC_API_KEY': 'test_anthropic_key',
        'GOOGLE_API_KEY': 'test_google_key'
    })
    def test_empty_api_key_treated_as_missing(self):
        """Test that empty API key is treated as missing."""
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY pour OpenAI"):
            check_api_keys()
