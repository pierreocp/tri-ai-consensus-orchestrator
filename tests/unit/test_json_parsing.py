import pytest
import json
from consensus_system import safe_json_parse

class TestSafeJsonParse:
    """Tests for the safe JSON parsing utility."""
    
    def test_valid_json_parsing(self):
        """Test parsing of valid JSON strings."""
        valid_json = '{"key": "value", "number": 42}'
        result = safe_json_parse(valid_json)
        
        assert result is not None
        assert result["key"] == "value"
        assert result["number"] == 42
    
    def test_json_with_whitespace(self):
        """Test parsing JSON with extra whitespace and newlines."""
        json_with_whitespace = '''
        {
            "key": "value",
            "array": [1, 2, 3]
        }
        '''
        result = safe_json_parse(json_with_whitespace)
        
        assert result is not None
        assert result["key"] == "value"
        assert result["array"] == [1, 2, 3]
    
    def test_invalid_json_returns_none(self):
        """Test that invalid JSON returns None."""
        invalid_json = '{"key": "value", "unclosed": '
        result = safe_json_parse(invalid_json)
        
        assert result is None
    
    def test_odd_quote_count_returns_none(self):
        """Test that JSON with odd number of quotes returns None."""
        odd_quotes = '{"key": "value"}'  # This should be valid
        result = safe_json_parse(odd_quotes)
        assert result is not None
        
        odd_quotes = '{"key": "value'  # Missing closing quote
        result = safe_json_parse(odd_quotes)
        assert result is None
    
    def test_empty_string_returns_none(self):
        """Test that empty string returns None."""
        result = safe_json_parse("")
        assert result is None
    
    def test_non_json_string_returns_none(self):
        """Test that non-JSON strings return None."""
        result = safe_json_parse("This is not JSON")
        assert result is None
    
    def test_json_with_special_characters(self):
        """Test JSON with special characters gets cleaned."""
        json_with_tabs = '{\t"key":\n"value"\r}'
        result = safe_json_parse(json_with_tabs)
        
        assert result is not None
        assert result["key"] == "value"
