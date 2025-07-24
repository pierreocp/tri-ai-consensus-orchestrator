import pytest
from self_improvement import SafetyManager

class TestSecurityValidation:
    """Tests for code security validation."""
    
    @pytest.fixture
    def safety_manager(self):
        """Fixture providing a SafetyManager instance."""
        return SafetyManager()
    
    def test_safe_code_passes_validation(self, safety_manager):
        """Test that safe code passes validation."""
        safe_code = '''
def add_numbers(a, b):
    return a + b

result = add_numbers(1, 2)
print(f"Result: {result}")
'''
        is_safe, message = safety_manager.validate_code_safety(safe_code)
        
        assert is_safe is True
        assert "semble sûr" in message
    
    def test_dangerous_os_system_rejected(self, safety_manager):
        """Test that os.system calls are rejected."""
        dangerous_code = 'os.system("rm -rf /")'
        is_safe, message = safety_manager.validate_code_safety(dangerous_code)
        
        assert is_safe is False
        assert "os.system(" in message
    
    def test_dangerous_subprocess_call_rejected(self, safety_manager):
        """Test that subprocess.call is rejected."""
        dangerous_code = 'subprocess.call(["rm", "-rf", "/"])'
        is_safe, message = safety_manager.validate_code_safety(dangerous_code)
        
        assert is_safe is False
        assert "subprocess.call(" in message
    
    def test_dangerous_exec_rejected(self, safety_manager):
        """Test that exec() calls are rejected."""
        dangerous_code = 'exec("malicious_code")'
        is_safe, message = safety_manager.validate_code_safety(dangerous_code)
        
        assert is_safe is False
        assert "exec(" in message
    
    def test_dangerous_eval_rejected(self, safety_manager):
        """Test that eval() calls are rejected."""
        dangerous_code = 'result = eval(user_input)'
        is_safe, message = safety_manager.validate_code_safety(dangerous_code)
        
        assert is_safe is False
        assert "eval(" in message
    
    def test_dangerous_import_rejected(self, safety_manager):
        """Test that __import__ calls are rejected."""
        dangerous_code = '__import__("os").system("ls")'
        is_safe, message = safety_manager.validate_code_safety(dangerous_code)
        
        assert is_safe is False
        assert "__import__(" in message
    
    def test_dangerous_file_operations_rejected(self, safety_manager):
        """Test that open() calls are rejected (conservative approach)."""
        dangerous_code = 'with open("/etc/passwd", "r") as f: content = f.read()'
        is_safe, message = safety_manager.validate_code_safety(dangerous_code)
        
        assert is_safe is False
        assert "open(" in message
    
    def test_dangerous_rm_command_rejected(self, safety_manager):
        """Test that rm -rf commands are rejected."""
        dangerous_code = 'command = "rm -rf /important/data"'
        is_safe, message = safety_manager.validate_code_safety(dangerous_code)
        
        assert is_safe is False
        assert "rm -rf" in message
    
    def test_dangerous_del_statement_rejected(self, safety_manager):
        """Test that del statements are rejected."""
        dangerous_code = 'del important_variable'
        is_safe, message = safety_manager.validate_code_safety(dangerous_code)
        
        assert is_safe is False
        assert "del " in message
    
    def test_multiple_dangerous_patterns(self, safety_manager):
        """Test code with multiple dangerous patterns."""
        dangerous_code = '''
import os
os.system("rm -rf /")
exec("malicious_code")
'''
        is_safe, message = safety_manager.validate_code_safety(dangerous_code)
        
        assert is_safe is False
        assert any(pattern in message for pattern in ["os.system(", "exec("])
