import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from self_improvement import VersionManager

class TestVersionManager:
    """Tests for Git version management operations."""
    
    @pytest.fixture
    def temp_repo_dir(self):
        """Fixture providing a temporary directory for Git operations."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def version_manager(self, temp_repo_dir):
        """Fixture providing a VersionManager instance with temp repo."""
        return VersionManager(temp_repo_dir)
    
    def test_version_manager_initialization(self, version_manager):
        """Test that VersionManager initializes correctly."""
        assert version_manager.repo is not None
        assert version_manager.repo_path.exists()
    
    def test_create_branch_new_branch(self, version_manager):
        """Test creating a new branch."""
        branch_name = "test_branch"
        result = version_manager.create_branch(branch_name)
        
        assert result == branch_name
        assert version_manager.repo.active_branch.name == branch_name
    
    def test_create_branch_existing_branch(self, version_manager):
        """Test creating a branch that already exists."""
        branch_name = "existing_branch"
        
        version_manager.create_branch(branch_name)
        
        result = version_manager.create_branch(branch_name)
        
        assert result == branch_name
        assert version_manager.repo.active_branch.name == branch_name
    
    def test_commit_changes_with_specific_files(self, version_manager, temp_repo_dir):
        """Test committing specific files."""
        test_file = Path(temp_repo_dir) / "test.py"
        test_file.write_text("print('Hello, World!')")
        
        commit_hash = version_manager.commit_changes(
            "Test commit", 
            files=["test.py"]
        )
        
        assert commit_hash is not None
        assert len(commit_hash) == 40  # SHA-1 hash length
    
    def test_commit_changes_all_files(self, version_manager, temp_repo_dir):
        """Test committing all files."""
        test_file1 = Path(temp_repo_dir) / "test1.py"
        test_file2 = Path(temp_repo_dir) / "test2.py"
        test_file1.write_text("print('Test 1')")
        test_file2.write_text("print('Test 2')")
        
        commit_hash = version_manager.commit_changes("Test commit all files")
        
        assert commit_hash is not None
        assert len(commit_hash) == 40
    
    def test_create_tag(self, version_manager, temp_repo_dir):
        """Test creating a Git tag."""
        test_file = Path(temp_repo_dir) / "test.py"
        test_file.write_text("print('Tagged version')")
        version_manager.commit_changes("Initial commit", files=["test.py"])
        
        tag_name = "v1.0.0"
        tag_message = "Version 1.0.0 release"
        version_manager.create_tag(tag_name, tag_message)
        
        tags = [tag.name for tag in version_manager.repo.tags]
        assert tag_name in tags
    
    def test_rollback_to_specific_commit(self, version_manager, temp_repo_dir):
        """Test rolling back to a specific commit."""
        test_file = Path(temp_repo_dir) / "test.py"
        test_file.write_text("Version 1")
        first_commit = version_manager.commit_changes("First commit", files=["test.py"])
        
        test_file.write_text("Version 2")
        version_manager.commit_changes("Second commit", files=["test.py"])
        
        version_manager.rollback_to_commit(first_commit)
        
        current_content = test_file.read_text()
        assert current_content == "Version 1"
    
    def test_rollback_to_previous_commit(self, version_manager, temp_repo_dir):
        """Test rolling back to previous commit without specifying hash."""
        test_file = Path(temp_repo_dir) / "test.py"
        test_file.write_text("Version 1")
        version_manager.commit_changes("First commit", files=["test.py"])
        
        test_file.write_text("Version 2")
        version_manager.commit_changes("Second commit", files=["test.py"])
        
        version_manager.rollback_to_commit()
        
        current_content = test_file.read_text()
        assert current_content == "Version 1"
    
    def test_rollback_with_no_previous_commit(self, version_manager, temp_repo_dir):
        """Test rollback when there's no previous commit."""
        test_file = Path(temp_repo_dir) / "test.py"
        test_file.write_text("Only version")
        version_manager.commit_changes("Only commit", files=["test.py"])
        
        version_manager.rollback_to_commit()
        
        if test_file.exists():
            current_content = test_file.read_text()
            assert current_content == "Only version"
    
    def test_get_diff_between_commits(self, version_manager, temp_repo_dir):
        """Test getting diff between two commits."""
        test_file = Path(temp_repo_dir) / "test.py"
        test_file.write_text("def hello():\n    print('Hello')")
        first_commit = version_manager.commit_changes("First commit", files=["test.py"])
        
        test_file.write_text("def hello():\n    print('Hello, World!')")
        second_commit = version_manager.commit_changes("Second commit", files=["test.py"])
        
        diff = version_manager.get_diff(first_commit, second_commit)
        
        assert diff is not None
        assert "Hello" in diff
        assert "Hello, World!" in diff
    
    def test_get_diff_default_previous(self, version_manager, temp_repo_dir):
        """Test getting diff with default previous commit."""
        test_file = Path(temp_repo_dir) / "test.py"
        test_file.write_text("Version 1")
        version_manager.commit_changes("First commit", files=["test.py"])
        
        test_file.write_text("Version 2")
        version_manager.commit_changes("Second commit", files=["test.py"])
        
        diff = version_manager.get_diff()
        
        assert diff is not None
        assert "Version" in diff
