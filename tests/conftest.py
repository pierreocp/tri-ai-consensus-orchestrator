import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from tri_ai_orchestrator import OpenAIClient, AnthropicClient, GeminiClient, ChatMessage

@pytest.fixture
def sample_messages():
    """Fixture providing sample chat messages for testing."""
    return [
        ChatMessage(role="system", name="moderator", content="You are a helpful assistant."),
        ChatMessage(role="user", name="user", content="Hello, how are you?"),
        ChatMessage(role="assistant", name="ChatGPT", content="I'm doing well, thank you!")
    ]

@pytest.fixture
def openai_client():
    """Fixture providing an OpenAI client instance."""
    return OpenAIClient("ChatGPT", "gpt-4o")

@pytest.fixture
def anthropic_client():
    """Fixture providing an Anthropic client instance."""
    return AnthropicClient("Claude", "claude-3-5-sonnet-20240620")

@pytest.fixture
def gemini_client():
    """Fixture providing a Gemini client instance."""
    return GeminiClient("Gemini", "gemini-1.5-pro")

@pytest.fixture
def mock_httpx_response():
    """Fixture providing a mock httpx response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    return mock_response

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
