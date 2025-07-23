import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import httpx
from tri_ai_orchestrator import OpenAIClient, AnthropicClient, GeminiClient, ChatMessage, _post_with_retry

class TestOpenAIClient:
    """Tests for OpenAI client functionality."""
    
    @pytest.fixture
    def openai_client(self):
        return OpenAIClient("ChatGPT", "gpt-4o")
    
    @pytest.fixture
    def sample_openai_response(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": "Hello! How can I help you today?"
                    }
                }
            ]
        }
    
    @pytest.mark.asyncio
    @patch('tri_ai_orchestrator.os.getenv')
    @patch('tri_ai_orchestrator._post_with_retry')
    async def test_openai_successful_request(self, mock_post, mock_getenv, openai_client, sample_openai_response, sample_messages):
        """Test successful OpenAI API request."""
        mock_getenv.return_value = "test_api_key"
        
        mock_response = Mock()
        mock_response.json.return_value = sample_openai_response
        mock_response.text = json.dumps(sample_openai_response)
        mock_post.return_value = mock_response
        
        result = await openai_client.send(
            sample_messages, 
            max_tokens=100, 
            timeout=30, 
            verbose=False
        )
        
        assert result == "Hello! How can I help you today?"
        mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('tri_ai_orchestrator.os.getenv')
    async def test_openai_missing_api_key(self, mock_getenv, openai_client, sample_messages):
        """Test OpenAI client with missing API key."""
        mock_getenv.return_value = None
        
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY manquant"):
            await openai_client.send(
                sample_messages,
                max_tokens=100,
                timeout=30,
                verbose=False
            )
    
    @pytest.mark.asyncio
    @patch('tri_ai_orchestrator.os.getenv')
    @patch('tri_ai_orchestrator._post_with_retry')
    async def test_openai_api_error_handling(self, mock_post, mock_getenv, openai_client, sample_messages):
        """Test OpenAI client error handling."""
        mock_getenv.return_value = "test_api_key"
        
        mock_post.side_effect = httpx.HTTPStatusError(
            "API Error", 
            request=Mock(), 
            response=Mock(status_code=429)
        )
        
        with pytest.raises(httpx.HTTPStatusError):
            await openai_client.send(
                sample_messages,
                max_tokens=100,
                timeout=30,
                verbose=False
            )

class TestAnthropicClient:
    """Tests for Anthropic client functionality."""
    
    @pytest.fixture
    def anthropic_client(self):
        return AnthropicClient("Claude", "claude-3-5-sonnet-20240620")
    
    @pytest.fixture
    def sample_anthropic_response(self):
        return {
            "content": [
                {
                    "text": "Hello! I'm Claude, how can I assist you?"
                }
            ]
        }
    
    @pytest.mark.asyncio
    @patch('tri_ai_orchestrator.os.getenv')
    @patch('tri_ai_orchestrator._post_with_retry')
    async def test_anthropic_successful_request(self, mock_post, mock_getenv, anthropic_client, sample_anthropic_response, sample_messages):
        """Test successful Anthropic API request."""
        mock_getenv.return_value = "test_api_key"
        
        mock_response = Mock()
        mock_response.json.return_value = sample_anthropic_response
        mock_response.text = json.dumps(sample_anthropic_response)
        mock_post.return_value = mock_response
        
        result = await anthropic_client.send(
            sample_messages,
            max_tokens=100,
            timeout=30,
            verbose=False
        )
        
        assert result == "Hello! I'm Claude, how can I assist you?"
        mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('tri_ai_orchestrator.os.getenv')
    async def test_anthropic_missing_api_key(self, mock_getenv, anthropic_client, sample_messages):
        """Test Anthropic client with missing API key."""
        mock_getenv.return_value = None
        
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY manquant"):
            await anthropic_client.send(
                sample_messages,
                max_tokens=100,
                timeout=30,
                verbose=False
            )
    
    @pytest.mark.asyncio
    @patch('tri_ai_orchestrator.os.getenv')
    @patch('tri_ai_orchestrator._post_with_retry')
    async def test_anthropic_message_transformation(self, mock_post, mock_getenv, anthropic_client, sample_anthropic_response):
        """Test Anthropic message format transformation."""
        mock_getenv.return_value = "test_api_key"
        
        mock_response = Mock()
        mock_response.json.return_value = sample_anthropic_response
        mock_response.text = json.dumps(sample_anthropic_response)
        mock_post.return_value = mock_response
        
        messages = [
            ChatMessage(role="system", name="moderator", content="You are helpful"),
            ChatMessage(role="user", name="user", content="Hello"),
            ChatMessage(role="assistant", name="Claude", content="Hi there!")
        ]
        
        await anthropic_client.send(messages, max_tokens=100, timeout=30, verbose=False)
        
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        
        assert request_data['system'] == "You are helpful"
        assert len(request_data['messages']) == 2
        assert request_data['messages'][0]['role'] == 'user'
        assert request_data['messages'][1]['role'] == 'assistant'

class TestGeminiClient:
    """Tests for Gemini client functionality."""
    
    @pytest.fixture
    def gemini_client(self):
        return GeminiClient("Gemini", "gemini-1.5-pro")
    
    @pytest.fixture
    def sample_gemini_response(self):
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Hello! I'm Gemini, ready to help!"
                            }
                        ]
                    }
                }
            ]
        }
    
    @pytest.mark.asyncio
    @patch('tri_ai_orchestrator.os.getenv')
    @patch('tri_ai_orchestrator._post_with_retry')
    async def test_gemini_successful_request(self, mock_post, mock_getenv, gemini_client, sample_gemini_response, sample_messages):
        """Test successful Gemini API request."""
        mock_getenv.return_value = "test_api_key"
        
        mock_response = Mock()
        mock_response.json.return_value = sample_gemini_response
        mock_response.text = json.dumps(sample_gemini_response)
        mock_post.return_value = mock_response
        
        result = await gemini_client.send(
            sample_messages,
            max_tokens=100,
            timeout=30,
            verbose=False
        )
        
        assert result == "Hello! I'm Gemini, ready to help!"
        mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('tri_ai_orchestrator.os.getenv')
    async def test_gemini_missing_api_key(self, mock_getenv, gemini_client, sample_messages):
        """Test Gemini client with missing API key."""
        mock_getenv.return_value = None
        
        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY manquant"):
            await gemini_client.send(
                sample_messages,
                max_tokens=100,
                timeout=30,
                verbose=False
            )
    
    @pytest.mark.asyncio
    @patch('tri_ai_orchestrator.os.getenv')
    @patch('tri_ai_orchestrator._post_with_retry')
    async def test_gemini_message_transformation(self, mock_post, mock_getenv, gemini_client, sample_gemini_response):
        """Test Gemini message format transformation."""
        mock_getenv.return_value = "test_api_key"
        
        mock_response = Mock()
        mock_response.json.return_value = sample_gemini_response
        mock_response.text = json.dumps(sample_gemini_response)
        mock_post.return_value = mock_response
        
        messages = [
            ChatMessage(role="system", name="moderator", content="You are helpful"),
            ChatMessage(role="user", name="user", content="Hello"),
            ChatMessage(role="assistant", name="Gemini", content="Hi!")
        ]
        
        await gemini_client.send(messages, max_tokens=100, timeout=30, verbose=False)
        
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        
        assert len(request_data['contents']) == 3
        assert request_data['contents'][0]['parts'][0]['text'] == "You are helpful"
        assert request_data['contents'][1]['parts'][0]['text'] == "Hello"
        assert request_data['contents'][2]['parts'][0]['text'] == "Hi!"

class TestRetryMechanism:
    """Tests for the retry mechanism used by all clients."""
    
    @pytest.mark.asyncio
    @patch('tri_ai_orchestrator.asyncio.sleep')
    @patch('tri_ai_orchestrator.httpx.AsyncClient.post')
    async def test_retry_on_connection_error(self, mock_post, mock_sleep):
        """Test retry mechanism on connection errors."""
        mock_post.side_effect = [
            httpx.ConnectError("Connection failed"),
            httpx.ConnectError("Connection failed"),
            Mock(status_code=200, json=lambda: {"test": "success"})
        ]
        
        async with httpx.AsyncClient() as client:
            result = await _post_with_retry(
                client,
                "https://test.com",
                headers={"Authorization": "Bearer test"},
                json={"test": "data"},
                timeout=30
            )
        
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2
        assert result.json() == {"test": "success"}
    
    @pytest.mark.asyncio
    @patch('tri_ai_orchestrator.asyncio.sleep')
    @patch('tri_ai_orchestrator.httpx.AsyncClient.post')
    async def test_retry_on_timeout_error(self, mock_post, mock_sleep):
        """Test retry mechanism on timeout errors."""
        mock_post.side_effect = [
            httpx.TimeoutException("Request timeout"),
            Mock(status_code=200, json=lambda: {"test": "success"})
        ]
        
        async with httpx.AsyncClient() as client:
            result = await _post_with_retry(
                client,
                "https://test.com",
                headers={"Authorization": "Bearer test"},
                json={"test": "data"},
                timeout=30
            )
        
        assert mock_post.call_count == 2
        assert mock_sleep.call_count == 1
        assert result.json() == {"test": "success"}
    
    @pytest.mark.asyncio
    @patch('tri_ai_orchestrator.asyncio.sleep')
    @patch('tri_ai_orchestrator.httpx.AsyncClient.post')
    async def test_retry_on_rate_limit(self, mock_post, mock_sleep):
        """Test retry mechanism on rate limit errors."""
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Rate limited", request=Mock(), response=rate_limit_response
        )
        
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {"test": "success"}
        success_response.raise_for_status.return_value = None
        
        mock_post.side_effect = [rate_limit_response, success_response]
        
        async with httpx.AsyncClient() as client:
            result = await _post_with_retry(
                client,
                "https://test.com",
                headers={"Authorization": "Bearer test"},
                json={"test": "data"},
                timeout=30
            )
        
        assert mock_post.call_count == 2
        assert mock_sleep.call_count == 1
        assert result.json() == {"test": "success"}
    
    @pytest.mark.asyncio
    @patch('tri_ai_orchestrator.asyncio.sleep')
    @patch('tri_ai_orchestrator.httpx.AsyncClient.post')
    async def test_max_retries_exceeded(self, mock_post, mock_sleep):
        """Test that max retries are respected."""
        mock_post.side_effect = httpx.ConnectError("Connection failed")
        
        with pytest.raises(httpx.ConnectError):
            async with httpx.AsyncClient() as client:
                await _post_with_retry(
                    client,
                    "https://test.com",
                    headers={"Authorization": "Bearer test"},
                    json={"test": "data"},
                    timeout=30
                )
        
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2
    
    @pytest.mark.asyncio
    @patch('tri_ai_orchestrator.asyncio.sleep')
    @patch('tri_ai_orchestrator.httpx.AsyncClient.post')
    async def test_exponential_backoff(self, mock_post, mock_sleep):
        """Test exponential backoff timing."""
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {"test": "success"}
        success_response.raise_for_status.return_value = None
        
        mock_post.side_effect = [
            httpx.ConnectError("Connection failed"),
            httpx.ConnectError("Connection failed"),
            success_response
        ]
        
        async with httpx.AsyncClient() as client:
            result = await _post_with_retry(
                client,
                "https://test.com",
                headers={"Authorization": "Bearer test"},
                json={"test": "data"},
                timeout=30
            )
        
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert len(sleep_calls) == 2  # 2 retries = 2 sleep calls
        assert sleep_calls[0] == 1    # First retry: 1 second
        assert sleep_calls[1] == 2    # Second retry: 2 seconds
        assert mock_post.call_count == 3  # 3 total calls (2 failures + 1 success)
        assert result.json() == {"test": "success"}

class TestMentionDetection:
    """Tests for mention detection in orchestrator."""
    
    def test_detect_mentions_single_mention(self):
        """Test detection of single AI mention using regex."""
        import re
        
        message = "Hey @ChatGPT, what do you think about this?"
        mentions = re.findall(r'@(\w+)', message)
        
        assert mentions == ["ChatGPT"]
    
    def test_detect_mentions_multiple_mentions(self):
        """Test detection of multiple AI mentions using regex."""
        import re
        
        message = "I agree with @Claude but @Gemini has a point too"
        mentions = re.findall(r'@(\w+)', message)
        
        assert set(mentions) == {"Claude", "Gemini"}
    
    def test_detect_mentions_no_mentions(self):
        """Test when no mentions are present."""
        import re
        
        message = "This is a regular message without any mentions"
        mentions = re.findall(r'@(\w+)', message)
        
        assert mentions == []
    
    def test_detect_mentions_case_insensitive(self):
        """Test case insensitive mention detection."""
        import re
        
        message = "Hey @chatgpt and @CLAUDE, what do you think?"
        mentions = re.findall(r'@(\w+)', message)
        
        assert set(mentions) == {"chatgpt", "CLAUDE"}
