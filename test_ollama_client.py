import unittest
from unittest.mock import patch

from llm.ollama import OllamaClient


class TestOllamaClient(unittest.TestCase):
    def test_chat_timeout_raises_runtime_error(self) -> None:
        client = OllamaClient(base_url="http://localhost:11434", model="test-model", timeout_seconds=0.1)

        with patch("llm.ollama.urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaises(RuntimeError) as ctx:
                client.chat(system="system", user="user")

        message = str(ctx.exception)
        self.assertIn("Ollama", message)
        self.assertIn("localhost:11434", message)


if __name__ == "__main__":
    unittest.main()
