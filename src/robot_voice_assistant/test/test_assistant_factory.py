from robot_voice_assistant.assistant_factory import BrainSettings, create_backend
from robot_voice_assistant.llm_backends import FallbackBackend


def test_groq_with_llama_fallback_when_a_key_is_set() -> None:
    backend, description = create_backend(BrainSettings(), environment={"GROQ_API_KEY": "test-key"})
    assert isinstance(backend, FallbackBackend)
    assert "llama-3.3-70b-versatile" in description and "llama3.1:8b" in description


def test_only_the_local_model_without_a_key() -> None:
    backend, description = create_backend(BrainSettings(), environment={})
    assert isinstance(backend, FallbackBackend)
    assert "GROQ_API_KEY" in description  # tells the user why Groq is not used


def test_local_only_when_groq_is_switched_off() -> None:
    _, description = create_backend(BrainSettings(use_groq=False), environment={"GROQ_API_KEY": "k"})
    assert "groq" not in description.lower()
