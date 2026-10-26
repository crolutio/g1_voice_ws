"""The fast Qwen3-TTS sub-code loop must choose the same codes as the library's Hugging Face generate()."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
qwen_models = pytest.importorskip("qwen_tts.core.models.modeling_qwen3_tts")
qwen_configs = pytest.importorskip("qwen_tts.core.models.configuration_qwen3_tts")

from robot_voice_assistant.fast_code_predictor import FastCodePredictor  # noqa: E402

NUM_CODE_GROUPS = 6
TALKER_HIDDEN = 24


def tiny_predictor() -> torch.nn.Module:
    torch.manual_seed(0)
    predictor_config = qwen_configs.Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_code_groups=NUM_CODE_GROUPS,
    )
    predictor_config._attn_implementation = "sdpa"
    talker_config = qwen_configs.Qwen3TTSTalkerConfig(
        hidden_size=TALKER_HIDDEN, num_code_groups=NUM_CODE_GROUPS, code_predictor_config=predictor_config.to_dict()
    )
    predictor = qwen_models.Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(predictor_config, talker_config)
    return predictor.eval()


def prompt(batch: int = 1) -> torch.Tensor:
    return torch.randn(batch, 2, TALKER_HIDDEN)


def library_codes(predictor: torch.nn.Module, embeds: torch.Tensor, **sampling: object) -> torch.Tensor:
    with torch.inference_mode():
        result = predictor.generate(
            inputs_embeds=embeds,
            max_new_tokens=NUM_CODE_GROUPS - 1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **sampling,
        )
    return result.sequences


def test_greedy_codes_match_the_library() -> None:
    predictor = tiny_predictor()
    fast = FastCodePredictor(predictor)
    for _ in range(5):
        embeds = prompt()
        expected = library_codes(predictor, embeds, do_sample=False)
        result = fast.generate(inputs_embeds=embeds, max_new_tokens=NUM_CODE_GROUPS - 1, do_sample=False)
        assert torch.equal(result.sequences, expected)


def test_top_k_of_one_samples_the_greedy_codes() -> None:
    predictor = tiny_predictor()
    fast = FastCodePredictor(predictor)
    embeds = prompt()
    expected = library_codes(predictor, embeds, do_sample=False)
    result = fast.generate(
        inputs_embeds=embeds, max_new_tokens=NUM_CODE_GROUPS - 1, do_sample=True, top_k=1, top_p=1.0, temperature=0.9
    )
    assert torch.equal(result.sequences, expected)


def test_sampling_only_picks_from_the_top_k_codes() -> None:
    predictor = tiny_predictor()
    fast = FastCodePredictor(predictor)
    embeds = prompt()
    top_k = 3
    with torch.inference_mode():
        first_logits = predictor(inputs_embeds=embeds).logits[0, -1]  # the prompt alone decides the first code
    allowed = set(torch.topk(first_logits, top_k).indices.tolist())
    seen = set()
    for _ in range(40):
        result = fast.generate(
            inputs_embeds=embeds, max_new_tokens=NUM_CODE_GROUPS - 1, do_sample=True, top_k=top_k, temperature=0.9
        )
        seen.add(result.sequences[0, 0].item())
    assert seen <= allowed
    assert len(seen) > 1  # it really samples


def test_settings_it_does_not_handle_use_the_library() -> None:
    predictor = tiny_predictor()
    fast = FastCodePredictor(predictor)
    embeds = prompt(batch=2)  # only one sentence at a time is accelerated
    torch.manual_seed(1)
    expected = library_codes(predictor, embeds, do_sample=True, top_k=5, top_p=0.8, temperature=0.9)
    torch.manual_seed(1)
    result = fast.generate(
        inputs_embeds=embeds,
        max_new_tokens=NUM_CODE_GROUPS - 1,
        do_sample=True,
        top_k=5,
        top_p=0.8,
        temperature=0.9,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    assert torch.equal(result.sequences, expected)


def test_install_replaces_generate_on_the_predictor() -> None:
    predictor = tiny_predictor()
    fast = FastCodePredictor(predictor)
    fast.install()
    embeds = prompt()
    with torch.inference_mode():
        result = predictor.generate(inputs_embeds=embeds, max_new_tokens=NUM_CODE_GROUPS - 1, do_sample=False)
    assert result.sequences.shape == (1, NUM_CODE_GROUPS - 1)
