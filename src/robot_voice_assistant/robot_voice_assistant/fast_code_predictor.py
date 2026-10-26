"""Faster sub-code generation for Qwen3-TTS.

For every frame of audio (1/12 s), Qwen3-TTS predicts 15 more codes with a small 5-layer model. The library does
this with a full Hugging Face ``generate()`` call per frame, which costs about 40 ms of Python overhead for very
little GPU work, and makes the voice slower than real time while other models share the GPU.

This replaces that call with a fixed-shape loop: the sequence is at most 16 positions long, so it is recomputed
in full at every step (no key-value cache) and causal attention keeps later positions from affecting earlier ones.
On the GPU the whole 15-step loop is captured once as a CUDA graph and replayed for every frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch


@dataclass(frozen=True)
class Sampling:
    do_sample: bool
    temperature: float
    top_k: int


class FastCodePredictor:
    """Drop-in replacement for ``code_predictor.generate`` (one sentence at a time, top-k sampling)."""

    def __init__(self, predictor: torch.nn.Module) -> None:
        self._predictor = predictor
        self._library_generate = predictor.generate
        self._new_codes = predictor.config.num_code_groups - 1
        self._length = 1 + self._new_codes  # the 2 prompt positions, then one input per new code but the last
        self._on_gpu = next(predictor.parameters()).is_cuda
        positions = torch.arange(self._length, device=next(predictor.parameters()).device)[None]
        dtype = next(predictor.parameters()).dtype
        self._rotary = predictor.model.rotary_emb(torch.empty(0, dtype=dtype, device=positions.device), positions)
        self._graphs: dict[Sampling, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}

    def install(self) -> None:
        """Route the talker's calls to ``code_predictor.generate`` through this loop."""
        self._predictor.generate = self.generate

    def generate(
        self,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool = True,
        top_p: float = 1.0,
        top_k: int = 50,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> SimpleNamespace:
        supported = inputs_embeds.shape[0] == 1 and inputs_embeds.shape[1] == 2 and max_new_tokens == self._new_codes
        if not supported or (do_sample and top_p < 1.0):
            return self._library_generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                **kwargs,
            )
        sampling = Sampling(bool(do_sample), float(temperature), int(top_k or 0))
        with torch.inference_mode():
            if self._on_gpu:
                return SimpleNamespace(sequences=self._replay(inputs_embeds, sampling))
            return SimpleNamespace(sequences=self._codes(inputs_embeds, sampling))

    def _sequence_buffer(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        prompt = self._predictor.small_to_mtp_projection(inputs_embeds)
        sequence = prompt.new_zeros(prompt.shape[0], self._length, prompt.shape[-1])
        sequence[:, :2] = prompt
        return sequence

    def _embed(self, step: int, code: torch.Tensor) -> torch.Tensor:
        embedding = self._predictor.model.codec_embedding[step](code)
        return self._predictor.small_to_mtp_projection(embedding)

    def _forward(self, sequence: torch.Tensor) -> torch.Tensor:
        model = self._predictor.model
        hidden = sequence
        for layer in model.layers[: model.config.num_hidden_layers]:
            # no attention mask: SDPA then applies plain causal attention over the fixed-length sequence
            hidden = layer(hidden, attention_mask=None, position_embeddings=self._rotary)[0]
        return model.norm(hidden)

    def _choose(self, logits: torch.Tensor, sampling: Sampling) -> torch.Tensor:
        if not sampling.do_sample:
            return logits.argmax(dim=-1)
        scores = logits.float() / sampling.temperature
        if 0 < sampling.top_k < scores.shape[-1]:
            kth_best = torch.topk(scores, sampling.top_k).values[..., -1:]
            scores = scores.masked_fill(scores < kth_best, float("-inf"))
        # Sampling from softmax(scores) without a CPU sync: argmax of probability / Exponential(1) noise
        noise = torch.empty_like(scores).exponential_()
        return (scores - noise.log()).argmax(dim=-1)

    def _codes(self, inputs_embeds: torch.Tensor, sampling: Sampling) -> torch.Tensor:
        sequence = self._sequence_buffer(inputs_embeds)
        codes = []
        for step in range(self._new_codes):
            hidden = self._forward(sequence)[:, 1 + step]
            code = self._choose(self._predictor.lm_head[step](hidden), sampling)
            codes.append(code)
            if step + 1 < self._new_codes:
                sequence[:, 2 + step] = self._embed(step, code)
        return torch.stack(codes, dim=1)

    def _replay(self, inputs_embeds: torch.Tensor, sampling: Sampling) -> torch.Tensor:
        if sampling not in self._graphs:
            self._graphs[sampling] = self._capture(inputs_embeds, sampling)
        graph, static_input, static_codes = self._graphs[sampling]
        static_input.copy_(inputs_embeds)
        graph.replay()
        return static_codes.clone()

    def _capture(
        self, inputs_embeds: torch.Tensor, sampling: Sampling
    ) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]:
        static_input = inputs_embeds.clone()
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):  # warm-up outside the graph, as CUDA graph capture requires
                self._codes(static_input, sampling)
        torch.cuda.current_stream().wait_stream(side_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_codes = self._codes(static_input, sampling)
        return graph, static_input, static_codes
