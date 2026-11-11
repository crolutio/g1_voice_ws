"""Names a small object from its box in the camera image with CLIP (zero-shot, among the vision vocabulary).

YOLO finds held things well but names small ones badly (a red apple in a hand is a "sports ball"); CLIP, shown
just the box and a margin around it, says how well each word of the vocabulary fits it.
"""

from __future__ import annotations

import numpy as np

CROP_MARGIN = 0.3  # of the box's longer side, on every side: the hand and a little background help CLIP


class ClipObjectNamer:
    def __init__(self, weights: str, words: list[str], device: str = "cuda") -> None:
        import clip  # imported here: loading takes a few seconds
        import torch

        self._torch, self._device, self.words = torch, device, list(words)
        self._model, self._prepare = clip.load(weights, device=device)
        with torch.no_grad():
            text = self._model.encode_text(clip.tokenize([f"a photo of a {word}." for word in self.words]).to(device))
        self._text = (text / text.norm(dim=-1, keepdim=True)).float()

    def shares(self, rgb: np.ndarray, box: tuple[float, float, float, float]) -> dict[str, float]:
        """CLIP's belief that what is in ``box`` (shares of the image) is each word; the shares add up to 1, and
        there is no "none of these". Empty for a box too small to look at."""
        from PIL import Image

        height, width = rgb.shape[:2]
        margin = CROP_MARGIN * max((box[2] - box[0]) * width, (box[3] - box[1]) * height)
        x1, y1 = int(max(0, box[0] * width - margin)), int(max(0, box[1] * height - margin))
        x2, y2 = int(min(width, box[2] * width + margin)), int(min(height, box[3] * height + margin))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return {}
        image = self._prepare(Image.fromarray(np.ascontiguousarray(rgb[y1:y2, x1:x2]))).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            features = self._model.encode_image(image).float()
        features = features / features.norm(dim=-1, keepdim=True)
        shares = (100.0 * features @ self._text.T).softmax(dim=-1)[0].tolist()
        return dict(zip(self.words, shares))
