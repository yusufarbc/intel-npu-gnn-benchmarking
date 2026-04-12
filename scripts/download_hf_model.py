from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from huggingface_hub import HfApi, hf_hub_download


@dataclass(frozen=True)
class ModelRequest:
    display_name: str
    search_terms: Tuple[str, ...]
    output_filename: str


class HuggingFaceONNXFinder:
    def __init__(self) -> None:
        self.api = HfApi()

    @staticmethod
    def _score_filename(filename: str, prefer: Iterable[str]) -> int:
        score = 0
        lower = filename.lower()
        for token in prefer:
            if token in lower:
                score += 10
        if lower.endswith(".onnx"):
            score += 5
        if any(x in lower for x in ["int8", "quant", "fp16", "bf16"]):
            score -= 2
        if any(x in lower for x in ["decoder", "encoder"]):
            score += 1
        return score

    def _pick_onnx_file(self, repo_id: str, prefer_tokens: Iterable[str]) -> Optional[str]:
        try:
            files = self.api.list_repo_files(repo_id=repo_id, repo_type="model")
        except Exception:
            return None

        onnx_files = [f for f in files if f.lower().endswith(".onnx")]
        if not onnx_files:
            return None

        ranked = sorted(
            onnx_files,
            key=lambda f: self._score_filename(f, prefer=prefer_tokens),
            reverse=True,
        )
        return ranked[0]

    def find_repo_and_file(self, request: ModelRequest) -> Tuple[str, str]:
        prefer_tokens = [t.lower() for t in request.search_terms] + ["model", "onnx"]
        query = " ".join(request.search_terms)

        # Search a handful of candidate repos; then pick the best .onnx file inside.
        candidates = list(self.api.list_models(search=query, limit=30))
        for candidate in candidates:
            repo_id = candidate.modelId
            filename = self._pick_onnx_file(repo_id, prefer_tokens)
            if filename:
                return repo_id, filename

        raise RuntimeError(
            f"Could not find an ONNX file on Hugging Face for: {request.display_name} (search={query!r})"
        )


class ModelDownloader:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.finder = HuggingFaceONNXFinder()

    def download(self, request: ModelRequest) -> Path:
        repo_id, filename = self.finder.find_repo_and_file(request)
        print(f"[download] {request.display_name}: {repo_id}/{filename}")

        cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
        destination = self.models_dir / request.output_filename
        shutil.copy2(cached_path, destination)
        print(f"[download] Saved: {destination}")
        return destination


def main() -> None:
    requests: List[ModelRequest] = [
        ModelRequest(
            display_name="MobileNetV2",
            search_terms=("mobilenetv2", "onnx"),
            output_filename="MobileNetV2.onnx",
        ),
        ModelRequest(
            display_name="ResNet-50",
            search_terms=("resnet", "50", "onnx"),
            output_filename="ResNet50.onnx",
        ),
        ModelRequest(
            display_name="BERT-tiny",
            search_terms=("bert", "tiny", "onnx"),
            output_filename="BERT-tiny.onnx",
        ),
    ]

    downloader = ModelDownloader(models_dir=Path("models"))
    for request in requests:
        downloader.download(request)


if __name__ == "__main__":
    main()
