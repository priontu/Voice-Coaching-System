from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PhonemeInterval:
    phoneme: str
    start_sec: float
    end_sec: float
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PhonemePrediction:
    wav_path: str
    model_name: str
    sample_rate: int
    phonemes: list[PhonemeInterval]
    transcript: str

    def to_dict(self) -> dict:
        return {
            "wav_path": self.wav_path,
            "model_name": self.model_name,
            "sample_rate": self.sample_rate,
            "transcript": self.transcript,
            "phonemes": [item.to_dict() for item in self.phonemes],
        }


class HFPhonemeExtractor:
    """Wav2Vec2-Phoneme extractor with rough CTC frame boundaries."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-lv-60-espeak-cv-ft",
        device: str | torch.device | None = None,
    ):
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = None
        self.model = None
        self.target_sample_rate = 16000

    def predict(self, wav_path: str | Path) -> dict:
        self._load_model()
        waveform = self._load_audio(wav_path)
        duration_sec = waveform.numel() / self.target_sample_rate
        inputs = self.processor(
            waveform.cpu().numpy(),
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=False,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            logits = self.model(**inputs).logits[0]
            probs = F.softmax(logits, dim=-1)

        token_ids = logits.argmax(dim=-1).detach().cpu()
        token_conf = probs.max(dim=-1).values.detach().cpu()
        intervals = self._ctc_intervals(token_ids, token_conf, duration_sec)
        transcript = self.processor.batch_decode(token_ids.unsqueeze(0))[0]
        return PhonemePrediction(
            wav_path=str(wav_path),
            model_name=self.model_name,
            sample_rate=self.target_sample_rate,
            phonemes=intervals,
            transcript=transcript,
        ).to_dict()

    def _load_model(self) -> None:
        if self.model is not None:
            return
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        except ImportError as exc:
            raise RuntimeError("transformers is required for PhonemeExtraction_v1") from exc

        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        except ImportError as exc:
            raise RuntimeError(
                "The facebook/wav2vec2 phoneme tokenizer requires phonemizer. Install with:\n"
                "  pip install phonemizer\n"
                "It may also require the espeak-ng system backend. With conda, try:\n"
                "  conda install -c conda-forge espeak-ng\n"
            ) from exc

        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.model_name,
            use_safetensors=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def _load_audio(self, wav_path: str | Path) -> torch.Tensor:
        try:
            import torchaudio
            import torchaudio.functional as F_audio
        except ImportError as exc:
            raise RuntimeError("torchaudio is required for PhonemeExtraction_v1") from exc

        audio, sample_rate = torchaudio.load(wav_path)
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        if sample_rate != self.target_sample_rate:
            audio = F_audio.resample(audio, sample_rate, self.target_sample_rate)
        return audio.float().contiguous()

    def _ctc_intervals(
        self,
        token_ids: torch.Tensor,
        token_conf: torch.Tensor,
        duration_sec: float,
    ) -> list[PhonemeInterval]:
        if self.processor is None:
            raise RuntimeError("Processor is not loaded; _load_model() must succeed before _ctc_intervals() is called.")
        vocab = self.processor.tokenizer.get_vocab()
        id_to_token = {idx: token for token, idx in vocab.items()}
        blank_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        word_delim = getattr(self.processor.tokenizer, "word_delimiter_token", "|")

        frame_count = int(token_ids.numel())
        if frame_count == 0:
            return []
        frame_sec = duration_sec / frame_count

        intervals: list[PhonemeInterval] = []
        prev_token: int | None = None
        current_start = 0
        current_confs: list[float] = []

        for index, token_id_tensor in enumerate(token_ids):
            token_id = int(token_id_tensor.item())
            is_blank = blank_id is not None and token_id == int(blank_id)
            if is_blank:
                if prev_token is not None:
                    self._append_interval(intervals, id_to_token, prev_token, current_start, index, frame_sec, current_confs, word_delim)
                    prev_token = None
                    current_confs = []
                continue

            if prev_token is None:
                prev_token = token_id
                current_start = index
                current_confs = [float(token_conf[index].item())]
            elif token_id == prev_token:
                current_confs.append(float(token_conf[index].item()))
            else:
                self._append_interval(intervals, id_to_token, prev_token, current_start, index, frame_sec, current_confs, word_delim)
                prev_token = token_id
                current_start = index
                current_confs = [float(token_conf[index].item())]

        if prev_token is not None:
            self._append_interval(intervals, id_to_token, prev_token, current_start, frame_count, frame_sec, current_confs, word_delim)

        return intervals

    def _append_interval(
        self,
        intervals: list[PhonemeInterval],
        id_to_token: dict[int, str],
        token_id: int,
        start_frame: int,
        end_frame: int,
        frame_sec: float,
        confidences: list[float],
        word_delim: str,
    ) -> None:
        token = id_to_token.get(token_id, str(token_id))
        if token in {"<pad>", "<s>", "</s>", "<unk>", word_delim, "|"}:
            return
        clean = token.replace("▁", "").strip()
        if not clean:
            return
        intervals.append(
            PhonemeInterval(
                phoneme=clean,
                start_sec=start_frame * frame_sec,
                end_sec=end_frame * frame_sec,
                confidence=float(sum(confidences) / len(confidences)) if confidences else 0.0,
            )
        )
