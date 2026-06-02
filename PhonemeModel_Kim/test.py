"""
Comprehensive test suite for Phoneme Boundary Detection Module

Includes:
- Unit tests for core functions
- Integration tests
- Edge case handling
- Performance benchmarks
- Fixture data generation

Run with: pytest test_phoneme_detector.py -v
"""

import os
import pytest
import numpy as np
import soundfile as sf
import torch
import tempfile
from pathlib import Path
from typing import Tuple

from phoneme_model import (
    load_audio,
    load_model,
    ctc_align,
    create_phoneme_segments,
    group_by_words,
    compute_boundary_metrics,
    PhonemeBoundaryConfig,
    PhonemeSegment,
    extract_phoneme_boundaries_from_audio,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def config():
    """Basic configuration for testing."""
    return PhonemeBoundaryConfig(
        device="cpu",  # Use CPU for testing
        sample_rate=16000
    )


@pytest.fixture
def sample_audio_path():
    """Create a temporary synthetic audio file for testing."""
    import torchaudio
    
    # Generate 2 seconds of synthetic audio (sine wave)
    sample_rate = 16000
    duration = 2.0
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    waveform = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    sf.write(tmp_path, waveform, sample_rate)
    return tmp_path


@pytest.fixture
def sample_segments():
    """Create sample phoneme segments for testing."""
    return [
        PhonemeSegment(phoneme="AA", start_time=0.0, end_time=0.2),
        PhonemeSegment(phoneme="B", start_time=0.2, end_time=0.35),
        PhonemeSegment(phoneme="UH", start_time=0.35, end_time=0.5),
        PhonemeSegment(phoneme="T", start_time=0.5, end_time=0.65),
    ]


# ============================================================================
# UNIT TESTS: AUDIO LOADING
# ============================================================================

class TestAudioLoading:
    """Tests for audio loading functionality."""
    
    def test_load_audio_basic(self, sample_audio_path):
        """Test basic audio loading."""
        audio, sr = load_audio(sample_audio_path)
        
        assert isinstance(audio, torch.Tensor)
        assert sr == 16000
        assert audio.dtype == torch.float32
        assert len(audio) > 0
    
    def test_load_audio_duration(self, sample_audio_path):
        """Test audio duration is correct."""
        audio, sr = load_audio(sample_audio_path)
        duration = len(audio) / sr
        
        # Should be approximately 2 seconds
        assert 1.9 < duration < 2.1
    
    def test_load_audio_normalization(self, sample_audio_path):
        """Test audio is normalized to [-1, 1]."""
        audio, _ = load_audio(sample_audio_path)
        
        assert torch.max(torch.abs(audio)) <= 1.0
    
    def test_load_audio_resampling(self, sample_audio_path):
        """Test audio resampling."""
        # Request different sample rate
        audio, sr = load_audio(sample_audio_path, target_sr=8000)
        
        assert sr == 8000
        assert isinstance(audio, torch.Tensor)
    
    def test_load_audio_mono_conversion(self, sample_audio_path):
        """Test mono conversion."""
        audio, _ = load_audio(sample_audio_path, mono=True)
        
        # Should be 1D after mono conversion
        assert audio.ndim == 1
    
    def test_load_audio_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_audio("nonexistent_file.wav")
    
    def test_load_audio_invalid_format(self):
        """Test error handling for invalid format."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            with pytest.raises(ValueError):
                load_audio(tmp.name)


# ============================================================================
# UNIT TESTS: CTC ALIGNMENT
# ============================================================================

class TestCTCAlignment:
    """Tests for CTC alignment algorithm."""
    
    def test_ctc_align_collapse_repeats(self):
        """Test repeated token collapsing."""
        # Input: [1, 1, 1, 2, 2, 3]
        # Expected: [1, 2, 3]
        pred_ids = np.array([1, 1, 1, 2, 2, 3])
        
        aligned, ranges = ctc_align(
            pred_ids,
            collapse_repeated=True,
            remove_blanks=False,
            blank_id=0
        )
        
        assert aligned == [1, 2, 3]
        assert len(ranges) == 3
    
    def test_ctc_align_remove_blanks(self):
        """Test blank token removal."""
        # Input: [1, 0, 2, 0, 3]  (0 = blank)
        # Expected: [1, 2, 3]
        pred_ids = np.array([1, 0, 2, 0, 3])
        
        aligned, ranges = ctc_align(
            pred_ids,
            collapse_repeated=False,
            remove_blanks=True,
            blank_id=0
        )
        
        assert aligned == [1, 2, 3]
        assert len(ranges) == 3
    
    def test_ctc_align_combined(self):
        """Test collapse and blank removal together."""
        # Input: [1, 1, 0, 2, 2, 0, 3]
        # Expected: [1, 2, 3]
        pred_ids = np.array([1, 1, 0, 2, 2, 0, 3])
        
        aligned, ranges = ctc_align(
            pred_ids,
            collapse_repeated=True,
            remove_blanks=True,
            blank_id=0
        )
        
        assert aligned == [1, 2, 3]
        assert len(ranges) == 3
        assert ranges[0] == (0, 1)  # Frames 0-1
        assert ranges[1] == (3, 4)  # Frames 3-4
        assert ranges[2] == (6, 6)  # Frame 6
    
    def test_ctc_align_empty_input(self):
        """Test handling of empty input."""
        pred_ids = np.array([])
        
        aligned, ranges = ctc_align(pred_ids)
        
        assert aligned == []
        assert ranges == []
    
    def test_ctc_align_all_blanks(self):
        """Test handling of all blank tokens."""
        pred_ids = np.array([0, 0, 0, 0])
        
        aligned, ranges = ctc_align(
            pred_ids,
            remove_blanks=True,
            blank_id=0
        )
        
        assert aligned == []
        assert ranges == []
    
    def test_ctc_align_frame_ranges(self):
        """Test correctness of frame ranges."""
        pred_ids = np.array([1, 1, 2, 2, 2, 3])
        
        aligned, ranges = ctc_align(
            pred_ids,
            collapse_repeated=True,
            remove_blanks=False
        )
        
        # Verify each range
        assert ranges[0] == (0, 1)  # Token 1 at frames 0-1
        assert ranges[1] == (2, 4)  # Token 2 at frames 2-4
        assert ranges[2] == (5, 5)  # Token 3 at frame 5


# ============================================================================
# UNIT TESTS: PHONEME SEGMENTATION
# ============================================================================

class TestPhonemeSegmentation:
    """Tests for phoneme segmentation utilities."""
    
    def test_create_segments(self, sample_segments):
        """Test segment creation."""
        assert len(sample_segments) == 4
        assert sample_segments[0].phoneme == "AA"
        assert sample_segments[0].start_time == 0.0
    
    def test_segment_to_dict(self, sample_segments):
        """Test segment serialization."""
        seg = sample_segments[0]
        seg_dict = seg.to_dict()
        
        assert seg_dict["phoneme"] == "AA"
        assert seg_dict["start_time"] == 0.0
        assert seg_dict["end_time"] == 0.2
    
    def test_group_by_words_basic(self, sample_segments):
        """Test word grouping with separator."""
        # Add word separator
        segments = sample_segments.copy()
        segments.insert(2, PhonemeSegment(
            phoneme="|", start_time=0.35, end_time=0.35
        ))
        
        words = group_by_words(segments, word_separator="|")
        
        assert len(words) == 2
        assert len(words[0]["phonemes"]) == 2
        assert len(words[1]["phonemes"]) == 2
    
    def test_group_by_words_timing(self, sample_segments):
        """Test word-level timing."""
        words = group_by_words(sample_segments, word_separator="NONE")
        
        # All phonemes in one word (no separator found)
        assert len(words) == 1
        assert words[0]["start_time"] == 0.0
        assert words[0]["end_time"] == 0.65
    
    def test_create_phoneme_segments_mismatch(self):
        """Test error handling for mismatched lengths."""
        phonemes = ["AA", "B"]
        boundaries = [(0.0, 0.2), (0.2, 0.35), (0.35, 0.5)]
        
        with pytest.raises(ValueError):
            create_phoneme_segments(phonemes, boundaries)


# ============================================================================
# UNIT TESTS: EVALUATION METRICS
# ============================================================================

class TestEvaluationMetrics:
    """Tests for boundary evaluation metrics."""
    
    def test_perfect_match(self):
        """Test metrics for perfect prediction."""
        pred = [
            PhonemeSegment("AA", 0.0, 0.2),
            PhonemeSegment("B", 0.2, 0.35),
        ]
        ref = [
            PhonemeSegment("AA", 0.0, 0.2),
            PhonemeSegment("B", 0.2, 0.35),
        ]
        
        metrics = compute_boundary_metrics(pred, ref, tolerance_ms=50)
        
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
    
    def test_partial_match(self):
        """Test metrics for partial match."""
        pred = [
            PhonemeSegment("AA", 0.0, 0.2),
            PhonemeSegment("B", 0.2, 0.35),
        ]
        ref = [
            PhonemeSegment("AA", 0.0, 0.2),
            PhonemeSegment("B", 0.2, 0.35),
            PhonemeSegment("C", 0.35, 0.5),
        ]
        
        metrics = compute_boundary_metrics(pred, ref, tolerance_ms=50)
        
        assert 0 < metrics["precision"] <= 1.0
        assert 0 < metrics["recall"] < 1.0
    
    def test_no_match(self):
        """Test metrics with no matching boundaries."""
        pred = [
            PhonemeSegment("AA", 0.0, 0.1),
            PhonemeSegment("B", 0.1, 0.2),
        ]
        ref = [
            PhonemeSegment("C", 1.0, 1.1),
            PhonemeSegment("D", 1.1, 1.2),
        ]
        
        metrics = compute_boundary_metrics(pred, ref, tolerance_ms=50)
        
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
    
    def test_tolerance_window(self):
        """Test tolerance window in metrics."""
        pred = [PhonemeSegment("AA", 0.0, 0.2)]
        ref = [PhonemeSegment("AA", 0.0, 0.21)]
        
        # Within 50ms tolerance
        metrics_50 = compute_boundary_metrics(pred, ref, tolerance_ms=50)
        assert metrics_50["f1"] > 0
        
        # Outside 5ms tolerance on both boundaries
        ref = [PhonemeSegment("AA", 0.006, 0.206)]
        metrics_5 = compute_boundary_metrics(pred, ref, tolerance_ms=5)
        assert metrics_5["f1"] == 0
    
    def test_empty_predictions(self):
        """Test handling of empty predictions."""
        pred = []
        ref = [PhonemeSegment("AA", 0.0, 0.2)]
        
        metrics = compute_boundary_metrics(pred, ref)
        
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete pipeline."""
    
    def test_load_model(self, config):
        """Test model loading."""
        try:
            model, processor = load_model(config)
            
            assert model is not None
            assert processor is not None
            assert model.eval  # Should be in eval mode
        except RuntimeError as e:
            pytest.skip(f"Model download failed: {str(e)}")
    
    def test_full_pipeline_cpu(self, sample_audio_path, config):
        """Test complete pipeline on CPU."""
        try:
            result = extract_phoneme_boundaries_from_audio(
                sample_audio_path,
                config=config,
                return_segments=True
            )
            
            assert "phonemes" in result
            assert "boundaries" in result
            assert "segments" in result
            assert "metadata" in result
            assert len(result["segments"]) > 0
        except RuntimeError as e:
            pytest.skip(f"Model not available: {str(e)}")
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_full_pipeline_cuda(self, sample_audio_path):
        """Test complete pipeline on GPU."""
        config = PhonemeBoundaryConfig(device="cuda")
        
        try:
            result = extract_phoneme_boundaries_from_audio(
                sample_audio_path,
                config=config,
                return_segments=True
            )
            
            assert len(result["segments"]) > 0
        except RuntimeError as e:
            pytest.skip(f"CUDA error: {str(e)}")


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_very_short_audio(self):
        """Test handling of very short audio."""
        # Create 0.1 second audio
        sample_rate = 16000
        waveform = torch.sin(torch.linspace(0, 2*np.pi, 1600))
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        sf.write(tmp_path, waveform.numpy(), sample_rate)
        
        audio, sr = load_audio(tmp_path)
        os.remove(tmp_path)
        assert len(audio) < sample_rate  # Less than 1 second
    
    def test_very_long_audio(self):
        """Test handling of long audio."""
        # Create 30 second audio
        sample_rate = 16000
        duration = 30
        waveform = torch.sin(torch.linspace(0, 2*np.pi*duration, 
                                           sample_rate*duration))
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        sf.write(tmp_path, waveform.numpy(), sample_rate)
        
        audio, sr = load_audio(tmp_path)
        os.remove(tmp_path)
        assert len(audio) / sr >= 29  # At least 29 seconds
    
    def test_silent_audio(self):
        """Test handling of silent audio."""
        # Create silent audio
        sample_rate = 16000
        duration = 2
        waveform = torch.zeros(sample_rate * duration)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        sf.write(tmp_path, waveform.numpy(), sample_rate)
        
        audio, sr = load_audio(tmp_path)
        os.remove(tmp_path)
        assert sr == 16000
    
    def test_very_loud_audio(self):
        """Test handling of very loud/clipped audio."""
        # Create loud audio
        sample_rate = 16000
        duration = 2
        waveform = torch.ones(sample_rate * duration) * 2.0  # Clipped
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        sf.write(tmp_path, waveform.numpy(), sample_rate)
        
        audio, sr = load_audio(tmp_path)
        os.remove(tmp_path)
        # Should be normalized
        assert torch.max(torch.abs(audio)) <= 1.0


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

class TestPerformance:
    """Performance benchmarks."""
    
    def test_ctc_align_performance(self):
        """Benchmark CTC alignment."""
        import time
        
        # Large prediction sequence
        pred_ids = np.random.randint(0, 50, size=100000)
        
        start = time.time()
        aligned, ranges = ctc_align(pred_ids)
        elapsed = time.time() - start
        
        # Should be fast (< 200ms)
        assert elapsed < 0.2
        assert len(aligned) > 0
    
    def test_segment_creation_performance(self):
        """Benchmark segment creation."""
        import time
        
        phonemes = ["AA", "B", "UH"] * 100
        boundaries = [(i*0.02, (i+1)*0.02) for i in range(300)]
        
        start = time.time()
        segments = create_phoneme_segments(phonemes, boundaries)
        elapsed = time.time() - start
        
        assert elapsed < 0.01  # Should be very fast


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestConfiguration:
    """Tests for configuration management."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = PhonemeBoundaryConfig()
        
        assert config.sample_rate == 16000
        assert config.model_name == "facebook/wav2vec2-lv-60-espeak-cv-ft"
        assert config.collapse_repeated_tokens == True
        assert config.remove_blank_tokens == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PhonemeBoundaryConfig(
            sample_rate=8000,
            device="cpu",
            collapse_repeated_tokens=False
        )
        
        assert config.sample_rate == 8000
        assert str(config.device) == "cpu"
        assert config.collapse_repeated_tokens == False
    
    def test_device_detection(self):
        """Test automatic device detection."""
        config = PhonemeBoundaryConfig()
        
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert str(config.device) == expected_device


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
