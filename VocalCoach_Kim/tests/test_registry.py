"""
tests/test_registry.py - Unit tests for models/registry.py

Tests use lightweight stubs so no model weights are required.

Run with:
    cd MusicAI/VocalCoach
    python -m pytest tests/test_registry.py -v
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from models.base import BaseInferenceModel
from models.registry import ModelRegistry, get_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubModel(BaseInferenceModel):
    """Minimal concrete model for testing the registry contract."""

    def __init__(self, name="stub", **kwargs):
        super().__init__()
        self.name = name
        self.load_called = 0

    def load_model(self) -> None:
        self.load_called += 1
        self._is_loaded = True

    def predict(self, audio):
        return f"prediction from {self.name}"


def _stub_factory(**kwargs):
    return _StubModel(**kwargs)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_and_listed(self):
        r = ModelRegistry()
        r.register("custom", _stub_factory)
        assert "custom" in r.registered_names()

    def test_default_models_registered(self):
        r = ModelRegistry()
        assert "phoneme" in r.registered_names()
        assert "pitch" in r.registered_names()
        assert "onset_offset" in r.registered_names()

    def test_overwrite_warns(self, caplog):
        import logging
        r = ModelRegistry()
        r.register("custom", _stub_factory)
        with caplog.at_level(logging.WARNING):
            r.register("custom", _stub_factory)
        assert "Overwriting" in caplog.text

    def test_unknown_raises_key_error(self):
        r = ModelRegistry()
        with pytest.raises(KeyError, match="unknown_model"):
            r.load("unknown_model")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestLoading:
    def test_load_calls_load_model(self):
        r = ModelRegistry()
        r.register("m", _stub_factory)
        model = r.load("m")
        assert model.is_loaded
        assert model.load_called == 1

    def test_cached_after_first_load(self):
        r = ModelRegistry()
        r.register("m", _stub_factory)
        m1 = r.load("m")
        m2 = r.load("m")
        assert m1 is m2

    def test_force_reload_creates_new_instance(self):
        r = ModelRegistry()
        r.register("m", _stub_factory)
        m1 = r.load("m")
        m2 = r.load("m", force_reload=True)
        assert m1 is not m2

    def test_kwargs_passed_to_factory(self):
        r = ModelRegistry()
        r.register("m", _stub_factory)
        model = r.load("m", name="customname")
        assert model.name == "customname"

    def test_is_loaded_after_load(self):
        r = ModelRegistry()
        r.register("m", _stub_factory)
        assert not r.is_loaded("m")
        r.load("m")
        assert r.is_loaded("m")


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

class TestAccessors:
    def test_get_before_load_returns_none(self):
        r = ModelRegistry()
        r.register("m", _stub_factory)
        assert r.get("m") is None

    def test_get_after_load_returns_instance(self):
        r = ModelRegistry()
        r.register("m", _stub_factory)
        r.load("m")
        assert r.get("m") is not None

    def test_loaded_names_empty_initially(self):
        r = ModelRegistry()
        assert r.loaded_names() == []

    def test_loaded_names_populated_after_load(self):
        r = ModelRegistry()
        r.register("m", _stub_factory)
        r.load("m")
        assert "m" in r.loaded_names()


# ---------------------------------------------------------------------------
# Unloading
# ---------------------------------------------------------------------------

class TestUnloading:
    def test_unload_removes_from_cache(self):
        r = ModelRegistry()
        r.register("m", _stub_factory)
        r.load("m")
        r.unload("m")
        assert not r.is_loaded("m")
        assert r.get("m") is None

    def test_unload_nonexistent_is_safe(self):
        r = ModelRegistry()
        r.unload("nonexistent")  # should not raise

    def test_can_reload_after_unload(self):
        r = ModelRegistry()
        r.register("m", _stub_factory)
        r.load("m")
        r.unload("m")
        m2 = r.load("m")
        assert m2.is_loaded


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

class TestGlobalRegistry:
    def test_get_registry_returns_same_instance(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_default_models_in_global_registry(self):
        r = get_registry()
        for name in ("phoneme", "pitch", "onset_offset"):
            assert name in r.registered_names()


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_registered_names(self):
        r = ModelRegistry()
        r.register("test_m", _stub_factory)
        rep = repr(r)
        assert "test_m" in rep
