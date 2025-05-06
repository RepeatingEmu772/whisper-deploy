import pytest
import os
import tempfile
from faster_whisper import WhisperModel

@pytest.fixture(scope="session")
def whisper_model():
    """Create a test Whisper model instance."""
    # Use a smaller model for testing
    model = WhisperModel(
        "base",
        device="cpu",  # Use CPU for testing
        compute_type="int8"
    )
    return model

@pytest.fixture(scope="session")
def test_audio_dir():
    """Create a temporary directory for test audio files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture(scope="session")
def test_audio_file(test_audio_dir):
    """Create a test audio file."""
    # You'll need to provide a small test audio file
    # This is just a placeholder
    test_file = os.path.join(test_audio_dir, "test.mp3")
    with open(test_file, "wb") as f:
        f.write(b"dummy audio data")
    return test_file 