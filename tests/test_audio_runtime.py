import threading

from src.hardware.audio_feedback import AudioFeedback


class _Engine:
    def __init__(self):
        self.text = ""

    def say(self, text):
        self.text = text

    def runAndWait(self):
        return None


def test_speech_completion_event_is_set_after_the_message_is_spoken():
    audio = AudioFeedback()
    audio._ready = True
    audio._running = True
    audio._engine = _Engine()
    audio._use_worker_sapi = False

    event = audio.announce_banknote("50_EGP")
    assert event is not None
    assert not event.is_set()

    worker = threading.Thread(target=audio._speech_worker)
    worker.start()
    try:
        assert event.wait(timeout=1.0)
    finally:
        audio._running = False
        worker.join(timeout=1.0)

    assert audio._engine.text == "50 Egyptian pounds"
