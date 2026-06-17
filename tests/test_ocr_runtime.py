from pathlib import Path

from src.perception.ocr import OCRReader, _valid_paddle_inference_dir


def test_arabic_script_detection_handles_real_unicode():
    assert OCRReader._is_predominantly_arabic("مرحبا بالعالم")
    assert OCRReader._is_predominantly_arabic("Shop شارع التحرير")
    assert not OCRReader._is_predominantly_arabic("hello world")


def test_paddle_inference_dir_validation(tmp_path: Path):
    assert not _valid_paddle_inference_dir("")
    assert not _valid_paddle_inference_dir(str(tmp_path))

    (tmp_path / "inference.pdmodel").write_bytes(b"model")
    assert not _valid_paddle_inference_dir(str(tmp_path))

    (tmp_path / "inference.pdiparams").write_bytes(b"params")
    assert _valid_paddle_inference_dir(str(tmp_path))
