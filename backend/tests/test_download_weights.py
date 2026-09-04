import hashlib
import io
from pathlib import Path
from urllib.error import HTTPError

import pytest

from scripts import download_weights as weights
from scripts import pin_dinov2


@pytest.mark.parametrize("failure", ["404", "checksum", "interrupted", "oversized"])
def test_download_failure_preserves_existing_file_and_cleans_temp(tmp_path, monkeypatch, failure):
    target = tmp_path / "mobile_sam.pt"
    target.write_bytes(b"old")
    def download(*args, **kwargs):
        if failure == "404":
            raise HTTPError("https://example.test", 404, "Not Found", {}, None)
        if failure == "interrupted":
            class Interrupted(io.BytesIO):
                def read(self, *args):
                    raise OSError("connection interrupted")
            return Interrupted()
        return io.BytesIO(b"Not Found")
    monkeypatch.setattr(weights, "urlopen", download)
    if failure == "oversized":
        monkeypatch.setattr(weights, "MOBILE_SAM_MAX_BYTES", 3)
    with pytest.raises((OSError, ValueError)):
        weights.install_checkpoint(target)
    assert target.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [target]


def test_download_repairs_corrupt_cache_then_reuses_verified_weights(tmp_path, monkeypatch):
    target = tmp_path / "mobile_sam.pt"
    target.write_bytes(b"Not Found")
    data = b"verified checkpoint"
    monkeypatch.setattr(weights, "MOBILE_SAM_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(weights, "urlopen", lambda *args, **kwargs: io.BytesIO(data))
    assert weights.install_checkpoint(target) == target
    monkeypatch.setattr(weights, "urlopen", lambda *args, **kwargs: pytest.fail("must reuse verified cache"))
    assert weights.install_checkpoint(target) == target
    assert target.read_bytes() == data


@pytest.mark.parametrize("interrupted", [False, True])
def test_dinov2_repairs_corrupt_cache_atomically(tmp_path, monkeypatch, interrupted):
    import torch
    target = tmp_path / "checkpoints/dinov2_vits14_pretrain.pth"
    target.parent.mkdir()
    target.write_bytes(b"corrupt")
    data = b"verified DINOv2 checkpoint"
    expected = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(pin_dinov2, "DINOV2_WEIGHTS_SHA256", expected)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: None)
    def download(url, destination, hash_prefix):
        assert hash_prefix == expected
        Path(destination).write_bytes(data)
        if interrupted:
            raise OSError("interrupted")
    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    if interrupted:
        with pytest.raises(OSError, match="interrupted"):
            pin_dinov2.download_backbone()
        assert target.read_bytes() == b"corrupt"
    else:
        assert pin_dinov2.download_backbone()[1] == target
        monkeypatch.setattr(torch.hub, "download_url_to_file",
                            lambda *a, **kw: pytest.fail("must reuse verified cache"))
        pin_dinov2.download_backbone()
        assert target.read_bytes() == data
    assert list(target.parent.iterdir()) == [target]
