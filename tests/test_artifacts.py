import pytest

from coffeecam import server


@pytest.fixture
def client():
    return server.create_app(start_worker=False).test_client()


@pytest.fixture
def artifacts_dir(tmp_path, monkeypatch):
    d = tmp_path / "scratch"
    (d / "sub").mkdir(parents=True)
    (d / "a.gif").write_bytes(b"GIF89a" + b"\0" * 10)
    (d / "notes.txt").write_text("hello")
    (d / "sub" / "b.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 10)
    (d / "secret.pt").write_bytes(b"weights")  # not a served suffix
    monkeypatch.setenv("COFFEECAM_ARTIFACTS_DIR", str(d))
    return d


def test_index_lists_only_served_suffixes(client, artifacts_dir):
    body = client.get("/artifacts").get_data(as_text=True)
    assert "a.gif" in body and "notes.txt" in body and "sub/b.png" in body
    assert "secret.pt" not in body


def test_serves_file_with_mimetype(client, artifacts_dir):
    r = client.get("/artifacts/a.gif")
    assert r.status_code == 200 and r.mimetype == "image/gif"
    assert client.get("/artifacts/sub/b.png").status_code == 200


def test_unsupported_suffix_is_415(client, artifacts_dir):
    assert client.get("/artifacts/secret.pt").status_code == 415


def test_missing_file_404(client, artifacts_dir):
    assert client.get("/artifacts/nope.gif").status_code == 404


def test_path_traversal_blocked(client, artifacts_dir, tmp_path):
    (tmp_path / "outside.gif").write_bytes(b"GIF89a")  # sibling of the scratch dir
    for attempt in (
        "/artifacts/../server.py",
        "/artifacts/..%2f..%2fetc%2fpasswd",
        "/artifacts/..%2foutside.gif",  # served suffix, still must not escape
    ):
        r = client.get(attempt)
        assert r.status_code in (403, 404, 415)
        assert b"GIF89a" not in r.data and b"import" not in r.data


def test_index_404_when_dir_absent(client, tmp_path, monkeypatch):
    monkeypatch.setenv("COFFEECAM_ARTIFACTS_DIR", str(tmp_path / "missing"))
    assert client.get("/artifacts").status_code == 404


def test_index_empty_dir_ok(client, tmp_path, monkeypatch):
    monkeypatch.setenv("COFFEECAM_ARTIFACTS_DIR", str(tmp_path))
    r = client.get("/artifacts")
    assert r.status_code == 200 and "no artifacts yet" in r.get_data(as_text=True)
