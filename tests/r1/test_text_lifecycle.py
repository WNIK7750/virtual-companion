from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows Job Object lifecycle")
ROOT = Path(__file__).resolve().parents[2]


def launch(data_root):
    environment = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "apps/backend/src"),
        PYTHONUTF8="1",
        VC_DATA_ROOT=str(data_root),
    )
    process = subprocess.Popen(  # noqa: S603
        [sys.executable, "-u", "-m", "virtual_companion.text"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    process.stdin.write(secrets.token_hex(32) + "\n")
    process.stdin.flush()
    return process


def test_parent_pipe_eof_shuts_down_job_and_releases_port(tmp_path):
    process = launch(tmp_path)
    try:
        receipt = json.loads(process.stdout.readline())
        assert receipt["job"] is True
        assert receipt["pid"] > 0  # Windows venv uses a separate Python launcher.
        process.stdin.close()
        assert process.wait(timeout=5) == 0
        with socket.socket() as probe:
            assert probe.connect_ex(("127.0.0.1", 18084)) != 0
    finally:
        if not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_port_conflict_fails_without_killing_port_owner(tmp_path):
    with socket.socket() as owner:
        owner.bind(("127.0.0.1", 18084))
        owner.listen()
        process = launch(tmp_path)
        try:
            assert process.wait(timeout=5) != 0
            assert process.stdout.read() == ""
            assert owner.getsockname()[1] == 18084
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
