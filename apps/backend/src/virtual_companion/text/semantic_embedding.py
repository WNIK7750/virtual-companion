"""Host adapter for an explicitly pinned, locally installed FastEmbed model."""

import json
from hashlib import sha256
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path, PurePosixPath


class FastEmbedEncoder:
    """Small FastEmbed adapter that verifies a pinned local model before loading it."""

    def __init__(self, model, *, model_manifest_sha256, fastembed_version):
        self._model = model
        self.model_manifest_sha256 = model_manifest_sha256
        self.fastembed_version = fastembed_version

    @property
    def identity(self):
        return {
            "adapter": "fastembed-text.v1",
            "distribution": "fastembed",
            "version": self.fastembed_version,
            "execution_provider": "CPUExecutionProvider",
        }

    @classmethod
    def from_local_manifest(cls, model_directory, manifest_path, *, model_id,
                            model_revision, manifest_sha256, threads=2):
        """Load a verified FastEmbed ONNX directory with networking disabled.

        The manifest's file list and digest are the artifact identity. The model
        directory path is only a host setting and is deliberately not returned
        as a source identity.
        """
        if (not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64
                or any(character not in "0123456789abcdef" for character in manifest_sha256)):
            raise ValueError("Expected pinned model manifest SHA-256 required")
        manifest_file = Path(manifest_path).resolve(strict=True)
        manifest_bytes = manifest_file.read_bytes()
        if sha256(manifest_bytes).hexdigest() != manifest_sha256:
            raise ValueError("Local model manifest digest differs from configured identity")
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Local model manifest is not valid UTF-8 JSON") from error
        if (not isinstance(manifest, dict) or manifest.get("model") != model_id
                or manifest.get("revision") != model_revision):
            raise ValueError("Local model manifest identity differs from configured model")

        root = Path(model_directory).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Local model directory is not a directory")
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("Local model manifest has no files")
        verified_names = set()
        for entry in files:
            if not isinstance(entry, dict):
                raise ValueError("Local model file manifest entry must be a mapping")
            relative = entry.get("file")
            expected_size = entry.get("bytes")
            expected_hash = entry.get("sha256")
            if (not isinstance(relative, str) or not relative
                    or type(expected_size) is not int or expected_size < 0
                    or not isinstance(expected_hash, str) or len(expected_hash) != 64
                    or any(character not in "0123456789abcdef" for character in expected_hash)):
                raise ValueError("Local model file manifest entry is malformed")
            pure = PurePosixPath(relative)
            if ("\\" in relative or ":" in relative or pure.is_absolute()
                    or any(part in {"", ".", ".."} for part in pure.parts)):
                raise ValueError("Local model manifest path must remain inside its directory")
            local_file = root.joinpath(*pure.parts).resolve(strict=True)
            if root not in local_file.parents or not local_file.is_file():
                raise ValueError("Local model manifest path escapes its directory")
            if relative in verified_names:
                raise ValueError("Local model manifest contains a duplicate path")
            verified_names.add(relative)
            content = local_file.read_bytes()
            if len(content) != expected_size or sha256(content).hexdigest() != expected_hash:
                raise ValueError(f"Local model file failed manifest verification: {relative}")

        try:
            from fastembed import TextEmbedding
        except ImportError as error:
            raise RuntimeError("Optional FastEmbed semantic dependency is unavailable") from error
        try:
            installed_version = package_version("fastembed")
        except PackageNotFoundError as error:
            raise RuntimeError("FastEmbed package version is unavailable") from error
        if installed_version != "0.9.0":
            raise RuntimeError("Pinned FastEmbed adapter requires version 0.9.0")
        model = TextEmbedding(
            model_name=model_id,
            specific_model_path=str(root),
            local_files_only=True,
            threads=threads,
            providers=["CPUExecutionProvider"],
        )
        return cls(model, model_manifest_sha256=manifest_sha256,
                   fastembed_version=installed_version)

    def query_embed(self, query):
        return self._model.query_embed(query)

    def embed(self, texts):
        return self._model.embed(texts)
