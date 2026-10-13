"""Current-user Windows DPAPI secret references, bound to the selected endpoint.

Only ciphertext is persisted. This protects secrets at rest, not against code
already running as the same Windows user. No plaintext fallback or key export.
"""

import ctypes
import os
import re
from ctypes import wintypes
from hashlib import sha256
from pathlib import Path
from uuid import uuid4


class _Blob(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def _blob(data):
    buffer = ctypes.create_string_buffer(data)
    return _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer


def _crypt(data, endpoint, *, decrypt):
    if os.name != "nt":
        raise ValueError("Windows DPAPI is unavailable; no plaintext fallback")
    if not isinstance(endpoint, str) or not endpoint or len(endpoint) > 2048:
        raise ValueError("Explicit secret endpoint binding required")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = crypt32.CryptUnprotectData if decrypt else crypt32.CryptProtectData
    function.argtypes = [
        ctypes.POINTER(_Blob),
        ctypes.c_void_p,
        ctypes.POINTER(_Blob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_Blob),
    ]
    function.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    source, source_buffer = _blob(data)
    entropy, entropy_buffer = _blob(
        sha256(("virtual-companion.model-key.v1:" + endpoint).encode()).digest()
    )
    output = _Blob()
    try:
        # CRYPTPROTECT_UI_FORBIDDEN; never CRYPTPROTECT_LOCAL_MACHINE.
        if not function(
            ctypes.byref(source), None, ctypes.byref(entropy), None, None, 1, ctypes.byref(output)
        ):
            raise ValueError("Cannot decrypt secret" if decrypt else "Cannot protect secret")
        return ctypes.string_at(output.data, output.size)
    finally:
        ctypes.memset(source_buffer, 0, len(source_buffer))
        ctypes.memset(entropy_buffer, 0, len(entropy_buffer))
        if output.data:
            ctypes.memset(output.data, 0, output.size)
            kernel32.LocalFree(ctypes.cast(output.data, ctypes.c_void_p))


class WindowsSecretStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if any((path / ".git").exists() for path in (self.root, *self.root.parents)):
            raise ValueError("Managed secrets must be outside Git workspaces")
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, ref):
        if not isinstance(ref, str) or not re.fullmatch(r"dpapi:[0-9a-f]{32}", ref):
            raise ValueError("Invalid managed secret reference")
        path = self.root / (ref[6:] + ".bin")
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Invalid managed secret reference")
        return path

    def put(self, value, endpoint):
        if (
            not isinstance(value, str)
            or not 1 <= len(value) <= 8192
            or any(not 33 <= ord(c) <= 126 for c in value)
        ):
            raise ValueError("Bounded nonempty credential required")
        encrypted = _crypt(value.encode(), endpoint, decrypt=False)
        ref = "dpapi:" + uuid4().hex
        with self._path(ref).open("xb") as file:
            file.write(encrypted)
            file.flush()
            os.fsync(file.fileno())
        return ref

    def get(self, ref, endpoint):
        path = self._path(ref)
        try:
            with path.open("rb") as file:
                encrypted = file.read(65537)
        except OSError:
            raise ValueError("Managed secret unavailable") from None
        if not encrypted or len(encrypted) > 65536:
            raise ValueError("Invalid protected credential")
        return _crypt(encrypted, endpoint, decrypt=True).decode("utf-8")

    def remove(self, ref):
        # Removing the chosen secret is explicit; configs retaining its reference fail.
        self._path(ref).unlink(missing_ok=True)
