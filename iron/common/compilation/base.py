# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Declarative description of what an operator compiles.

An operator describes its build with two things: a :class:`DesignGenerator`,
which produces the MLIR module, and a list of kernel artifacts naming the
object files its cores link against. The artifacts here hold data only.

:mod:`iron.common.compilation.jit` turns that description into an mlir-aie
``CompilableDesign``, which plans the build, tracks dependencies by content,
caches the result and runs the compiler.

Every artifact feeds a cache key, so each one carries an explicit ``__repr__``.
The default ``repr`` of an object embeds its address, which differs between
processes, and a key built from it would never hit the cache.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ADDRESS = re.compile(r" (?:object )?at 0x[0-9a-fA-F]+")


def stable_repr(value: Any) -> str:
    """``repr(value)``, guaranteed equal across processes.

    Raises for a value whose ``repr`` embeds an address, rather than returning
    a key that moves on every run and defeats the compilation cache.
    """
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return repr(value)
    if isinstance(value, Path):
        return repr(str(value))
    if isinstance(value, (list, tuple)):
        items = ", ".join(stable_repr(item) for item in value)
        return f"[{items}]" if isinstance(value, list) else f"({items})"
    if isinstance(value, dict):
        items = ", ".join(
            f"{stable_repr(k)}: {stable_repr(v)}" for k, v in sorted(value.items())
        )
        return f"{{{items}}}"
    # An IRON device: identify it by target and grid rather than by address.
    if hasattr(value, "resolve") and hasattr(value, "cols") and hasattr(value, "rows"):
        return (
            f"{type(value).__name__}({value.resolve().name},{value.cols}x{value.rows})"
        )
    text = repr(value)
    if _ADDRESS.search(text):
        raise TypeError(
            f"{type(value).__name__} has no stable repr ({text!r}), so it cannot "
            "identify a compiled design. Pass a name or another value that "
            "describes it, or give the class a __repr__."
        )
    return text


@dataclass
class DesignGenerator:
    """Imports ``source_path`` and calls ``fn_name``, returning an MLIR module.

    The design module is re-imported on every call. Designs build MLIR into the
    ambient context, so a module cached from an earlier call would hand out
    operations belonging to a context that has since been discarded.
    """

    source_path: Path
    fn_name: str
    args: tuple = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __call__(self):
        spec = importlib.util.spec_from_file_location(
            self.source_path.name, self.source_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, self.fn_name)(*self.args, **self.kwargs)

    def __repr__(self) -> str:
        return (
            f"DesignGenerator({self.source_path.name}:{self.fn_name}, "
            f"{stable_repr(self.args)}, {stable_repr(self.kwargs)})"
        )

    @property
    def source_paths(self) -> list[Path]:
        """The Python files whose content decides what this design generates."""
        return [Path(self.source_path)]


@dataclass
class SourceArtifact:
    """A C/C++ source file on disk."""

    filename: str | Path

    def __repr__(self) -> str:
        return f"Source({self.filename})"

    @property
    def path(self) -> Path:
        return Path(self.filename)


@dataclass
class KernelObjectArtifact:
    """One object file, compiled from ``dependencies[0]`` as a translation unit.

    Further dependencies are sources that the first one includes. They do not
    compile on their own; listing them makes editing one rebuild the object.

    ``rename_symbols`` maps a symbol as the source spells it to the name the
    MLIR calls it by. Each entry becomes a ``-D`` for the preprocessor, which
    reaches names built by token pasting as well as names written out in full.

    ``symbol_prefix`` prefixes the entry points a design calls into. A fused
    sequence gives each operator its own, so that the objects of two operators
    built from one source stay apart.
    """

    filename: str
    dependencies: list[SourceArtifact] = field(default_factory=list)
    extra_flags: list[str] = field(default_factory=list)
    rename_symbols: dict[str, str] = field(default_factory=dict)
    symbol_prefix: str = ""

    def __repr__(self) -> str:
        return (
            f"KernelObject({self.filename}, {stable_repr(self.dependencies)}, "
            f"{stable_repr(self.extra_flags)}, {stable_repr(self.rename_symbols)}, "
            f"{self.symbol_prefix})"
        )

    @property
    def sources(self) -> list[Path]:
        return [dep.path for dep in self.dependencies]


@dataclass
class KernelArchiveArtifact:
    """One object file built from several sources in a single translation unit.

    Each dependency contributes its own compile flags, so the flags of all of
    them apply to the combined unit.
    """

    filename: str
    dependencies: list[KernelObjectArtifact] = field(default_factory=list)
    symbol_prefix: str = ""

    def __repr__(self) -> str:
        return (
            f"KernelArchive({self.filename}, {stable_repr(self.dependencies)}, "
            f"{self.symbol_prefix})"
        )

    @property
    def sources(self) -> list[Path]:
        return [source for dep in self.dependencies for source in dep.sources]

    @property
    def extra_flags(self) -> list[str]:
        return [flag for dep in self.dependencies for flag in dep.extra_flags]

    @property
    def rename_symbols(self) -> dict[str, str]:
        renames: dict[str, str] = {}
        for dep in self.dependencies:
            renames.update(dep.rename_symbols)
        return renames


@dataclass
class PythonGeneratedMLIRArtifact:
    """An MLIR module produced by a Python design function."""

    filename: str
    generator: DesignGenerator

    def __repr__(self) -> str:
        return f"PythonGeneratedMLIR({self.generator!r})"


def _sha256_of(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


@dataclass
class RemoteFileArtifact:
    """A file downloaded from a URL and pinned by its SHA-256 digest.

    The digest pins the content, so ``url`` must name an immutable revision of
    the file -- a commit SHA rather than a branch.
    """

    filename: str
    url: str
    sha256: str

    def __repr__(self) -> str:
        return f"RemoteFile({self.filename}, {self.sha256})"

    def fetch(self, directory: Path) -> Path:
        """Return the local path, downloading the file if it is absent."""
        target = Path(directory) / self.filename
        if target.exists() and _sha256_of(target) == self.sha256:
            return target
        if not self.url.startswith("https://"):
            raise ValueError(f"refusing to download over {self.url!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Download beside the target and rename, so an interrupted fetch cannot
        # leave a truncated file that a later run reports as a digest mismatch.
        partial_path = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(self.url, timeout=60) as response:
            partial_path.write_bytes(response.read())
        digest = _sha256_of(partial_path)
        if digest != self.sha256:
            partial_path.unlink()
            raise RuntimeError(
                f"{self.url} has SHA-256 {digest}, expected {self.sha256}"
            )
        partial_path.replace(target)
        return target
