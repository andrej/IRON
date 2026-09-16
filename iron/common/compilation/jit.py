# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Bridge from IRON's artifact description to mlir-aie's ``CompilableDesign``.

``CompilableDesign`` compiles a design by running a generator function, then
compiling every ``ExternalFunction`` that the generator registered while it
ran. IRON's designs declare their kernels as plain ``Kernel`` objects, which
name an object file but do not say how to build it; the build instructions sit
in the operator's kernel artifacts instead. :func:`build_design` joins the two,
so that the design files stay untouched.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import aie.utils.config
from aie import ir
from aie.iron import CompileTime
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.compilabledesign import CompilableDesign

from ..device_utils import get_kernel_dir
from .base import KernelArchiveArtifact

logger = logging.getLogger(__name__)

# Peano's clang warns about this; xchesscc's front end does not recognise the flag.
_PEANO_FLAGS = ["-Wno-missing-template-arg-list-after-template-kw"]


def _link_targets(module) -> dict[str, str]:
    """Every core function the module calls, as symbol name -> object file."""
    targets: dict[str, str] = {}

    def visit(operation):
        attributes = operation.attributes
        if "link_with" in attributes and "sym_name" in attributes:
            symbol = ir.StringAttr(attributes["sym_name"]).value
            targets[symbol] = ir.StringAttr(attributes["link_with"]).value
        for region in operation.regions:
            for block in region.blocks:
                for nested in block.operations:
                    visit(nested.operation)

    visit(module.operation)
    return targets


def _translation_units(artifact) -> list[Path]:
    """The sources that compile, as opposed to the ones they include."""
    if isinstance(artifact, KernelArchiveArtifact):
        return [dep.sources[0] for dep in artifact.dependencies]
    return artifact.sources[:1]


def _runtime_lib_include() -> Path:
    arch = get_kernel_dir()
    return Path(aie.utils.config.root_path()) / "aie_runtime_lib" / arch.upper()


def _defines(symbols, artifact) -> dict[str, str]:
    """The ``-D`` renames that make the object export the names the MLIR calls.

    A design reaches a kernel under a name the source does not spell in two
    cases: a fused sequence prefixes the entry points of each operator, and
    stream-dse suffixes a GEMM's symbols with its tile shape. Renaming through
    the preprocessor covers both, and reaches names built by token pasting.
    """
    source_of = {new: old for old, new in artifact.rename_symbols.items()}
    prefix = artifact.symbol_prefix
    defines: dict[str, str] = {}
    for symbol in symbols:
        name = symbol[len(prefix) :] if prefix and symbol.startswith(prefix) else symbol
        source_name = source_of.get(name, name)
        if source_name != symbol:
            defines[source_name] = symbol
    for old, new in artifact.rename_symbols.items():
        defines.setdefault(old, new)
    return defines


def _declare_kernel(object_file, symbols, artifact, use_chess):
    """Declare the ExternalFunction that builds ``object_file``.

    One declaration covers the whole object, however many symbols the design
    calls in it: mlir-aie compiles one object per ExternalFunction, and a
    second declaration naming the same object would compile it a second time.
    """
    units = _translation_units(artifact)
    include_dirs = [str(_runtime_lib_include())]
    include_dirs += [str(unit.parent) for unit in units]

    flags = list(artifact.extra_flags)
    flags += [
        f"-D{old}={new}" for old, new in sorted(_defines(symbols, artifact).items())
    ]
    if not use_chess:
        flags = _PEANO_FLAGS + flags

    if len(units) == 1:
        source = {"source_file": str(units[0])}
    else:
        source = {"source_string": "\n".join(f'#include "{u}"' for u in units) + "\n"}

    return ExternalFunction(
        Path(object_file).stem,
        object_file_name=object_file,
        arg_types=[],
        include_dirs=include_dirs,
        compile_flags=flags,
        use_chess=use_chess,
        **source,
    )


def declare_kernels(module, kernels, use_chess=False) -> None:
    """Register an ExternalFunction for every object ``module`` links against.

    An object no artifact claims is left alone: it comes from a binary the
    operator supplies itself, such as a prebuilt overlay.
    """
    by_object = {artifact.filename: artifact for artifact in kernels}
    symbols_by_object: dict[str, list[str]] = defaultdict(list)
    for symbol, object_file in sorted(_link_targets(module).items()):
        symbols_by_object[object_file].append(symbol)

    for object_file, symbols in sorted(symbols_by_object.items()):
        artifact = by_object.get(object_file)
        if artifact is None:
            logger.debug("no kernel artifact builds %s", object_file)
            continue
        _declare_kernel(object_file, symbols, artifact, use_chess)

    unused = sorted(set(by_object) - set(symbols_by_object))
    if unused:
        logger.debug("kernel artifacts no design calls into: %s", ", ".join(unused))


def build_design(
    *,
    design: CompileTime[Any],
    kernels: CompileTime[tuple] = (),
    use_chess: CompileTime[bool] = False,
):
    """Generate a design's MLIR and declare the kernels its cores link against."""
    module = design()
    # A design may hand back the text it prints to rather than the module.
    if isinstance(module, str):
        module = ir.Module.parse(module)
    declare_kernels(module, kernels, use_chess)
    return module


def _bridge_sources() -> list[Path]:
    """This package's own Python files.

    They decide how a design is generated and how its kernels are declared, so
    editing one changes the build as surely as editing a design does. mlir-aie
    hashes only the generator function it is handed, which is ``build_design``.
    """
    return sorted(Path(__file__).parent.glob("*.py"))


def build_compilable(
    design,
    kernels=(),
    *,
    use_chess: bool = False,
    full_elf: bool = False,
    aiecc_flags=(),
) -> CompilableDesign:
    """A ``CompilableDesign`` for one operator or one fused sequence.

    ``source_files`` lists the design's Python sources and every C++ kernel
    source, so that editing one invalidates the cached build; mlir-aie hashes
    them by content.
    """
    kernels = tuple(kernels)
    sources = _bridge_sources() + list(design.source_paths)
    for artifact in kernels:
        sources.extend(artifact.sources)
    return CompilableDesign(
        build_design,
        compile_kwargs={
            "design": design,
            "kernels": kernels,
            "use_chess": use_chess,
        },
        source_files=sources,
        aiecc_flags=list(aiecc_flags),
        full_elf=full_elf,
    )
