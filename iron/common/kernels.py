# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declaring the C++ entry points a design calls."""

from __future__ import annotations

from pathlib import Path

import aie.utils as aie_utils
from aie.iron import ExternalFunction

from .device_utils import get_kernel_dir

# Peano's clang warns about this; xchesscc's front end does not recognise the flag.
_PEANO_FLAGS = ["-Wno-missing-template-arg-list-after-template-kw"]


def _runtime_lib_include() -> str:
    arch = get_kernel_dir()
    return str(Path(aie_utils.config.root_path()) / "aie_runtime_lib" / arch.upper())


def kernel_object(
    sources,
    symbols,
    *,
    object_name: str | None = None,
    prefix: str = "",
    flags=(),
    use_chess: bool = False,
) -> dict[str, object]:
    """The entry points of one object file, keyed by the name its source spells.

    ``symbols`` maps each entry point to its argument types. The first source
    is the translation unit; the rest are included into it.

    One ``ExternalFunction`` owns the object and the others bind to its
    ``ObjectFile``, so the source compiles once however many entry points a
    design calls. ``prefix`` renames every symbol the object defines, which is
    what keeps two operators built from one source apart inside a fused module.
    """
    sources = [Path(s) for s in sources]
    if object_name is None:
        object_name = f"{sources[0].stem}.o"

    compile_flags = list(flags)
    if not use_chess:
        compile_flags = _PEANO_FLAGS + compile_flags

    include_dirs = [_runtime_lib_include()] + [str(s.parent) for s in sources]

    if len(sources) == 1:
        source = {"source_file": str(sources[0])}
    else:
        source = {"source_string": "".join(f'#include "{s}"\n' for s in sources)}

    names = list(symbols)
    owner = ExternalFunction(
        names[0],
        object_file_name=object_name,
        arg_types=list(symbols[names[0]]),
        include_dirs=include_dirs,
        compile_flags=compile_flags,
        symbol_prefix=prefix or None,
        use_chess=use_chess,
        **source,
    )
    kernels = {names[0]: owner}
    for name in names[1:]:
        kernels[name] = owner.object_file.bind(name, list(symbols[name]))
    return kernels
