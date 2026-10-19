"""Reads remembered by the version of the file they were made from.

The plot dialogs each read the output before they can open — the spectra it
holds, its PES constants, its VCI states — and that read is the wait: a few
hundred ms on a 3.5 MB run, more on a large one. Opening the same dialog on the
same output again, to make a second plot, read it all again. Here the result is
kept, and handed back while the file is the one it was read from.

"The one it was read from" is the file's real path, its size and its
modification time, to the nanosecond — the version of the file, never what it
is called. A file written again since is read again; one reached by another
route (a link, a relative path) is the same file. Only a handful of whole reads
are kept, since they hold the parsed output, lines and all; small facts read
from a file — its Fermi level, how many bands it has — are kept in their dozens.

Thread-safe, because the reads themselves happen on a worker.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Any, Callable, Optional

#: What :func:`get` returns for a read it does not have. ``None`` cannot be it:
#: "this output holds no VCI states" is an answer worth remembering too.
MISSING = object()

_KEPT = {False: 4, True: 64}           # light -> how many are kept
_stores = {False: OrderedDict(), True: OrderedDict()}
_lock = threading.Lock()


def key(kind: str, *paths: str) -> Optional[tuple]:
    """What a read of ``kind`` from ``paths`` is remembered by, or ``None``.

    ``None`` when a file cannot be looked at: then nothing is remembered, and
    the read goes ahead and reports whatever is wrong with it.
    """
    parts: list = [kind]
    for path in paths:
        try:
            status = os.stat(path)
        except OSError:
            return None
        parts.append((os.path.realpath(path), status.st_size, status.st_mtime_ns))
    return tuple(parts)


def get(read_key: Optional[tuple], light: bool = False) -> Any:
    """The remembered result for ``read_key``, or :data:`MISSING`."""
    if read_key is None:
        return MISSING
    store = _stores[light]
    with _lock:
        if read_key not in store:
            return MISSING
        store.move_to_end(read_key)
        return store[read_key]


def put(read_key: Optional[tuple], value: Any, light: bool = False) -> None:
    """Remember ``value`` for ``read_key``, letting the oldest go past the limit.

    ``light`` for a small fact about a file rather than the parsed file
    itself: many more of those are kept.
    """
    if read_key is None:
        return
    store = _stores[light]
    with _lock:
        store[read_key] = value
        store.move_to_end(read_key)
        while len(store) > _KEPT[light]:
            store.popitem(last=False)


def read(kind: str, paths: tuple, reader: Callable[[], Any], light: bool = False) -> Any:
    """``reader()``, or what it returned last time for this version of ``paths``.

    A read that raises is not remembered: it is tried again next time, and
    says again what is wrong.
    """
    read_key = key(kind, *paths)
    found = get(read_key, light)
    if found is MISSING:
        found = reader()
        put(read_key, found, light)
    return found


def forget() -> None:
    """Remember nothing (for tests, and for nothing else)."""
    with _lock:
        for store in _stores.values():
            store.clear()


__all__ = ["MISSING", "forget", "get", "key", "put", "read"]
