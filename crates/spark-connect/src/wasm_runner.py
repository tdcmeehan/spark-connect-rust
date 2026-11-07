#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""WASM UDF runners, run by Spark's Python workers (``spark_connect::wasm_udf``).

The Rust client embeds this source in each UDF's pickled ``command``. When the
worker loads the command, it compiles this source into a fresh namespace and
calls ``_spark_connect_wasm_udf_command``, which returns the ``(runner,
return_type)`` pair a Python UDF's command holds. So the client needs no Python,
and the executors need only Spark's Python worker, ``wasmtime``, and (for batch
UDFs) ``pyarrow``.

The per-row binary ABI (little-endian, mirroring ``spark_connect::wasm_udf::AbiType``):

* ``i32``/``f32``: 4 bytes; ``i64``/``f64``: 8 bytes; ``bool``: 1 byte.
* ``string``: ``u32`` byte-length then UTF-8; ``binary``: ``u32`` length then bytes.
* ``array:<T>``: ``u32`` count then each element; ``option:<T>``: 1 tag byte
  (0=None, 1=Some) then the value if present.
* Arguments are concatenated into one buffer; the result is a single value.

The module exports ``spark_udf_alloc(size, align)`` / ``spark_udf_free(ptr, size,
align)`` and each UDF as ``fn(args_ptr, args_len) -> (ptr << 32 | len)`` of the result.

The per-batch path is a host of ``spark_udf.h``: the module's library comes from
``spark_udf_init``, and each batch is passed as Arrow C data arrays laid out in the
module's memory (32-bit pointers), so any language that targets the header works.
"""

import ctypes
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple


# --- binary codec (the exact contract mirrored by the Rust runtime) ----------


def _split(desc: str) -> Tuple[str, str]:
    """Split ``"array:option:i64"`` into ``("array", "option:i64")``."""
    head, _, tail = desc.partition(":")
    return head, tail


def _encode(out: bytearray, desc: str, value: Any) -> None:
    kind, inner = _split(desc)
    if kind == "option":
        if value is None:
            out.append(0)
        else:
            out.append(1)
            _encode(out, inner, value)
        return
    if value is None:
        raise ValueError(
            f"null value for non-nullable WASM UDF argument of type {desc!r}; "
            f"use Option<...> in the Rust signature"
        )
    if kind == "i32":
        out.extend(struct.pack("<i", int(value)))
    elif kind == "i64":
        out.extend(struct.pack("<q", int(value)))
    elif kind == "f32":
        out.extend(struct.pack("<f", float(value)))
    elif kind == "f64":
        out.extend(struct.pack("<d", float(value)))
    elif kind == "bool":
        out.append(1 if value else 0)
    elif kind == "string":
        b = str(value).encode("utf-8")
        out.extend(struct.pack("<I", len(b)))
        out.extend(b)
    elif kind == "binary":
        b = bytes(value)
        out.extend(struct.pack("<I", len(b)))
        out.extend(b)
    elif kind == "array":
        out.extend(struct.pack("<I", len(value)))
        for elem in value:
            _encode(out, inner, elem)
    else:
        raise ValueError(f"unsupported WASM ABI type: {desc!r}")


def _decode(buf: bytes, off: int, desc: str) -> Tuple[Any, int]:
    kind, inner = _split(desc)
    if kind == "option":
        tag = buf[off]
        off += 1
        if tag == 0:
            return None, off
        return _decode(buf, off, inner)
    if kind == "i32":
        return struct.unpack_from("<i", buf, off)[0], off + 4
    if kind == "i64":
        return struct.unpack_from("<q", buf, off)[0], off + 8
    if kind == "f32":
        return struct.unpack_from("<f", buf, off)[0], off + 4
    if kind == "f64":
        return struct.unpack_from("<d", buf, off)[0], off + 8
    if kind == "bool":
        return (buf[off] != 0), off + 1
    if kind == "string":
        (n,) = struct.unpack_from("<I", buf, off)
        off += 4
        return buf[off : off + n].decode("utf-8"), off + n
    if kind == "binary":
        (n,) = struct.unpack_from("<I", buf, off)
        off += 4
        return bytes(buf[off : off + n]), off + n
    if kind == "array":
        (n,) = struct.unpack_from("<I", buf, off)
        off += 4
        items = []
        for _ in range(n):
            item, off = _decode(buf, off, inner)
            items.append(item)
        return items, off
    raise ValueError(f"unsupported WASM ABI type: {desc!r}")


def encode_args(arg_types: List[str], args: Tuple[Any, ...]) -> bytes:
    """Encode a row's arguments into the ABI byte buffer."""
    out = bytearray()
    for desc, value in zip(arg_types, args):
        _encode(out, desc, value)
    return bytes(out)


def decode_value(ret_type: str, buf: bytes) -> Any:
    """Decode a single ABI-encoded value (the result buffer)."""
    value, _ = _decode(buf, 0, ret_type)
    return value


# --- the per-row callable -----------------------------------------------------


class WasmScalarUDF:
    """A per-row callable that runs a WASM export via ``wasmtime``.

    The wasmtime state is created lazily, on the first call.
    """

    def __init__(
        self,
        wasm: bytes,
        entrypoint: str,
        arg_types: List[str],
        ret_type: str,
    ) -> None:
        self.wasm = wasm
        self.entrypoint = entrypoint
        self.arg_types = list(arg_types)
        self.ret_type = ret_type
        self._rt = None  # lazily-built wasmtime state

    def _ensure(self):
        if self._rt is not None:
            return self._rt
        try:
            import wasmtime
        except ImportError as exc:  # pragma: no cover - depends on executor env
            raise ImportError(
                "The 'wasmtime' package is required to run WASM UDFs on the "
                "Spark executors. Install it in the workers' Python env."
            ) from exc

        engine = wasmtime.Engine()
        module = wasmtime.Module(engine, self.wasm)
        store = wasmtime.Store(engine)
        instance = wasmtime.Instance(store, module, [])
        exports = instance.exports(store)

        def need(name):
            e = exports.get(name)
            if e is None:
                raise ValueError(f"WASM module does not export '{name}'")
            return e

        self._rt = {
            "store": store,
            "memory": need("memory"),
            "alloc": need("spark_udf_alloc"),
            "free": need("spark_udf_free"),
            "entry": need(self.entrypoint),
        }
        return self._rt

    def __call__(self, *args: Any) -> Any:
        if len(args) != len(self.arg_types):
            raise ValueError(
                f"WASM UDF '{self.entrypoint}' expected {len(self.arg_types)} "
                f"argument(s), got {len(args)}"
            )
        rt = self._ensure()
        store, memory = rt["store"], rt["memory"]
        alloc, free, entry = rt["alloc"], rt["free"], rt["entry"]

        buf = encode_args(self.arg_types, args)
        args_ptr = alloc(store, len(buf), 1)
        if buf:
            memory.write(store, buf, args_ptr)

        packed = entry(store, args_ptr, len(buf)) & 0xFFFFFFFFFFFFFFFF
        res_ptr = (packed >> 32) & 0xFFFFFFFF
        res_len = packed & 0xFFFFFFFF

        res = bytes(memory.read(store, res_ptr, res_ptr + res_len))
        value = decode_value(self.ret_type, res)

        free(store, args_ptr, len(buf), 1)
        free(store, res_ptr, res_len, 1)
        return value


ARRAY = 64  # sizeof(ArrowArray) on wasm32
OK = 0

# Argument descriptors of the client (spark_connect::wasm_udf::AbiType) -> Arrow format.
FORMATS = {
    "i32": "i",
    "i64": "l",
    "f32": "f",
    "f64": "g",
    "bool": "b",
    "string": "u",
    "binary": "z",
}
WIDTHS = {"i": 4, "l": 8, "f": 4, "g": 8}


def _base(desc: str) -> str:
    return desc[len("option:"):] if desc.startswith("option:") else desc


def _align(n: int, a: int = 8) -> int:
    return (n + a - 1) & ~(a - 1)


class _Module:
    """One instance of a module, shared by every function of it in a worker."""

    def __init__(self, wasm: bytes) -> None:
        try:
            import wasmtime
        except ImportError as exc:  # pragma: no cover - depends on executor env
            raise ImportError(
                "The 'wasmtime' package is required to run WASM UDFs on the "
                "Spark executors. Install it in the workers' Python env."
            ) from exc

        self.store = wasmtime.Store(wasmtime.Engine())
        module = wasmtime.Module(self.store.engine, wasm)
        if module.imports:
            names = ", ".join(f"{i.module}.{i.name}" for i in module.imports)
            raise ValueError(f"spark_udf.h modules take no imports; this one imports {names}")
        ex = wasmtime.Instance(self.store, module, []).exports(self.store)
        self.memory = ex["memory"]
        self.table = ex["__indirect_function_table"]
        self.alloc = ex["spark_udf_alloc"]
        if "_initialize" in ex:
            ex["_initialize"](self.store)
        lib = ex["spark_udf_init"](self.store, 0) & 0xFFFFFFFF
        if lib == 0:
            raise ValueError("spark_udf_init returned NULL")
        _, version, self.last_error, nfun, funs = struct.unpack("<qiIqI", self.read(lib, 28))
        if version != 1:
            raise ValueError(f"unsupported spark_udf.h ABI version {version}")
        self.functions: Dict[str, Tuple[int, int, int]] = {}
        for k in range(nfun):
            _, name, kind, create, eval_, close = struct.unpack("<qIiIII", self.read(funs + 32 * k, 28))
            self.functions[self.cstr(name)] = (create, eval_, close)
        self.scratch = 0
        self.scratch_cap = 0

    def base(self) -> int:
        return ctypes.addressof(self.memory.data_ptr(self.store).contents)

    def read(self, ptr: int, n: int) -> bytes:
        return ctypes.string_at(self.base() + ptr, n)

    def write(self, ptr: int, data: bytes) -> None:
        ctypes.memmove(self.base() + ptr, data, len(data))

    def cstr(self, ptr: int) -> str:
        return ctypes.string_at(self.base() + ptr).decode("utf-8", "replace")

    def call(self, index: int, *args: int) -> int:
        return self.table.get(self.store, index)(self.store, *args)

    def malloc(self, n: int) -> int:
        p = self.alloc(self.store, n, 8) & 0xFFFFFFFF
        if p == 0:
            raise MemoryError(f"spark_udf_alloc({n}) failed")
        return p

    def region(self, n: int) -> int:
        if n > self.scratch_cap:
            self.scratch = self.malloc(n)
            self.scratch_cap = n
        return self.scratch

    def error(self, what: str) -> ValueError:
        p = self.call(self.last_error) & 0xFFFFFFFF
        return ValueError(f"{what}: {self.cstr(p) if p else 'unknown error'}")


def _module(wasm: bytes) -> _Module:
    import hashlib

    cache = sys.__dict__.setdefault("_spark_udf_wasm_modules", {})
    key = hashlib.sha256(wasm).hexdigest()
    if key not in cache:
        cache[key] = _Module(wasm)
    return cache[key]


class WasmArrowUDF:
    """A per-batch callable for a scalar Arrow UDF (``SQL_SCALAR_ARROW_UDF``)."""

    def __init__(self, wasm: bytes, entrypoint: str, arg_types: List[str], ret_type: str) -> None:
        self.wasm = wasm
        self.entrypoint = entrypoint
        self.arg_types = list(arg_types)
        self.ret_type = ret_type
        self._fn: Optional[Tuple[_Module, int, int]] = None

    def _ensure(self) -> Tuple[_Module, int, int]:
        if self._fn is not None:
            return self._fn
        m = _module(self.wasm)
        if self.entrypoint not in m.functions:
            raise ValueError(f"WASM module has no spark_udf.h function '{self.entrypoint}'")
        create, eval_, _ = m.functions[self.entrypoint]
        fmts = [FORMATS[_base(d)].encode() + b"\0" for d in self.arg_types]
        ret = FORMATS[_base(self.ret_type)].encode() + b"\0"
        size = 24 + 4 * len(fmts) + 4 + sum(len(f) for f in fmts) + len(ret)
        info = m.malloc(_align(size))
        ptrs, off = [], info + 24 + 4 * len(fmts) + 4
        for f in fmts + [ret]:
            m.write(off, f)
            ptrs.append(off)
            off += len(f)
        state_out = info + 24 + 4 * len(fmts)
        m.write(info + 24, struct.pack("<%dI" % len(fmts), *ptrs[:-1]))
        m.write(info, struct.pack("<qqII", 24, len(fmts), info + 24, ptrs[-1]))
        m.write(state_out, b"\0\0\0\0")
        if m.call(create, info, state_out) != OK:
            raise m.error(f"creating WASM UDF '{self.entrypoint}'")
        state = struct.unpack("<I", m.read(state_out, 4))[0]
        self._fn = (m, eval_, state)
        return self._fn

    def __call__(self, *arrays: Any) -> Any:
        import pyarrow as pa

        out_fmt = FORMATS[_base(self.ret_type)]
        out_type = {"i": pa.int32(), "l": pa.int64(), "f": pa.float32(), "g": pa.float64(),
                    "b": pa.bool_(), "u": pa.string(), "z": pa.binary()}[out_fmt]
        if len(arrays) != len(self.arg_types):
            raise ValueError(f"WASM UDF '{self.entrypoint}' expected {len(self.arg_types)} "
                             f"argument(s), got {len(arrays)}")
        cols = []
        for a in arrays:
            if isinstance(a, pa.ChunkedArray):
                a = a.combine_chunks()
            if pa.types.is_large_string(a.type):
                a = a.cast(pa.string())
            elif pa.types.is_large_binary(a.type):
                a = a.cast(pa.binary())
            cols.append(a)
        n = len(cols[0]) if cols else 0

        # Layout: parent array, child pointers, then per child its array, its
        # buffer pointers and its buffers; then the output array.
        nc = len(cols)
        parent_bufs = ARRAY + _align(4 * nc)
        off = parent_bufs + 8
        plan = []
        for a in cols:
            bufs = a.buffers()
            if a.null_count == 0:
                bufs[0] = None
            arr_at = off
            off += ARRAY + _align(4 * len(bufs))
            slots = []
            for b in bufs:
                if b is None or b.size == 0:
                    slots.append(None)
                else:
                    slots.append((off, b))
                    off = _align(off + b.size)
            plan.append((a, arr_at, slots))
        out_at = off
        m, eval_, state = self._ensure()
        sp = m.region(out_at + ARRAY)

        base = m.base()
        child_ptrs = []
        for a, arr_at, slots in plan:
            ptrs = []
            for s in slots:
                if s is None:
                    ptrs.append(0)
                else:
                    ctypes.memmove(base + sp + s[0], s[1].address, s[1].size)
                    ptrs.append(sp + s[0])
            bufs_at = sp + arr_at + ARRAY
            ctypes.memmove(base + bufs_at, struct.pack("<%dI" % len(ptrs), *ptrs), 4 * len(ptrs))
            hdr = struct.pack("<qqqqqIIIII", len(a), a.null_count, a.offset, len(ptrs), 0,
                              bufs_at, 0, 0, 0, 0)
            ctypes.memmove(base + sp + arr_at, hdr, len(hdr))
            child_ptrs.append(sp + arr_at)
        ctypes.memmove(base + sp + ARRAY, struct.pack("<%dI" % nc, *child_ptrs), 4 * nc)
        # The parent has one buffer (validity), NULL: a record batch has no null rows.
        ctypes.memmove(base + sp + parent_bufs, b"\0" * 8, 8)
        hdr = struct.pack("<qqqqqIIIII", n, 0, 0, 1, nc, sp + parent_bufs, sp + ARRAY, 0, 0, 0)
        ctypes.memmove(base + sp, hdr, len(hdr))

        if m.call(eval_, state, sp, sp + out_at) != OK:
            raise m.error(f"WASM UDF '{self.entrypoint}' failed")

        length, nulls, offset, nbuf, _, bufs_p, _, _, release, _ = struct.unpack(
            "<qqqqqIIIII", m.read(sp + out_at, 60))
        ptrs = struct.unpack("<%dI" % nbuf, m.read(bufs_p, 4 * nbuf))
        end = offset + length

        def copy(ptr: int, size: int):
            buf = pa.allocate_buffer(size)
            if size:
                ctypes.memmove(buf.address, m.base() + ptr, size)
            return buf

        bits = (end + 7) // 8
        out = [copy(ptrs[0], bits) if ptrs[0] and nulls else None]
        if out_fmt in ("u", "z"):
            offsets = copy(ptrs[1], 4 * (end + 1))
            data_len = struct.unpack("<i", offsets.to_pybytes()[4 * end:4 * end + 4])[0]
            out += [offsets, copy(ptrs[2], data_len)]
        elif out_fmt == "b":
            out.append(copy(ptrs[1], bits))
        else:
            out.append(copy(ptrs[1], WIDTHS[out_fmt] * end))
        m.call(release, sp + out_at)
        return pa.Array.from_buffers(out_type, length, out, null_count=nulls, offset=offset)


# --- the command: called when the worker loads the pickle the client wrote ---


def _spark_connect_wasm_udf_command(
    wasm: bytes,
    entrypoint: str,
    arg_types: List[str],
    ret_type: str,
    output_type: str,
    batch: bool,
) -> Tuple[Any, Any]:
    """Return ``(runner, return_type)``; ``output_type`` is Spark's type JSON."""
    from pyspark.sql.types import _parse_datatype_json_string

    runner_cls = WasmArrowUDF if batch else WasmScalarUDF
    runner = runner_cls(wasm, entrypoint, arg_types, ret_type)
    return runner, _parse_datatype_json_string(output_type)
