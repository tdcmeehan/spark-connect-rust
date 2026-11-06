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
"""WASM UDF packer, embedded in the Rust crate (`spark_connect::wasm_udf`).

The Rust client runs this source as ``python -c <this>`` (so it executes as
``__main__``), feeding a JSON spec on stdin and reading the raw cloudpickled
``command`` bytes from stdout. Because the ``WasmScalarUDF`` runner is defined
here in ``__main__``, cloudpickle serializes it **by value** automatically -- so
neither a separate ``pyspark_wasm_udf`` pip package on the client nor this module
on the executors is required. Executors need only the ``wasmtime`` package (and
Spark's own Python worker) to unpickle and run the ``(WasmScalarUDF, return_type)``
tuple. The only client-side requirement is a Python interpreter with ``pyspark``
(for its vendored ``cloudpickle`` and the pure-Python type parser).

Input JSON (stdin)::

    {
      "wasm_b64":    "<base64 of the .wasm module>",
      "entrypoint":  "add_one",
      "arg_types":   ["i64", "array:string", ...],   # ABI descriptors
      "ret_type":    "i64",
      "output_type": "long"                           # Spark type-JSON value
    }

The binary ABI (little-endian, mirroring ``spark_connect::wasm_udf::AbiType``):

* ``i32``/``f32``: 4 bytes; ``i64``/``f64``: 8 bytes; ``bool``: 1 byte.
* ``string``: ``u32`` byte-length then UTF-8; ``binary``: ``u32`` length then bytes.
* ``array:<T>``: ``u32`` count then each element; ``option:<T>``: 1 tag byte
  (0=None, 1=Some) then the value if present.
* Arguments are concatenated into one buffer; the result is a single value.

The module exports ``spark_udf_alloc(len)->ptr`` / ``spark_udf_dealloc(ptr,len)``
and each UDF as ``fn(args_ptr, args_len) -> (ptr << 32 | len)`` of the result.
"""

import base64
import json
import struct
import sys
from typing import Any, List, Tuple


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


# --- the picklable, per-row callable -----------------------------------------


class WasmScalarUDF:
    """A per-row callable that runs a WASM export via ``wasmtime``.

    Serialized by value with cloudpickle (it is defined in ``__main__``); all
    live wasmtime state is created lazily on the executor (never pickled).
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

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_rt"] = None
        return state

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
            "dealloc": need("spark_udf_dealloc"),
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
        alloc, dealloc, entry = rt["alloc"], rt["dealloc"], rt["entry"]

        buf = encode_args(self.arg_types, args)
        args_ptr = alloc(store, len(buf))
        if buf:
            memory.write(store, buf, args_ptr)

        packed = entry(store, args_ptr, len(buf)) & 0xFFFFFFFFFFFFFFFF
        res_ptr = (packed >> 32) & 0xFFFFFFFF
        res_len = packed & 0xFFFFFFFF

        res = bytes(memory.read(store, res_ptr, res_ptr + res_len))
        value = decode_value(self.ret_type, res)

        dealloc(store, args_ptr, len(buf))
        dealloc(store, res_ptr, res_len)
        return value


class WasmArrowUDF:
    """A per-batch callable for a scalar Arrow UDF (``SQL_SCALAR_ARROW_UDF``).

    Each call gets one ``pyarrow.Array`` per argument, copies their buffers into
    the module's memory, calls the batch export once, and copies the result's
    buffers out. The instance is cached per worker process by the module's hash,
    so a reused worker compiles the module once.
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
        self._rt = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_rt"] = None
        return state

    @staticmethod
    def _base(desc: str) -> str:
        return desc[len("option:"):] if desc.startswith("option:") else desc

    def _ensure(self):
        if self._rt is not None:
            return self._rt
        import hashlib

        try:
            import wasmtime
        except ImportError as exc:  # pragma: no cover - depends on executor env
            raise ImportError(
                "The 'wasmtime' package is required to run WASM UDFs on the "
                "Spark executors. Install it in the workers' Python env."
            ) from exc

        cache = sys.__dict__.setdefault("_spark_connect_wasm_udf_instances", {})
        key = hashlib.sha256(self.wasm).hexdigest()
        inst = cache.get(key)
        if inst is None:
            engine = wasmtime.Engine()
            store = wasmtime.Store(engine)
            module = wasmtime.Module(engine, self.wasm)
            exports = wasmtime.Instance(store, module, []).exports(store)
            inst = {
                "store": store,
                "exports": exports,
                "memory": exports["memory"],
                "scratch": exports["spark_udf_scratch"],
                "scratch_ptr": 0,
                "scratch_cap": 0,
            }
            cache[key] = inst
        entry = inst["exports"].get(self.entrypoint)
        if entry is None:
            raise ValueError(f"WASM module does not export '{self.entrypoint}'")
        self._rt = (inst, entry)
        return self._rt

    def __call__(self, *arrays: Any) -> Any:
        import ctypes

        import pyarrow as pa

        types = {
            "i32": pa.int32(),
            "i64": pa.int64(),
            "f32": pa.float32(),
            "f64": pa.float64(),
            "bool": pa.bool_(),
            "string": pa.string(),
            "binary": pa.binary(),
        }
        ret = self._base(self.ret_type)
        out_type = types[ret]
        if len(arrays) != len(self.arg_types):
            raise ValueError(
                f"WASM UDF '{self.entrypoint}' expected {len(self.arg_types)} "
                f"argument(s), got {len(arrays)}"
            )
        n = len(arrays[0]) if arrays else 0
        if n == 0:
            return pa.array([], type=out_type)

        # Per column: (validity, values, offsets) buffers, None where absent.
        cols = []
        for desc, a in zip(self.arg_types, arrays):
            if isinstance(a, pa.ChunkedArray):
                a = a.combine_chunks()
            if pa.types.is_large_string(a.type):
                a = a.cast(pa.string())
            elif pa.types.is_large_binary(a.type):
                a = a.cast(pa.binary())
            if a.offset:
                a = pa.concat_arrays([a])
            bufs = a.buffers()
            validity = bufs[0] if a.null_count else None
            if self._base(desc) in ("string", "binary"):
                cols.append((validity, bufs[2], bufs[1]))
            else:
                cols.append((validity, bufs[1], None))

        # Scratch layout: the descriptor, then each buffer at an 8-byte-aligned offset.
        off = (8 + 20 * len(cols) + 7) & ~7
        placed = []
        for col in cols:
            slots = []
            for buf in col:
                if buf is None or buf.size == 0:
                    slots.append((0, None))
                else:
                    slots.append((off, buf))
                    off = (off + buf.size + 7) & ~7
            placed.append(slots)

        inst, entry = self._ensure()
        store, memory = inst["store"], inst["memory"]
        if off > inst["scratch_cap"]:
            inst["scratch_ptr"] = inst["scratch"](store, off) & 0xFFFFFFFF
            inst["scratch_cap"] = off
        sp = inst["scratch_ptr"]
        base = ctypes.addressof(memory.data_ptr(store).contents)

        words = [n, len(cols)]
        for (vo, vb), (xo, xb), (oo, ob) in placed:
            words += [
                sp + vo if vb else 0,
                sp + xo if xb else 0,
                xb.size if xb else 0,
                sp + oo if ob else 0,
                ob.size if ob else 0,
            ]
            for o, b in ((vo, vb), (xo, xb), (oo, ob)):
                if b is not None:
                    ctypes.memmove(base + sp + o, b.address, b.size)
        desc_bytes = struct.pack("<%dI" % len(words), *words)
        ctypes.memmove(base + sp, desc_bytes, len(desc_bytes))

        out = entry(store, sp) & 0xFFFFFFFF
        base = ctypes.addressof(memory.data_ptr(store).contents)
        status, nulls, vp, vl, xp, xl, op, ol = struct.unpack(
            "<8I", ctypes.string_at(base + out, 32)
        )
        if status != 0:
            raise ValueError(ctypes.string_at(base + vp, vl).decode("utf-8", "replace"))

        def copy_out(ptr: int, size: int):
            buf = pa.allocate_buffer(size)
            if size:
                ctypes.memmove(buf.address, base + ptr, size)
            return buf

        validity = copy_out(vp, vl) if vp else None
        values = copy_out(xp, xl)
        if ret in ("string", "binary"):
            buffers = [validity, copy_out(op, ol), values]
        else:
            buffers = [validity, values]
        return pa.Array.from_buffers(out_type, n, buffers, null_count=nulls)


# --- packing: (WasmScalarUDF, return_type) -> cloudpickled command bytes -----


def build_command(spec: dict) -> bytes:
    # Prefer Spark's vendored cloudpickle (the exact one the reference client and
    # the worker ship); fall back to the standalone cloudpickle package.
    try:
        from pyspark import cloudpickle
    except ImportError:  # pragma: no cover - depends on client env
        import cloudpickle

    from pyspark.sql.types import _parse_datatype_json_value

    wasm = base64.b64decode(spec["wasm_b64"])
    if spec.get("batch_entrypoint"):
        runner = WasmArrowUDF(
            wasm,
            spec["batch_entrypoint"],
            spec["arg_types"],
            spec["ret_type"],
        )
    else:
        runner = WasmScalarUDF(
            wasm,
            spec["entrypoint"],
            spec["arg_types"],
            spec["ret_type"],
        )
    # `output_type` is an already-parsed JSON value (str or dict), parsed generically
    # (pure Python; no JVM). `runner`'s class lives in __main__, so cloudpickle embeds
    # it by value automatically -- executors need only `wasmtime`, not this packer.
    output_type = _parse_datatype_json_value(spec["output_type"])
    return cloudpickle.dumps((runner, output_type))


def main() -> None:
    spec = json.loads(sys.stdin.read())
    sys.stdout.buffer.write(build_command(spec))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
