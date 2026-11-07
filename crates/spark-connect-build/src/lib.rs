//! `build.rs` helper for Rust UDFs compiled to WebAssembly.
//!
//! Call [`embed_wasm_udf`] from a crate's `build.rs` to compile a Rust source
//! file containing `#[spark_wasm_udf]` functions to a `wasm32` module and expose
//! it to the crate as the `WASM_UDFS_MODULE` env var (for `include_bytes!`).
//!
//! # Requirements (build machine)
//!
//! * the `wasm32-unknown-unknown` target
//!   (`rustup target add wasm32-unknown-unknown`), and
//! * `apache-spark-connect-macros` listed as a **build-dependency** of the
//!   calling crate, so its proc-macro is compiled before `build.rs` runs and
//!   this helper can find its dylib.
//!
//! # Example `build.rs`
//!
//! ```ignore
//! spark_connect_build::embed_wasm_udf("src/main.rs");
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

/// Compile `src_file` to a `wasm32-unknown-unknown` cdylib and expose the
/// resulting module to the crate.
///
/// Emits `cargo:rustc-env=WASM_UDFS_MODULE=<path>` (so the crate can
/// `include_bytes!(env!("WASM_UDFS_MODULE"))`) and a `rerun-if-changed` for
/// `src_file`, and returns the module path. Panics with a descriptive message
/// on failure (the intended behavior for a build script).
///
/// The source is compiled as a separate crate: any host-only code (the client,
/// the macro-generated constructors) must be gated behind
/// `#[cfg(not(target_arch = "wasm32"))]` so the wasm build keeps only the
/// exported UDF functions.
pub fn embed_wasm_udf(src_file: impl AsRef<Path>) -> PathBuf {
    let src_file = src_file.as_ref();
    let out_dir = PathBuf::from(env("OUT_DIR"));
    let manifest_dir = PathBuf::from(env("CARGO_MANIFEST_DIR"));
    let src_abs = if src_file.is_absolute() {
        src_file.to_path_buf()
    } else {
        manifest_dir.join(src_file)
    };

    let stem = src_abs
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "wasm_udf".to_string());
    let wasm_out = out_dir.join(format!("{stem}.wasm"));

    // deps dir: OUT_DIR = target/<profile>/build/<crate>-<hash>/out
    let deps_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("unexpected OUT_DIR layout")
        .join("deps");
    let macro_dylib = find_proc_macro(&deps_dir, "spark_connect_macros").unwrap_or_else(|| {
        panic!(
            "could not find the spark_connect_macros proc-macro dylib in {}. \
             Add `apache-spark-connect-macros` to [build-dependencies].",
            deps_dir.display()
        )
    });

    // The macro's wasm export wrappers reference `crate::spark_wasm_rt` for the
    // binary-ABI codec + allocator. Prepend that runtime and compile the
    // combined source, so the module exports `spark_udf_alloc`/`_free` once.
    let user_src = std::fs::read_to_string(&src_abs)
        .unwrap_or_else(|e| panic!("reading {}: {e}", src_abs.display()));
    // User source first so its crate-level `//!` docs stay at the top; the
    // runtime is appended (item order does not affect name resolution).
    let wrapper = out_dir.join(format!("{stem}_wasm_udf_wrapper.rs"));
    std::fs::write(&wrapper, format!("{user_src}\n{WASM_RUNTIME}"))
        .unwrap_or_else(|e| panic!("writing {}: {e}", wrapper.display()));

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let status = Command::new(&rustc)
        .args(["--edition", "2021", "--crate-type", "cdylib"])
        .args(["--target", "wasm32-unknown-unknown"])
        // Keep the embedded module small and self-contained.
        .args([
            "-C",
            "opt-level=s",
            "-C",
            "panic=abort",
            "-C",
            "strip=symbols",
            // spark_udf.h hosts call the library's function pointers through the table.
            "-C",
            "link-arg=--export-table",
        ])
        .arg("-L")
        .arg(format!("dependency={}", deps_dir.display()))
        .arg("--extern")
        .arg(format!("spark_connect_macros={}", macro_dylib.display()))
        .arg(&wrapper)
        .arg("-o")
        .arg(&wasm_out)
        .status()
        .unwrap_or_else(|e| panic!("failed to invoke `{rustc}` for the wasm build: {e}"));

    if !status.success() {
        eprintln!();
        eprintln!("cargo:warning=wasm32 build of {} failed", src_abs.display());
        eprintln!("cargo:warning=");
        eprintln!("cargo:warning=The `wasm32-unknown-unknown` Rust target is required to build WASM UDF crates.");
        eprintln!("cargo:warning=Install it with: rustup target add wasm32-unknown-unknown");
        eprintln!();
        panic!("wasm32 build of {} failed", src_abs.display());
    }

    println!("cargo:rustc-env=WASM_UDFS_MODULE={}", wasm_out.display());
    println!("cargo:rerun-if-changed={}", src_abs.display());
    wasm_out
}

/// The runtime prepended to the wasm build: the exported allocator and the
/// length-prefixed binary codec (`Abi`) the macro's export wrappers call. Must
/// stay byte-compatible with `spark_connect::wasm_udf::AbiType` and the Python
/// runner's codec (`spark-connect/src/wasm_runner.py`).
const WASM_RUNTIME: &str = r#"
#[allow(dead_code)]
pub mod spark_wasm_rt {
    use std::alloc::{alloc as __alloc, dealloc as __dealloc, Layout};

    #[no_mangle]
    pub extern "C" fn spark_udf_alloc(size: usize, align: usize) -> *mut u8 {
        if size == 0 { return align.max(1) as *mut u8; }
        unsafe { __alloc(Layout::from_size_align(size, align.max(1)).unwrap()) }
    }
    #[no_mangle]
    pub extern "C" fn spark_udf_free(ptr: *mut u8, size: usize, align: usize) {
        if size == 0 || ptr.is_null() { return; }
        unsafe { __dealloc(ptr, Layout::from_size_align(size, align.max(1)).unwrap()) }
    }

    pub struct Reader<'a> { b: &'a [u8], off: usize }

    impl<'a> Reader<'a> {
        pub fn new(b: &'a [u8]) -> Self { Reader { b, off: 0 } }
        fn take(&mut self, n: usize) -> &'a [u8] {
            let s = &self.b[self.off..self.off + n];
            self.off += n;
            s
        }
        fn u32(&mut self) -> usize {
            u32::from_le_bytes(self.take(4).try_into().unwrap()) as usize
        }
    }

    pub trait Abi: Sized {
        fn decode(r: &mut Reader) -> Self;
        fn encode(&self, out: &mut Vec<u8>);
    }

    macro_rules! prim { ($t:ty, $n:expr) => {
        impl Abi for $t {
            fn decode(r: &mut Reader) -> Self { <$t>::from_le_bytes(r.take($n).try_into().unwrap()) }
            fn encode(&self, out: &mut Vec<u8>) { out.extend_from_slice(&self.to_le_bytes()); }
        }
    }}
    prim!(i32, 4); prim!(i64, 8); prim!(f32, 4); prim!(f64, 8);

    impl Abi for bool {
        fn decode(r: &mut Reader) -> Self { r.take(1)[0] != 0 }
        fn encode(&self, out: &mut Vec<u8>) { out.push(*self as u8); }
    }
    impl Abi for u8 {
        fn decode(r: &mut Reader) -> Self { r.take(1)[0] }
        fn encode(&self, out: &mut Vec<u8>) { out.push(*self); }
    }
    impl Abi for String {
        fn decode(r: &mut Reader) -> Self {
            let n = r.u32();
            String::from_utf8(r.take(n).to_vec()).unwrap()
        }
        fn encode(&self, out: &mut Vec<u8>) {
            out.extend_from_slice(&(self.len() as u32).to_le_bytes());
            out.extend_from_slice(self.as_bytes());
        }
    }
    impl<T: Abi> Abi for Vec<T> {
        fn decode(r: &mut Reader) -> Self {
            let n = r.u32();
            (0..n).map(|_| T::decode(r)).collect()
        }
        fn encode(&self, out: &mut Vec<u8>) {
            out.extend_from_slice(&(self.len() as u32).to_le_bytes());
            for x in self { x.encode(out); }
        }
    }
    impl<T: Abi> Abi for Option<T> {
        fn decode(r: &mut Reader) -> Self {
            if r.take(1)[0] == 0 { None } else { Some(T::decode(r)) }
        }
        fn encode(&self, out: &mut Vec<u8>) {
            match self { None => out.push(0), Some(v) => { out.push(1); v.encode(out); } }
        }
    }

    pub fn finish(out: Vec<u8>) -> u64 {
        let p = spark_udf_alloc(out.len(), 1);
        unsafe { std::slice::from_raw_parts_mut(p, out.len()) }.copy_from_slice(&out);
        ((p as u64) << 32) | (out.len() as u64)
    }

    // The structs of spark_udf.h version 1; Arrow arrays per the Arrow C data interface.
    pub mod abi {
        use std::cell::RefCell;

        #[repr(C)]
        pub struct ArrowArray {
            pub length: i64,
            pub null_count: i64,
            pub offset: i64,
            pub n_buffers: i64,
            pub n_children: i64,
            pub buffers: *mut *const u8,
            pub children: *mut *mut ArrowArray,
            pub dictionary: *mut ArrowArray,
            pub release: Option<unsafe extern "C" fn(*mut ArrowArray)>,
            pub private_data: *mut u8,
        }

        #[repr(C)]
        pub struct CreateInfo {
            pub size: i64,
            pub num_args: i64,
            pub arg_formats: *const *const u8,
            pub ret_format: *const u8,
        }

        pub type Create = unsafe extern "C" fn(*const CreateInfo, *mut *mut u8) -> i32;
        pub type Eval = unsafe extern "C" fn(*mut u8, *const ArrowArray, *mut ArrowArray) -> i32;
        pub type Close = unsafe extern "C" fn(*mut u8);

        #[repr(C)]
        pub struct Function {
            pub size: i64,
            pub name: *const u8,
            pub kind: i32,
            pub create: Create,
            pub eval: Eval,
            pub close: Close,
        }
        unsafe impl Sync for Function {}

        #[repr(C)]
        pub struct Library {
            pub size: i64,
            pub abi_version: i32,
            pub last_error: unsafe extern "C" fn() -> *const u8,
            pub num_functions: i64,
            pub functions: *const Function,
        }
        unsafe impl Sync for Library {}

        pub const SCALAR: i32 = 1;

        thread_local! {
            static LAST_ERROR: RefCell<Vec<u8>> = RefCell::new(Vec::new());
        }

        pub fn fail(msg: String) -> i32 {
            LAST_ERROR.with(|e| {
                let mut e = e.borrow_mut();
                e.clear();
                e.extend_from_slice(msg.as_bytes());
                e.push(0);
            });
            1
        }

        pub unsafe extern "C" fn last_error() -> *const u8 {
            LAST_ERROR.with(|e| {
                let e = e.borrow();
                if e.is_empty() { std::ptr::null() } else { e.as_ptr() }
            })
        }

        unsafe fn cstr<'a>(p: *const u8) -> &'a [u8] {
            let mut n = 0;
            while *p.add(n) != 0 { n += 1; }
            std::slice::from_raw_parts(p, n)
        }

        // `args` and `ret` are Arrow format strings.
        pub unsafe fn create(info: *const CreateInfo, state: *mut *mut u8, args: &[&str], ret: &str) -> i32 {
            *state = std::ptr::null_mut();
            let info = &*info;
            if info.num_args as usize != args.len() {
                return fail(format!("the function takes {} arguments, got {}", args.len(), info.num_args));
            }
            for (k, want) in args.iter().enumerate() {
                let got = cstr(*info.arg_formats.add(k));
                if got != want.as_bytes() {
                    return fail(format!(
                        "argument {k} has Arrow format '{}', the function takes '{want}'",
                        String::from_utf8_lossy(got)
                    ));
                }
            }
            if cstr(info.ret_format) != ret.as_bytes() {
                return fail(format!("the function returns Arrow format '{ret}'"));
            }
            0
        }

        pub unsafe extern "C" fn close(_state: *mut u8) {}
    }

    pub mod batch {
        use super::abi::{self, ArrowArray};

        pub struct Col {
            validity: *const u8,
            values: *const u8,
            offsets: *const i32,
            base: usize,
        }

        impl Col {
            unsafe fn new(a: &ArrowArray, parent_offset: i64) -> Self {
                let buf = |k: usize| *a.buffers.add(k);
                let var = a.n_buffers >= 3;
                Col {
                    validity: if a.null_count == 0 { std::ptr::null() } else { buf(0) },
                    values: if var { buf(2) } else { buf(1) },
                    offsets: if var { buf(1) as *const i32 } else { std::ptr::null() },
                    base: (a.offset + parent_offset) as usize,
                }
            }
            #[inline]
            fn bit(p: *const u8, j: usize) -> bool {
                unsafe { (*p.add(j >> 3) >> (j & 7)) & 1 == 1 }
            }
            #[inline]
            pub fn is_valid(&self, i: usize) -> bool {
                self.validity.is_null() || Self::bit(self.validity, self.base + i)
            }
            #[inline]
            fn bytes(&self, i: usize) -> &[u8] {
                unsafe {
                    let o = self.offsets.add(self.base + i);
                    let (s, e) = (*o as usize, *o.add(1) as usize);
                    std::slice::from_raw_parts(self.values.add(s), e - s)
                }
            }
        }

        pub trait FromCol: Sized {
            fn get_valid(col: &Col, i: usize) -> Self;
            // None: a null in an argument that isn't an Option.
            #[inline]
            fn get(col: &Col, i: usize) -> Option<Self> {
                if col.is_valid(i) { Some(Self::get_valid(col, i)) } else { None }
            }
        }

        macro_rules! prim_in { ($t:ty) => {
            impl FromCol for $t {
                #[inline]
                fn get_valid(col: &Col, i: usize) -> Self {
                    unsafe { (col.values as *const $t).add(col.base + i).read_unaligned() }
                }
            }
        }}
        prim_in!(i32); prim_in!(i64); prim_in!(f32); prim_in!(f64);

        impl FromCol for bool {
            #[inline]
            fn get_valid(col: &Col, i: usize) -> Self { Col::bit(col.values, col.base + i) }
        }
        impl FromCol for String {
            #[inline]
            fn get_valid(col: &Col, i: usize) -> Self {
                String::from_utf8(col.bytes(i).to_vec()).unwrap()
            }
        }
        impl FromCol for Vec<u8> {
            #[inline]
            fn get_valid(col: &Col, i: usize) -> Self { col.bytes(i).to_vec() }
        }
        impl<T: FromCol> FromCol for Option<T> {
            #[inline]
            fn get_valid(col: &Col, i: usize) -> Self { Some(T::get_valid(col, i)) }
            #[inline]
            fn get(col: &Col, i: usize) -> Option<Self> {
                Some(if col.is_valid(i) { Some(T::get_valid(col, i)) } else { None })
            }
        }

        pub struct Out {
            n: usize,
            row: usize,
            validity: Vec<u8>,
            null_count: i64,
            values: Vec<u8>,
            offsets: Vec<u8>,
            var: bool,
        }

        impl Out {
            fn new(n: usize, var: bool) -> Self {
                let mut offsets = Vec::new();
                if var {
                    offsets.reserve(4 * (n + 1));
                    offsets.extend_from_slice(&0i32.to_le_bytes());
                }
                Out {
                    n,
                    row: 0,
                    validity: vec![0xFF; (n + 7) / 8],
                    null_count: 0,
                    values: Vec::new(),
                    offsets,
                    var,
                }
            }
            #[inline]
            fn set_null(&mut self) {
                let r = self.row;
                self.validity[r >> 3] &= !(1u8 << (r & 7));
                self.null_count += 1;
            }
            #[inline]
            fn end_var(&mut self) {
                let end = self.values.len() as i32;
                self.offsets.extend_from_slice(&end.to_le_bytes());
            }
            #[inline]
            fn bits(&mut self) {
                if self.values.is_empty() {
                    self.values = vec![0u8; (self.n + 7) / 8];
                }
            }
            fn finish(self, out: *mut ArrowArray) {
                let validity = if self.null_count == 0 { Vec::new() } else { self.validity };
                let bufs = if self.var {
                    vec![validity, self.offsets, self.values]
                } else {
                    vec![validity, self.values]
                };
                let mut owned = Box::new(Owned { ptrs: Vec::new(), bufs });
                owned.ptrs = owned
                    .bufs
                    .iter()
                    .map(|b| if b.is_empty() { std::ptr::null() } else { b.as_ptr() })
                    .collect();
                let owned = Box::into_raw(owned);
                unsafe {
                    *out = ArrowArray {
                        length: self.row as i64,
                        null_count: self.null_count,
                        offset: 0,
                        n_buffers: (*owned).ptrs.len() as i64,
                        n_children: 0,
                        buffers: (*owned).ptrs.as_mut_ptr(),
                        children: std::ptr::null_mut(),
                        dictionary: std::ptr::null_mut(),
                        release: Some(release),
                        private_data: owned as *mut u8,
                    };
                }
            }
        }

        struct Owned {
            ptrs: Vec<*const u8>,
            bufs: Vec<Vec<u8>>,
        }

        unsafe extern "C" fn release(a: *mut ArrowArray) {
            drop(Box::from_raw((*a).private_data as *mut Owned));
            (*a).release = None;
        }

        pub trait ToCol {
            const VAR: bool = false;
            fn push(&self, out: &mut Out);
            fn push_null(out: &mut Out);
        }

        macro_rules! prim_out { ($t:ty) => {
            impl ToCol for $t {
                #[inline]
                fn push(&self, out: &mut Out) {
                    out.values.extend_from_slice(&self.to_le_bytes());
                    out.row += 1;
                }
                #[inline]
                fn push_null(out: &mut Out) {
                    out.set_null();
                    out.values.extend_from_slice(&<$t>::default().to_le_bytes());
                    out.row += 1;
                }
            }
        }}
        prim_out!(i32); prim_out!(i64); prim_out!(f32); prim_out!(f64);

        impl ToCol for bool {
            #[inline]
            fn push(&self, out: &mut Out) {
                out.bits();
                if *self {
                    let r = out.row;
                    out.values[r >> 3] |= 1u8 << (r & 7);
                }
                out.row += 1;
            }
            #[inline]
            fn push_null(out: &mut Out) {
                out.bits();
                out.set_null();
                out.row += 1;
            }
        }
        impl ToCol for String {
            const VAR: bool = true;
            #[inline]
            fn push(&self, out: &mut Out) {
                out.values.extend_from_slice(self.as_bytes());
                out.end_var();
                out.row += 1;
            }
            #[inline]
            fn push_null(out: &mut Out) {
                out.set_null();
                out.end_var();
                out.row += 1;
            }
        }
        impl ToCol for Vec<u8> {
            const VAR: bool = true;
            #[inline]
            fn push(&self, out: &mut Out) {
                out.values.extend_from_slice(self);
                out.end_var();
                out.row += 1;
            }
            #[inline]
            fn push_null(out: &mut Out) {
                out.set_null();
                out.end_var();
                out.row += 1;
            }
        }
        impl<T: ToCol> ToCol for Option<T> {
            const VAR: bool = T::VAR;
            #[inline]
            fn push(&self, out: &mut Out) {
                match self {
                    Some(v) => v.push(out),
                    None => T::push_null(out),
                }
            }
            #[inline]
            fn push_null(out: &mut Out) { T::push_null(out) }
        }

        // `f` evaluates row `i`; Err(arg) means a null in argument `arg`, which isn't an Option.
        pub unsafe fn run<R: ToCol, F>(
            input: *const ArrowArray,
            out: *mut ArrowArray,
            nargs: usize,
            mut f: F,
        ) -> i32
        where
            F: FnMut(&[Col], usize, &mut Out) -> Result<(), usize>,
        {
            let input = &*input;
            if input.n_children as usize != nargs {
                return abi::fail(format!(
                    "the batch has {} columns, the function takes {}",
                    input.n_children, nargs
                ));
            }
            let n = input.length as usize;
            let cols: Vec<Col> = (0..nargs)
                .map(|k| Col::new(&**input.children.add(k), input.offset))
                .collect();
            let mut o = Out::new(n, R::VAR);
            for i in 0..n {
                if let Err(arg) = f(&cols, i, &mut o) {
                    return abi::fail(format!(
                        "null value for non-nullable WASM UDF argument {arg} (row {i}); \
                         use Option<T> to accept nulls"
                    ));
                }
            }
            o.finish(out);
            0
        }
    }
}
"#;

fn env(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("`{key}` is not set (call from a build script)"))
}

/// Find `lib<name>-<hash>.{dylib,so,dll}` in `deps_dir`, newest first.
fn find_proc_macro(deps_dir: &Path, name: &str) -> Option<PathBuf> {
    let prefix = format!("lib{name}-");
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(deps_dir).ok()?.flatten() {
        let path = entry.path();
        let fname = match path.file_name() {
            Some(f) => f.to_string_lossy().into_owned(),
            None => continue,
        };
        let is_dylib =
            fname.ends_with(".dylib") || fname.ends_with(".so") || fname.ends_with(".dll");
        if fname.starts_with(&prefix) && is_dylib {
            if let Ok(mtime) = entry.metadata().and_then(|m| m.modified()) {
                if best.as_ref().map(|(t, _)| mtime > *t).unwrap_or(true) {
                    best = Some((mtime, path));
                }
            }
        }
    }
    best.map(|(_, p)| p)
}
