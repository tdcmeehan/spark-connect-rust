//! Procedural macros for the Rust Spark Connect client.
//!
//! Provides [`macro@spark_udf`], which turns a module of plain Rust
//! functions into Spark UDFs that run on the executors via WebAssembly.
//! [`macro@spark_wasm_udf`] is its deprecated old name.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{FnArg, GenericArgument, Item, ItemFn, ItemMod, Pat, PathArguments, ReturnType, Type};

/// Turn a module of plain Rust functions into Spark UDFs backed by WebAssembly.
///
/// Applied to a `mod`, for each free function inside it the macro:
///
/// * keeps the function callable natively,
/// * emits (when compiled for `wasm32`) a `#[no_mangle] extern "C"` export that
///   decodes the arguments and encodes the result with the length-prefixed
///   binary ABI (`spark_connect::rust_udf::AbiType`), and
/// * generates, on non-`wasm32` targets, a direct-call constructor
///   `udf::<name>(col0, col1, ...) -> Result<Column>` (one column per argument,
///   arity checked at compile time) plus a builder `udf::<name>_udf() ->
///   UserDefinedFunction` for advanced configuration — both with the Spark
///   signature **inferred** from the Rust signature and the crate's own compiled
///   wasm module embedded.
///
/// A function whose arguments and result are all `i32`, `i64`, `f32`, `f64`,
/// `bool`, `String`, `Vec<u8>`, or one `Option` of those also gets a batch
/// export, and its builder sends it as a scalar Arrow UDF that Spark calls once
/// per Arrow batch. `.with_eval_type(spark_connect::udf::eval_type::SQL_ARROW_BATCHED_UDF)`
/// on the builder calls it once per row instead, like functions with other types.
///
/// ```ignore
/// use spark_connect_macros::spark_udf;
///
/// #[spark_udf]
/// mod udfs {
///     pub fn add_one(x: i64) -> i64 { x + 1 }
///     pub fn shout(s: String) -> String { format!("{}!", s.to_uppercase()) }
/// }
///
/// // callers (build.rs embeds the module):
/// df.select(vec![udf::add_one(col("id"))?]);              // direct call
/// df.select(vec![udf::add_one_udf().as_nondeterministic().call(vec![col("id")])?]);
/// ```
///
/// Supported types: `i32`, `i64`, `f32`, `f64`, `bool`, `String`, `Vec<u8>`
/// (binary), `Vec<T>` (array), and `Option<T>` (nullable), nested arbitrarily.
///
/// # The async client
///
/// `#[spark_udf(crate = spark_connect_async)]` generates the host code for the async
/// client, `apache-spark-connect-async`, with its `rust-udf` feature: `udf::<name>_udf()`
/// returns its `rust_udf::UserDefinedFunction`, and `udf::<name>(...)` takes, for each
/// argument, anything that converts into its `Column`, such as a column name, and keeps one
/// `UserDefinedFunction` for all its calls, so it builds the payload once per process. The
/// value is a path to that crate as the generated `udf` module sees it, such as the name of the
/// dependency, renamed or not, or a path to a re-export of it; it may also be written as a
/// string, `crate = "spark_connect_async"`. Without the option the generated code uses
/// `spark_connect`, the crate of `apache-spark-connect`. Any other argument is an error.
///
/// ```ignore
/// use spark_connect_macros::spark_udf;
///
/// #[spark_udf(crate = spark_connect_async)]
/// mod udfs {
///     pub fn add_one(x: i64) -> i64 { x + 1 }
/// }
///
/// let df = session.range(3).select([udf::add_one("id")?]);
/// session.udf().register("add_one", &udf::add_one_udf()).await?;
/// ```
#[proc_macro_attribute]
pub fn spark_udf(attr: TokenStream, item: TokenStream) -> TokenStream {
    spark_udf_expansion(attr.into(), item.into()).into()
}

/// What `#[spark_udf(attr)]` makes of `item`: an argument other than `crate` is an error.
fn spark_udf_expansion(
    attr: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    expansion(Options::parse(attr), item)
}

/// What `#[spark_wasm_udf(attr)]` makes of `item`: the expansion of `#[spark_udf(attr)]` if
/// `attr` is a `crate` option that it accepts, else that of `#[spark_udf]`.
fn spark_wasm_udf_expansion(
    attr: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    // Its arguments were ignored before the `crate` option existed, so every attribute that
    // compiled then still does, and expands as it did.
    expansion(Ok(Options::parse(attr).unwrap_or_default()), item)
}

fn expansion(
    options: syn::Result<Options>,
    item: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let options = match options {
        Ok(options) => options,
        Err(e) => return e.to_compile_error(),
    };
    let module = match syn::parse2::<ItemMod>(item) {
        Ok(module) => module,
        Err(e) => return e.to_compile_error(),
    };
    expand(module, &options).unwrap_or_else(|e| e.to_compile_error())
}

/// The arguments of `#[spark_udf(...)]`.
#[derive(Default)]
struct Options {
    /// The path of the client crate that the host code names, from `crate = <path>`.
    client: Option<syn::Path>,
}

impl Options {
    fn parse(attr: proc_macro2::TokenStream) -> syn::Result<Options> {
        let mut options = Options::default();
        let parser = syn::meta::parser(|meta| {
            if !meta.path.is_ident("crate") {
                return Err(meta.error("unsupported #[spark_udf] option; the only one is `crate`"));
            }
            if options.client.is_some() {
                return Err(meta.error("duplicate `crate` option"));
            }
            let value = meta.value()?;
            options.client = Some(if value.peek(syn::LitStr) {
                value
                    .parse::<syn::LitStr>()?
                    .parse_with(syn::Path::parse_mod_style)?
            } else {
                syn::Path::parse_mod_style(value)?
            });
            Ok(())
        });
        syn::parse::Parser::parse2(parser, attr)?;
        Ok(options)
    }
}

/// The old name of [`macro@spark_udf`], kept as a deprecated alias.
///
/// It takes the `crate` option of [`macro@spark_udf`], and ignores any other arguments, as it
/// always has: with them, or with a `crate` option that [`macro@spark_udf`] rejects, it
/// expands as without arguments.
#[deprecated(
    note = "renamed `spark_udf`; the `wasm-udf` feature of apache-spark-connect is now `rust-udf`"
)]
#[proc_macro_attribute]
pub fn spark_wasm_udf(attr: TokenStream, item: TokenStream) -> TokenStream {
    spark_wasm_udf_expansion(attr.into(), item.into()).into()
}

/// Map a Rust type to a `<client>::rust_udf::AbiType` constructor expr.
fn abi_type(
    client: &proc_macro2::TokenStream,
    ty: &Type,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let root = quote!(#client::rust_udf::AbiType);
    if let Some(ident) = path_ident(ty) {
        match ident.as_str() {
            "i32" => return Ok(quote!(#root::I32)),
            "i64" => return Ok(quote!(#root::I64)),
            "f32" => return Ok(quote!(#root::F32)),
            "f64" => return Ok(quote!(#root::F64)),
            "bool" => return Ok(quote!(#root::Bool)),
            "String" => return Ok(quote!(#root::Str)),
            _ => {}
        }
    }
    if let Some(inner) = generic_inner(ty, "Vec") {
        if path_ident(inner).as_deref() == Some("u8") {
            return Ok(quote!(#root::Binary));
        }
        let inner = abi_type(client, inner)?;
        return Ok(quote!(#root::Array(::std::boxed::Box::new(#inner))));
    }
    if let Some(inner) = generic_inner(ty, "Option") {
        let inner = abi_type(client, inner)?;
        return Ok(quote!(#root::Nullable(::std::boxed::Box::new(#inner))));
    }
    Err(syn::Error::new_spanned(
        ty,
        "#[spark_udf] supports i32/i64/f32/f64/bool/String/Vec<u8>/Vec<T>/Option<T> \
         (use owned types, e.g. `String` not `&str`)",
    ))
}

fn batch_supported(ty: &Type) -> bool {
    if let Some(inner) = generic_inner(ty, "Option") {
        return generic_inner(inner, "Option").is_none() && batch_supported(inner);
    }
    if let Some(inner) = generic_inner(ty, "Vec") {
        return path_ident(inner).as_deref() == Some("u8");
    }
    matches!(
        path_ident(ty).as_deref(),
        Some("i32" | "i64" | "f32" | "f64" | "bool" | "String")
    )
}

/// The last path segment ident of a plain type path (`i64`, `String`, ...).
fn path_ident(ty: &Type) -> Option<String> {
    match ty {
        Type::Path(p) if p.qself.is_none() => p.path.segments.last().map(|s| s.ident.to_string()),
        _ => None,
    }
}

/// If `ty` is `Wrapper<Inner>` (e.g. `Vec<T>`, `Option<T>`), return `Inner`.
fn generic_inner<'a>(ty: &'a Type, wrapper: &str) -> Option<&'a Type> {
    let Type::Path(p) = ty else { return None };
    let seg = p.path.segments.last()?;
    if seg.ident != wrapper {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    args.args.iter().find_map(|a| match a {
        GenericArgument::Type(t) => Some(t),
        _ => None,
    })
}

struct Udf {
    name: syn::Ident,
    export: syn::Ident,
    batch_export: Option<syn::Ident>,
    arg_names: Vec<String>,
    arg_tys: Vec<Type>,
    ret_ty: Type,
}

fn collect_udf(func: &ItemFn) -> Result<Udf, syn::Error> {
    if func.sig.asyncness.is_some() {
        return Err(syn::Error::new_spanned(
            &func.sig,
            "#[spark_udf] cannot be applied to async functions",
        ));
    }
    if !func.sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &func.sig.generics,
            "#[spark_udf] cannot be applied to generic functions",
        ));
    }
    let name = func.sig.ident.clone();
    let mut arg_names = Vec::new();
    let mut arg_tys = Vec::new();
    for (i, input) in func.sig.inputs.iter().enumerate() {
        match input {
            FnArg::Typed(pt) => {
                arg_names.push(match &*pt.pat {
                    Pat::Ident(p) => p.ident.to_string(),
                    _ => format!("#{i}"),
                });
                arg_tys.push((*pt.ty).clone());
            }
            FnArg::Receiver(r) => {
                return Err(syn::Error::new_spanned(
                    r,
                    "#[spark_udf] cannot be applied to methods (`self`)",
                ))
            }
        }
    }
    if arg_tys.is_empty() {
        return Err(syn::Error::new_spanned(
            &func.sig,
            "#[spark_udf] requires at least one argument",
        ));
    }
    let ret_ty = match &func.sig.output {
        ReturnType::Type(_, ty) => (**ty).clone(),
        ReturnType::Default => {
            return Err(syn::Error::new_spanned(
                &func.sig,
                "#[spark_udf] requires a return type",
            ))
        }
    };
    let export = format_ident!("spark_udf_{}", name);
    let batch_export = arg_tys
        .iter()
        .chain([&ret_ty])
        .all(batch_supported)
        .then(|| format_ident!("spark_batch_udf_{}", name));
    Ok(Udf {
        name,
        export,
        batch_export,
        arg_names,
        arg_tys,
        ret_ty,
    })
}

fn expand(mut module: ItemMod, options: &Options) -> Result<proc_macro2::TokenStream, syn::Error> {
    let mod_ident = module.ident.clone();
    let client = match &options.client {
        Some(path) => quote!(#path),
        None => quote!(::spark_connect),
    };
    let Some((_, items)) = &module.content else {
        return Err(syn::Error::new_spanned(
            &module,
            "#[spark_udf] must be applied to an inline module with a body",
        ));
    };

    let udfs: Vec<Udf> = items
        .iter()
        .filter_map(|it| match it {
            Item::Fn(f) => Some(collect_udf(f)),
            _ => None,
        })
        .collect::<Result<_, _>>()?;
    if udfs.is_empty() {
        return Err(syn::Error::new_spanned(
            &module,
            "#[spark_udf] module contains no functions",
        ));
    }

    // WASM export wrappers, injected into the module (compiled only for wasm32).
    let mut wrappers = Vec::new();
    for u in &udfs {
        let Udf {
            name,
            export,
            batch_export,
            arg_names,
            arg_tys,
            ret_ty,
        } = u;
        let params: Vec<_> = arg_tys
            .iter()
            .map(|ty| quote!(<#ty as crate::spark_wasm_rt::Abi>::decode(&mut __r)))
            .collect();
        wrappers.push(quote! {
            #[cfg(target_arch = "wasm32")]
            #[no_mangle]
            pub extern "C" fn #export(__ptr: *const u8, __len: u32) -> u64 {
                let __b = unsafe { ::core::slice::from_raw_parts(__ptr, __len as usize) };
                let mut __r = crate::spark_wasm_rt::Reader::new(__b);
                let __out: #ret_ty = #name(#(#params),*);
                let mut __o = ::std::vec::Vec::new();
                crate::spark_wasm_rt::Abi::encode(&__out, &mut __o);
                crate::spark_wasm_rt::finish(__o)
            }
        });
        if let Some(batch_export) = batch_export {
            let vars: Vec<syn::Ident> = (0..arg_tys.len())
                .map(|i| format_ident!("__a{i}"))
                .collect();
            let gets = arg_tys.iter().zip(&vars).enumerate().map(|(i, (ty, v))| {
                quote! {
                    let #v: #ty =
                        <#ty as crate::spark_wasm_rt::batch::FromCol>::get(&__cols[#i], __i)
                            .ok_or(#i)?;
                }
            });
            wrappers.push(quote! {
                #[cfg(target_arch = "wasm32")]
                #[no_mangle]
                pub extern "C" fn #batch_export(__desc: u32) -> u32 {
                    crate::spark_wasm_rt::batch::run(__desc, &[#(#arg_names),*], |__cols, __i, __out| {
                        #(#gets)*
                        let __r: #ret_ty = #name(#(#vars),*);
                        crate::spark_wasm_rt::batch::ToCol::push(&__r, __out);
                        ::core::result::Result::Ok(())
                    })
                }
            });
        }
    }
    if let Some((_, content)) = &mut module.content {
        for w in wrappers {
            content.push(syn::parse2(w)?);
        }
    }

    // Host-side constructors, grouped under `udf::`. For each UDF the macro
    // emits two functions:
    //   * `udf::<name>(col0, col1, ...) -> Result<Column>` — the direct call,
    //     one column argument per Rust parameter (arity checked at compile
    //     time), mirroring calling a PySpark UDF on columns.
    //   * `udf::<name>_udf() -> UserDefinedFunction` — the builder, for advanced
    //     configuration (`.as_nondeterministic()`, `.with_packer()`, ...) before
    //     `.call(cols)`.
    let mut ctors = Vec::new();
    for u in &udfs {
        let Udf {
            name,
            export,
            batch_export,
            arg_tys,
            ret_ty,
            ..
        } = u;
        let name_str = name.to_string();
        let export_str = export.to_string();
        let builder = format_ident!("{}_udf", name);
        let arg_abis: Vec<_> = arg_tys
            .iter()
            .map(|ty| abi_type(&client, ty))
            .collect::<Result<_, _>>()?;
        let ret_abi = abi_type(&client, ret_ty)?;
        let with_batch = batch_export.as_ref().map(|b| {
            let b = b.to_string();
            quote!(.with_batch_entrypoint(#b))
        });
        let params: Vec<syn::Ident> = (0..arg_tys.len())
            .map(|i| format_ident!("__a{i}"))
            .collect();
        let builder_doc = match options.client {
            None => format!(
                "Builder for the `{name_str}` UDF, generated by `#[spark_udf]`. \
                 For advanced configuration (`.as_nondeterministic()`, `.with_packer()`) \
                 before `.call(cols)`; the common case is `{name_str}(cols...)`."
            ),
            Some(_) => format!(
                "Builder for the `{name_str}` UDF, generated by `#[spark_udf]`, to configure \
                 it (`.as_nondeterministic()`) before `.call(cols)` or to register it; the \
                 common case is `{name_str}(cols...)`."
            ),
        };
        let call_doc = match options.client {
            None => format!(
                "Apply the `{name_str}` UDF to columns, generated by `#[spark_udf]`. \
                 Takes one column per argument and returns the result column."
            ),
            Some(_) => format!(
                "Apply the `{name_str}` UDF to columns, generated by `#[spark_udf]`. \
                 Takes one column per argument, or anything that converts into one, such as \
                 a column name, and returns the result column."
            ),
        };
        let call = match options.client {
            None => quote! {
                #[doc = #call_doc]
                pub fn #name(
                    #(#params: ::spark_connect::Column),*
                ) -> ::spark_connect::Result<::spark_connect::Column> {
                    #builder().call(::std::vec![#(#params),*])
                }
            },
            // The client's `Column` may be its own type, so each argument converts into it. Its
            // UDF value keeps the payload it builds, so one value serves every call.
            Some(_) => quote! {
                #[doc = #call_doc]
                pub fn #name(
                    #(#params: impl ::core::convert::Into<#client::Column>),*
                ) -> #client::Result<#client::Column> {
                    static __UDF: ::std::sync::OnceLock<#client::rust_udf::UserDefinedFunction> =
                        ::std::sync::OnceLock::new();
                    __UDF.get_or_init(#builder).call(::std::vec![
                        #(::core::convert::Into::<#client::Column>::into(#params)),*
                    ])
                }
            },
        };
        ctors.push(quote! {
            #[doc = #builder_doc]
            pub fn #builder() -> #client::rust_udf::UserDefinedFunction {
                const __MODULE: &[u8] = include_bytes!(env!("WASM_UDFS_MODULE"));
                #client::rust_udf::udf(
                    #name_str,
                    __MODULE.to_vec(),
                    #export_str,
                    ::std::vec![#(#arg_abis),*],
                    #ret_abi,
                )
                #with_batch
            }

            #call
        });
    }

    let doc = format!("UDF constructors generated by `#[spark_udf]` from module `{mod_ident}`.");
    Ok(quote! {
        // Off wasm32 the exported UDF functions are called only from the
        // `#[cfg(target_arch = "wasm32")]` wrappers injected above, so on the host
        // build they legitimately look unused - they exist to be compiled to wasm.
        // Allow dead_code there (not on wasm32, where the wrappers do use them) so the
        // macro does not force `-D warnings` failures on every generated UDF module.
        #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
        #module

        #[cfg(not(target_arch = "wasm32"))]
        #[doc = #doc]
        pub mod udf {
            #(#ctors)*
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{batch_supported, expand, spark_udf_expansion, spark_wasm_udf_expansion, Options};
    use proc_macro2::TokenStream;
    use quote::quote;

    fn options(attr: &str) -> syn::Result<Options> {
        Options::parse(attr.parse().unwrap())
    }

    #[test]
    fn the_crate_option_is_a_path_or_a_string_of_one_and_nothing_else_is_accepted() {
        for (attr, expected) in [
            ("", Ok(None)),
            (
                "crate = spark_connect_async",
                Ok(Some("spark_connect_async")),
            ),
            (
                "crate = \"spark_connect_async\"",
                Ok(Some("spark_connect_async")),
            ),
            (
                "crate = ::spark_connect_async",
                Ok(Some(":: spark_connect_async")),
            ),
            ("crate = crate::client,", Ok(Some("crate :: client"))),
            ("crate = a, crate = b", Err("duplicate `crate` option")),
            (
                "client = a",
                Err("unsupported #[spark_udf] option; the only one is `crate`"),
            ),
            ("crate", Err("expected `=`")),
            ("crate = a<T>", Err("expected `,`")),
            ("crate = \"a b\"", Err("unexpected token")),
        ] {
            let parsed = options(attr).map(|o| o.client.map(|p| quote!(#p).to_string()));
            let parsed = parsed
                .as_ref()
                .map(Option::as_deref)
                .map_err(|e| e.to_string());
            assert_eq!(parsed, expected.map_err(str::to_string), "{attr}");
        }
    }

    #[test]
    fn the_old_name_ignores_the_arguments_that_the_new_name_rejects() {
        let module = quote!(
            mod m {
                pub fn plus(x: i64) -> i64 {
                    x
                }
            }
        );
        let expand = |expansion: fn(TokenStream, TokenStream) -> TokenStream, attr: &str| {
            expansion(attr.parse().unwrap(), module.clone()).to_string()
        };
        let default = expand(spark_udf_expansion, "");
        let for_the_async_client = expand(spark_udf_expansion, "crate = spark_connect_async");
        assert_ne!(default, for_the_async_client);
        assert_eq!(expand(spark_wasm_udf_expansion, ""), default);
        assert_eq!(
            expand(spark_wasm_udf_expansion, "crate = spark_connect_async"),
            for_the_async_client
        );
        for attr in [
            "anything",
            "x y z",
            "crate",
            "crate = 1",
            "crate = a<T>",
            "crate = spark_connect_async, anything",
            "anything = 1, crate = spark_connect_async",
        ] {
            let rejected = expand(spark_udf_expansion, attr);
            assert!(
                rejected.starts_with(":: core :: compile_error !"),
                "{attr}: {rejected}"
            );
            assert_eq!(expand(spark_wasm_udf_expansion, attr), default, "{attr}");
        }
    }

    #[test]
    fn the_host_code_names_the_crate_of_the_option_and_converts_the_arguments_into_its_columns() {
        let module: syn::ItemMod = syn::parse_quote! {
            mod m {
                pub fn plus(x: i64, y: Option<i64>) -> Option<i64> {
                    y.map(|y| x + y)
                }
            }
        };
        let builder = |client: proc_macro2::TokenStream, doc: &str| {
            quote! {
                #[doc = #doc]
                pub fn plus_udf() -> #client::rust_udf::UserDefinedFunction {
                    const __MODULE: &[u8] = include_bytes!(env!("WASM_UDFS_MODULE"));
                    #client::rust_udf::udf(
                        "plus",
                        __MODULE.to_vec(),
                        "spark_udf_plus",
                        ::std::vec![
                            #client::rust_udf::AbiType::I64,
                            #client::rust_udf::AbiType::Nullable(::std::boxed::Box::new(
                                #client::rust_udf::AbiType::I64
                            ))
                        ],
                        #client::rust_udf::AbiType::Nullable(::std::boxed::Box::new(
                            #client::rust_udf::AbiType::I64
                        )),
                    )
                    .with_batch_entrypoint("spark_batch_udf_plus")
                }
            }
        };
        let old_call_doc = "Apply the `plus` UDF to columns, generated by `#[spark_udf]`. Takes \
                            one column per argument and returns the result column.";
        let call_doc = "Apply the `plus` UDF to columns, generated by `#[spark_udf]`. Takes one \
                        column per argument, or anything that converts into one, such as a column \
                        name, and returns the result column.";
        let old_builder = builder(
            quote!(::spark_connect),
            "Builder for the `plus` UDF, generated by `#[spark_udf]`. For advanced configuration \
             (`.as_nondeterministic()`, `.with_packer()`) before `.call(cols)`; the common case \
             is `plus(cols...)`.",
        );
        let old = quote! {
            #old_builder
            #[doc = #old_call_doc]
            pub fn plus(
                __a0: ::spark_connect::Column, __a1: ::spark_connect::Column
            ) -> ::spark_connect::Result<::spark_connect::Column> {
                plus_udf().call(::std::vec![__a0, __a1])
            }
        };
        let converted = |client: proc_macro2::TokenStream| {
            let builder = builder(
                client.clone(),
                "Builder for the `plus` UDF, generated by `#[spark_udf]`, to configure it \
                 (`.as_nondeterministic()`) before `.call(cols)` or to register it; the common \
                 case is `plus(cols...)`.",
            );
            quote! {
                #builder
                #[doc = #call_doc]
                pub fn plus(
                    __a0: impl ::core::convert::Into<#client::Column>,
                    __a1: impl ::core::convert::Into<#client::Column>
                ) -> #client::Result<#client::Column> {
                    static __UDF: ::std::sync::OnceLock<#client::rust_udf::UserDefinedFunction> =
                        ::std::sync::OnceLock::new();
                    __UDF.get_or_init(plus_udf).call(::std::vec![
                        ::core::convert::Into::<#client::Column>::into(__a0),
                        ::core::convert::Into::<#client::Column>::into(__a1)
                    ])
                }
            }
        };
        for (attr, ctors) in [
            ("", old),
            (
                "crate = spark_connect_async",
                converted(quote!(spark_connect_async)),
            ),
            ("crate = \"::a::b\"", converted(quote!(::a::b))),
        ] {
            let expanded = expand(module.clone(), &options(attr).unwrap()).unwrap();
            let file: syn::File = syn::parse2(expanded).unwrap();
            let [syn::Item::Mod(_), syn::Item::Mod(udf)] = &file.items[..] else {
                panic!("{attr}: not a module and its `udf` module");
            };
            let doc = "UDF constructors generated by `#[spark_udf]` from module `m`.";
            let expected = quote! {
                #[cfg(not(target_arch = "wasm32"))]
                #[doc = #doc]
                pub mod udf {
                    #ctors
                }
            };
            assert_eq!(quote!(#udf).to_string(), expected.to_string(), "{attr}");
        }
    }

    #[test]
    fn batch_supported_accepts_scalars_and_one_option_level_only() {
        for (ty, expected) in [
            ("i32", true),
            ("i64", true),
            ("f32", true),
            ("f64", true),
            ("bool", true),
            ("String", true),
            ("std::string::String", true),
            ("Vec<u8>", true),
            ("Option<i64>", true),
            ("Option<String>", true),
            ("Option<Vec<u8>>", true),
            ("Vec<i64>", false),
            ("Vec<String>", false),
            ("Vec<Option<u8>>", false),
            ("Option<Vec<i64>>", false),
            ("Option<Option<i64>>", false),
        ] {
            let parsed: syn::Type = syn::parse_str(ty).unwrap();
            assert_eq!(batch_supported(&parsed), expected, "{ty}");
        }
    }
}
