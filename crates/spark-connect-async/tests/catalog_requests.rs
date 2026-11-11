// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

//! The requests of the catalog of the async API, recorded by the old API's recording server and
//! compared with the old API's.

#[path = "../../spark-connect/tests/request_snapshots/server.rs"]
#[allow(dead_code)]
mod server;
mod support;

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use futures::future::BoxFuture;
use prost::Message;
use spark_connect::SparkSession;
use spark_connect_async::actions::StorageLevel;
use spark_connect_async::types::{DataType, StructField};
use spark_connect_async::Session;
use spark_connect_proto as proto;

const SESSION_ID: &str = "00000000-0000-4000-8000-0000000000b9";

struct Case {
    name: &'static str,
    old: fn(&SparkSession),
    new: fn(Session) -> BoxFuture<'static, ()>,
}

/// A case of an operation of both APIs. Their results are not compared: the recording server
/// answers a catalog operation with no rows, apart from the current catalog and database, so
/// some fail.
macro_rules! case {
    ($name:expr, |$o:ident| $old:expr, |$n:ident| $new:expr) => {
        Case {
            name: $name,
            old: |$o| {
                let _ = $old;
            },
            new: |$n| {
                Box::pin(async move {
                    let _ = $new.await;
                })
            },
        }
    };
}

/// Collects `dataframe`, the DataFrame of an old listing, which sends its request.
fn collected<E>(dataframe: Result<spark_connect::DataFrame, E>) {
    if let Ok(dataframe) = dataframe {
        let _ = dataframe.collect();
    }
}

fn schema() -> DataType {
    DataType::Struct {
        fields: vec![StructField {
            name: "id".to_string(),
            data_type: DataType::Integer,
            nullable: true,
            metadata: BTreeMap::new(),
        }],
    }
}

fn pairs() -> HashMap<String, String> {
    HashMap::from([
        ("a".to_string(), "1".to_string()),
        ("b".to_string(), "2".to_string()),
    ])
}

fn cases() -> Vec<Case> {
    vec![
        case!("current_catalog", |o| o.catalog().current_catalog(), |s| s
            .catalog()
            .current_catalog()),
        case!(
            "set_current_catalog",
            |o| o.catalog().set_current_catalog("lake"),
            |s| s.catalog().set_current_catalog("lake")
        ),
        case!(
            "list_catalogs",
            |o| collected(o.catalog().list_catalogs()),
            |s| s.catalog().list_catalogs()
        ),
        case!(
            "list_catalogs_with_pattern",
            |o| collected(o.catalog().list_catalogs_with_pattern(Some("s*"))),
            |s| s.catalog().list_catalogs().pattern("s*")
        ),
        case!(
            "list_catalogs_typed",
            |o| o.catalog().list_catalogs_typed(Some("s*")),
            |s| s.catalog().list_catalogs().pattern("s*")
        ),
        case!(
            "current_database",
            |o| o.catalog().current_database(),
            |s| s.catalog().current_database()
        ),
        case!(
            "set_current_database",
            |o| o.catalog().set_current_database("sales"),
            |s| s.catalog().set_current_database("sales")
        ),
        case!(
            "list_databases",
            |o| collected(o.catalog().list_databases()),
            |s| s.catalog().list_databases()
        ),
        case!(
            "list_databases_with_pattern",
            |o| collected(o.catalog().list_databases_with_pattern(Some("s*"))),
            |s| s.catalog().list_databases().pattern("s*")
        ),
        case!(
            "list_databases_typed",
            |o| o.catalog().list_databases_typed(None),
            |s| s.catalog().list_databases()
        ),
        case!(
            "get_database",
            |o| collected(o.catalog().get_database("sales")),
            |s| s.catalog().get_database("sales")
        ),
        case!(
            "get_database_typed",
            |o| o.catalog().get_database_typed("lake.sales"),
            |s| s.catalog().get_database("lake.sales")
        ),
        case!(
            "database_exists",
            |o| o.catalog().database_exists("sales"),
            |s| s.catalog().database_exists("sales")
        ),
        case!(
            "list_tables",
            |o| collected(o.catalog().list_tables()),
            |s| s.catalog().list_tables()
        ),
        case!(
            "list_tables_in_database",
            |o| collected(o.catalog().list_tables_in_database("sales")),
            |s| s.catalog().list_tables().database("sales")
        ),
        case!(
            "list_tables_with_pattern",
            |o| collected(
                o.catalog()
                    .list_tables_with_pattern(Some("sales"), Some("o*"))
            ),
            |s| s.catalog().list_tables().database("sales").pattern("o*")
        ),
        case!(
            "list_tables_typed",
            |o| o.catalog().list_tables_typed(None, Some("o*")),
            |s| s.catalog().list_tables().pattern("o*")
        ),
        case!(
            "get_table",
            |o| collected(o.catalog().get_table("orders")),
            |s| s.catalog().get_table("orders")
        ),
        case!(
            "get_table_with_database",
            |o| collected(o.catalog().get_table_with_database("orders", Some("sales"))),
            |s| s.catalog().get_table("orders").database("sales")
        ),
        case!(
            "get_table_typed",
            |o| o.catalog().get_table_typed("sales.orders"),
            |s| s.catalog().get_table("sales.orders")
        ),
        case!(
            "table_exists",
            |o| o.catalog().table_exists("orders"),
            |s| s.catalog().table_exists("orders")
        ),
        case!(
            "table_exists_with_database",
            |o| o
                .catalog()
                .table_exists_with_database("orders", Some("sales")),
            |s| s.catalog().table_exists("orders").database("sales")
        ),
        case!(
            "list_columns",
            |o| collected(o.catalog().list_columns("orders")),
            |s| s.catalog().list_columns("orders")
        ),
        case!(
            "list_columns_with_database",
            |o| collected(
                o.catalog()
                    .list_columns_with_database("orders", Some("sales"))
            ),
            |s| s.catalog().list_columns("orders").database("sales")
        ),
        case!(
            "list_columns_typed",
            |o| o.catalog().list_columns_typed("orders", None),
            |s| s.catalog().list_columns("orders")
        ),
        case!(
            "list_functions",
            |o| collected(o.catalog().list_functions()),
            |s| s.catalog().list_functions()
        ),
        case!(
            "list_functions_in_database",
            |o| collected(o.catalog().list_functions_in_database("sales")),
            |s| s.catalog().list_functions().database("sales")
        ),
        case!(
            "list_functions_with_pattern",
            |o| collected(o.catalog().list_functions_with_pattern(None, Some("t*"))),
            |s| s.catalog().list_functions().pattern("t*")
        ),
        case!(
            "list_functions_typed",
            |o| o.catalog().list_functions_typed(Some("sales"), Some("t*")),
            |s| s.catalog().list_functions().database("sales").pattern("t*")
        ),
        case!(
            "get_function",
            |o| collected(o.catalog().get_function("tax")),
            |s| s.catalog().get_function("tax")
        ),
        case!(
            "get_function_with_database",
            |o| collected(o.catalog().get_function_with_database("tax", Some("sales"))),
            |s| s.catalog().get_function("tax").database("sales")
        ),
        case!(
            "get_function_typed",
            |o| o.catalog().get_function_typed("tax"),
            |s| s.catalog().get_function("tax")
        ),
        case!(
            "function_exists",
            |o| o.catalog().function_exists("tax"),
            |s| s.catalog().function_exists("tax")
        ),
        case!(
            "function_exists_with_database",
            |o| o
                .catalog()
                .function_exists_with_database("tax", Some("sales")),
            |s| s.catalog().function_exists("tax").database("sales")
        ),
        case!(
            "create_table",
            |o| o.catalog().create_table("orders", None, None, None),
            |s| s.catalog().create_table("orders")
        ),
        case!(
            "create_table with a path, a source and a description",
            |o| o
                .catalog()
                .create_table("orders", Some("/data"), Some("csv"), Some("all")),
            |s| s
                .catalog()
                .create_table("orders")
                .path("/data")
                .source("csv")
                .description("all")
        ),
        case!(
            "create_table_with",
            |o| o.catalog().create_table_with(
                "orders",
                Some("/data"),
                Some("csv"),
                Some(&schema()),
                Some("all"),
                pairs()
            ),
            |s| s
                .catalog()
                .create_table("orders")
                .path("/data")
                .source("csv")
                .schema(schema())
                .description("all")
                .option("a", 1)
                .options([("b", "2")])
        ),
        case!(
            "drop_temp_view",
            |o| o.catalog().drop_temp_view("recent"),
            |s| s.catalog().drop_temp_view("recent")
        ),
        case!(
            "drop_global_temp_view",
            |o| o.catalog().drop_global_temp_view("recent"),
            |s| s.catalog().drop_global_temp_view("recent")
        ),
        case!("cache_table", |o| o.catalog().cache_table("orders"), |s| s
            .catalog()
            .cache_table("orders")),
        case!(
            "cache_table_with_storage_level",
            |o| o.catalog().cache_table_with_storage_level(
                "orders",
                Some(proto::StorageLevel {
                    use_disk: true,
                    use_memory: false,
                    use_off_heap: false,
                    deserialized: false,
                    replication: 2,
                })
            ),
            |s| s
                .catalog()
                .cache_table("orders")
                .storage_level(StorageLevel::DISK_ONLY_2)
        ),
        case!(
            "uncache_table",
            |o| o.catalog().uncache_table("orders"),
            |s| s.catalog().uncache_table("orders")
        ),
        case!("is_cached", |o| o.catalog().is_cached("orders"), |s| s
            .catalog()
            .is_cached("orders")),
        case!("clear_cache", |o| o.catalog().clear_cache(), |s| s
            .catalog()
            .clear_cache()),
        case!(
            "refresh_table",
            |o| o.catalog().refresh_table("orders"),
            |s| s.catalog().refresh_table("orders")
        ),
        case!(
            "refresh_by_path",
            |o| o.catalog().refresh_by_path("/data"),
            |s| s.catalog().refresh_by_path("/data")
        ),
        case!(
            "recover_partitions",
            |o| o.catalog().recover_partitions("orders"),
            |s| s.catalog().recover_partitions("orders")
        ),
        case!(
            "create_database",
            |o| o.catalog().create_database("sales", false, HashMap::new()),
            |s| s.catalog().create_database("sales")
        ),
        case!(
            "create_database if it does not exist, with properties",
            |o| o.catalog().create_database("sales", true, pairs()),
            |s| s
                .catalog()
                .create_database("sales")
                .if_not_exists()
                .property("b", "2")
                .property("a", "1")
        ),
        case!(
            "drop_database",
            |o| o.catalog().drop_database("sales", false, false),
            |s| s.catalog().drop_database("sales")
        ),
        case!(
            "drop_database if it exists, with its tables",
            |o| o.catalog().drop_database("sales", true, true),
            |s| s.catalog().drop_database("sales").if_exists().cascade()
        ),
        case!(
            "drop_table",
            |o| o.catalog().drop_table("orders", false, false),
            |s| s.catalog().drop_table("orders")
        ),
        case!(
            "drop_table if it exists, purged",
            |o| o.catalog().drop_table("orders", true, true),
            |s| s.catalog().drop_table("orders").if_exists().purge()
        ),
        case!(
            "drop_view",
            |o| o.catalog().drop_view("recent", false),
            |s| s.catalog().drop_view("recent")
        ),
        case!(
            "drop_view if it exists",
            |o| o.catalog().drop_view("recent", true),
            |s| s.catalog().drop_view("recent").if_exists()
        ),
        case!(
            "truncate_table",
            |o| o.catalog().truncate_table("orders"),
            |s| s.catalog().truncate_table("orders")
        ),
        case!(
            "analyze_table",
            |o| o.catalog().analyze_table("orders", false),
            |s| s.catalog().analyze_table("orders")
        ),
        case!(
            "analyze_table without a scan",
            |o| o.catalog().analyze_table("orders", true),
            |s| s.catalog().analyze_table("orders").no_scan()
        ),
        case!(
            "get_create_table_string",
            |o| o.catalog().get_create_table_string("orders", false),
            |s| s.catalog().get_create_table_string("orders")
        ),
        case!(
            "get_create_table_string as SerDe",
            |o| o.catalog().get_create_table_string("orders", true),
            |s| s.catalog().get_create_table_string("orders").as_serde()
        ),
        case!(
            "get_table_properties",
            |o| o.catalog().get_table_properties("orders"),
            |s| s.catalog().get_table_properties("orders")
        ),
        case!(
            "list_partitions",
            |o| collected(o.catalog().list_partitions("orders")),
            |s| s.catalog().list_partitions("orders")
        ),
        case!(
            "list_partitions_typed",
            |o| o.catalog().list_partitions_typed("orders"),
            |s| s.catalog().list_partitions("orders")
        ),
        case!(
            "list_views",
            |o| collected(o.catalog().list_views(None, None)),
            |s| s.catalog().list_views()
        ),
        case!(
            "list_views of a database, with a pattern",
            |o| collected(o.catalog().list_views(Some("sales"), Some("r*"))),
            |s| s.catalog().list_views().database("sales").pattern("r*")
        ),
        case!(
            "list_views_typed",
            |o| o.catalog().list_views_typed(Some("sales"), None),
            |s| s.catalog().list_views().database("sales")
        ),
    ]
}

/// The relation of `request` if it executes a catalog operation.
fn catalog_relation(request: &mut proto::ExecutePlanRequest) -> Option<&mut proto::Relation> {
    match request.plan.as_mut()?.op_type.as_mut()? {
        proto::plan::OpType::Root(relation)
            if matches!(
                relation.rel_type,
                Some(proto::relation::RelType::Catalog(_))
            ) =>
        {
            Some(relation)
        }
        _ => None,
    }
}

/// Each entry as its message, without the operation id, which the client chooses at random,
/// and without the plan id of a catalog relation: the old API sends one only with its listings
/// and its creation of a table, and
/// `every_operation_sends_a_plan_id_and_the_tags_of_its_handle` checks those of the async API.
fn messages(entries: Vec<server::Entry>) -> Vec<String> {
    entries
        .into_iter()
        .map(|entry| {
            if entry.kind != "ExecutePlanRequest" {
                return format!("{entry:?}");
            }
            let mut request = proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap();
            request.operation_id = None;
            if let Some(common) = catalog_relation(&mut request).and_then(|r| r.common.as_mut()) {
                common.plan_id = None;
            }
            format!("{request:?}")
        })
        .collect()
}

fn start() -> (Arc<server::Recorder>, String) {
    let recorder = Arc::new(server::Recorder::default());
    let address = server::start(Arc::clone(&recorder));
    let url = format!("sc://{address}/;session_id={SESSION_ID};user_id=snapshots");
    (recorder, url)
}

#[tokio::test]
async fn the_catalog_sends_the_requests_of_the_old_api() {
    let (recorder, url) = start();
    let mut both = Vec::new();
    for case in cases() {
        let old_session = SparkSession::builder()
            .remote(&url)
            .get_or_create()
            .unwrap();
        (case.old)(&old_session);
        let old = messages(recorder.take());
        let session = Session::builder()
            .remote(url.clone())
            .connect()
            .await
            .unwrap();
        (case.new)(session).await;
        let new = messages(recorder.take());
        assert!(!old.is_empty(), "{}: the old API sent nothing", case.name);
        both.push((case.name, old, new));
    }
    let differences = support::differences(both);
    assert!(differences.is_empty(), "{differences}");
}

#[tokio::test]
async fn every_operation_sends_a_plan_id_and_the_tags_of_its_handle() {
    let (recorder, url) = start();
    let session = Session::builder().remote(url).connect().await.unwrap();
    let nightly = session.with_tag("nightly").unwrap();
    let mut failures = Vec::new();
    for case in cases() {
        (case.new)(nightly.clone()).await;
        let entries = recorder.take();
        if entries.is_empty() {
            failures.push(format!("{}: no request", case.name));
        }
        for entry in entries {
            let mut request = proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap();
            let tags = std::mem::take(&mut request.tags);
            let plan_id = catalog_relation(&mut request)
                .and_then(|relation| relation.common.as_ref())
                .and_then(|common| common.plan_id);
            if tags != ["nightly"] || plan_id.is_none() {
                failures.push(format!("{}: tags {tags:?}, plan id {plan_id:?}", case.name));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[tokio::test]
async fn a_created_table_is_read_by_the_dataframe_that_its_creation_returns() {
    let (recorder, url) = start();
    let session = Session::builder().remote(url).connect().await.unwrap();
    let created = session
        .catalog()
        .create_table("sales.orders")
        .source("parquet")
        .await
        .unwrap();
    assert_eq!(recorder.take().len(), 1);
    assert_eq!(
        support::renumbered(&created.to_relation()),
        support::renumbered(&session.table("sales.orders").to_relation())
    );
}

#[tokio::test]
async fn views_matched_without_a_database_are_matched_by_the_server_in_the_current_one() {
    let (recorder, url) = start();
    let session = Session::builder().remote(url).connect().await.unwrap();
    session.catalog().list_views().pattern("r*").await.unwrap();
    let entries = recorder.take();
    let [entry] = &entries[..] else {
        panic!("not one request: {entries:?}");
    };
    let mut request = proto::ExecutePlanRequest::decode(&entry.bytes[..]).unwrap();
    let Some(proto::relation::RelType::Catalog(catalog)) =
        catalog_relation(&mut request).and_then(|relation| relation.rel_type.take())
    else {
        panic!("not a catalog operation: {request:?}");
    };
    let list = proto::ListViews {
        db_name: None,
        pattern: Some("r*".to_string()),
    };
    assert_eq!(
        catalog.cat_type,
        Some(proto::catalog::CatType::ListViews(list))
    );
}
