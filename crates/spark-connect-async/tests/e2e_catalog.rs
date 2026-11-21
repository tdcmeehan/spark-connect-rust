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

//! The catalog of the async API against a Spark Connect server at `SPARK_REMOTE`; each test
//! passes without running when it is not set.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use spark_connect_async::types::{DataType, StructField};
use spark_connect_async::{ErrorKind, Session};

async fn session() -> Option<Session> {
    let remote = std::env::var("SPARK_REMOTE").ok()?;
    Some(Session::builder().remote(remote).connect().await.unwrap())
}

#[tokio::test]
async fn the_results_of_the_server_are_read_into_rust_types_with_their_nulls() {
    let Some(session) = session().await else {
        return;
    };
    let catalog = session.catalog();
    // A run that fails leaves the directories of its tables in the warehouse of the server.
    let started = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let db = format!("b9_sales_{}", started.as_nanos());
    let orders = format!("{db}.orders");
    catalog
        .create_database(&db)
        .property("owner", "b9")
        .await
        .unwrap();
    for statement in [
        format!(
            "CREATE TABLE {orders} (id INT COMMENT 'the id', day STRING) USING parquet \
             PARTITIONED BY (day) TBLPROPERTIES ('kind' = 'fact')"
        ),
        format!("INSERT INTO {orders} PARTITION (day = '2024-01-01') VALUES (1)"),
        "CREATE OR REPLACE TEMPORARY VIEW b9_recent AS SELECT 1 AS id".to_string(),
    ] {
        session.sql(statement).await.unwrap();
    }

    assert_eq!(catalog.current_catalog().await.unwrap(), "spark_catalog");
    assert_eq!(catalog.current_database().await.unwrap(), "default");
    let catalogs = catalog.list_catalogs().await.unwrap();
    assert!(
        catalogs
            .iter()
            .any(|catalog| catalog.name == "spark_catalog"),
        "{catalogs:?}"
    );

    let database = catalog.get_database(&db).await.unwrap();
    assert_eq!(database.name, db);
    assert_eq!(database.catalog.as_deref(), Some("spark_catalog"));
    assert!(database.location_uri.is_some(), "{database:?}");
    let databases = catalog.list_databases().pattern(&db).await.unwrap();
    assert_eq!(databases, [database]);
    assert!(catalog.database_exists(&db).await.unwrap());

    let tables = catalog.list_tables().database(&db).await.unwrap();
    let names: Vec<_> = tables.iter().map(|table| &table.name[..]).collect();
    assert_eq!(names, ["orders", "b9_recent"]);
    let (table, listed_view) = (&tables[0], &tables[1]);
    assert_eq!(
        (
            &table.name[..],
            table.catalog.as_deref(),
            table.namespace.as_deref(),
            table.description.as_deref(),
            &table.table_type[..],
            table.is_temporary
        ),
        (
            "orders",
            Some("spark_catalog"),
            Some(&[db.clone()][..]),
            None,
            "MANAGED",
            false
        )
    );
    let got = catalog.get_table("orders").database(&db).await.unwrap();
    assert_eq!(got, *table);
    assert!(catalog.table_exists("orders").database(&db).await.unwrap());
    let recent = catalog.get_table("b9_recent").await.unwrap();
    assert_eq!(
        (
            recent.catalog.as_deref(),
            recent.namespace.as_deref(),
            recent.description.as_deref(),
            &recent.table_type[..],
            recent.is_temporary
        ),
        (None, Some(&[][..]), None, "TEMPORARY", true)
    );
    assert_eq!(*listed_view, recent);
    let views = catalog.list_views().pattern("b9_*").await.unwrap();
    assert!(views.contains(&recent), "{views:?}");

    let columns = catalog.list_columns(&orders).await.unwrap();
    let columns: Vec<_> = columns
        .iter()
        .map(|column| {
            (
                &column.name[..],
                column.description.as_deref(),
                &column.data_type[..],
                column.is_partition,
                column.is_bucket,
            )
        })
        .collect();
    assert_eq!(
        columns,
        [
            ("id", Some("the id"), "int", false, false),
            ("day", None, "string", true, false)
        ]
    );

    let abs = catalog.get_function("abs").await.unwrap();
    assert_eq!(
        (abs.catalog, abs.namespace, abs.is_temporary),
        (None, None, true)
    );
    assert!(abs.class_name.is_some_and(|name| name.ends_with(".Abs")));
    let functions = catalog.list_functions().pattern("ab*").await.unwrap();
    assert!(functions.iter().any(|function| function.name == "abs"));
    assert!(catalog.function_exists("abs").await.unwrap());

    let partitions = catalog.list_partitions(&orders).await.unwrap();
    let partitions: Vec<_> = partitions.iter().map(|p| &p.partition[..]).collect();
    assert_eq!(partitions, ["day=2024-01-01"]);
    let properties = catalog.get_table_properties(&orders).await.unwrap();
    assert_eq!(properties.get("kind").map(String::as_str), Some("fact"));
    let statement = catalog.get_create_table_string(&orders).await.unwrap();
    assert!(statement.starts_with("CREATE TABLE"), "{statement}");

    catalog.cache_table(&orders).await.unwrap();
    assert!(catalog.is_cached(&orders).await.unwrap());
    catalog.uncache_table(&orders).await.unwrap();
    assert!(!catalog.is_cached(&orders).await.unwrap());
    catalog.refresh_table(&orders).await.unwrap();
    catalog.recover_partitions(&orders).await.unwrap();
    catalog.analyze_table(&orders).no_scan().await.unwrap();
    catalog.truncate_table(&orders).await.unwrap();
    assert_eq!(session.table(&orders).count().await.unwrap(), 0);

    let id = StructField {
        name: "id".to_string(),
        data_type: DataType::Integer,
        nullable: true,
        metadata: BTreeMap::new(),
    };
    let made_name = format!("{db}.made");
    let made = catalog
        .create_table(&made_name)
        .source("parquet")
        .schema(DataType::Struct { fields: vec![id] })
        .description("made here")
        .await
        .unwrap();
    assert_eq!(made.count().await.unwrap(), 0);
    let made = catalog.get_table(&made_name).await.unwrap();
    assert_eq!(made.description.as_deref(), Some("made here"));

    assert!(catalog.drop_temp_view("b9_recent").await.unwrap());
    assert!(!catalog.drop_temp_view("b9_recent").await.unwrap());
    catalog
        .drop_table(&made_name)
        .if_exists()
        .purge()
        .await
        .unwrap();
    catalog.drop_database(&db).cascade().await.unwrap();
    assert!(!catalog.database_exists(&db).await.unwrap());
}

#[tokio::test]
async fn a_failed_operation_is_an_error_with_the_condition_of_the_server() {
    let Some(session) = session().await else {
        return;
    };
    let error = session
        .catalog()
        .get_table("b9_missing.orders")
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Analysis);
    assert_eq!(error.error_class(), Some("TABLE_OR_VIEW_NOT_FOUND"));
    assert!(!error.to_string().contains("Exception"), "{error}");
}
