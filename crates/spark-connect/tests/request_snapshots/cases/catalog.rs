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

use std::collections::HashMap;

use spark_connect::StorageLevel;

use crate::Case;

pub fn cases() -> Vec<Case> {
    cases![
        "Catalog::current_catalog" => |c| c.spark.catalog().current_catalog().map(drop),
        "Catalog::set_current_catalog" => |c| c.spark.catalog().set_current_catalog("cat"),
        "Catalog::list_catalogs" => |c| c.df(&c.spark.catalog().list_catalogs()?),
        "Catalog::list_catalogs_with_pattern" => |c| c.df(&c.spark.catalog().list_catalogs_with_pattern(Some("c*"))?),
        "Catalog::current_database" => |c| c.spark.catalog().current_database().map(drop),
        "Catalog::set_current_database" => |c| c.spark.catalog().set_current_database("db"),
        "Catalog::list_databases" => |c| c.df(&c.spark.catalog().list_databases()?),
        "Catalog::list_databases_with_pattern" => |c| c.df(&c.spark.catalog().list_databases_with_pattern(Some("d*"))?),
        "Catalog::get_database" => |c| c.df(&c.spark.catalog().get_database("db")?),
        "Catalog::database_exists" => |c| c.spark.catalog().database_exists("db").map(drop),
        "Catalog::list_tables" => |c| c.df(&c.spark.catalog().list_tables()?),
        "Catalog::list_tables_in_database" => |c| c.df(&c.spark.catalog().list_tables_in_database("db")?),
        "Catalog::list_tables_with_pattern" => |c| c.df(&c.spark.catalog().list_tables_with_pattern(Some("db"), Some("t*"))?),
        "Catalog::list_tables_with_pattern/none" => |c| c.df(&c.spark.catalog().list_tables_with_pattern(None, None)?),
        "Catalog::get_table" => |c| c.df(&c.spark.catalog().get_table("db.t")?),
        "Catalog::get_table_with_database" => |c| c.df(&c.spark.catalog().get_table_with_database("t", Some("db"))?),
        "Catalog::table_exists" => |c| c.spark.catalog().table_exists("t").map(drop),
        "Catalog::table_exists_with_database" => |c| c.spark.catalog().table_exists_with_database("t", Some("db")).map(drop),
        "Catalog::list_columns" => |c| c.df(&c.spark.catalog().list_columns("t")?),
        "Catalog::list_columns_with_database" => |c| c.df(&c.spark.catalog().list_columns_with_database("t", Some("db"))?),
        "Catalog::list_functions" => |c| c.df(&c.spark.catalog().list_functions()?),
        "Catalog::list_functions_in_database" => |c| c.df(&c.spark.catalog().list_functions_in_database("db")?),
        "Catalog::list_functions_with_pattern" => |c| c.df(&c.spark.catalog().list_functions_with_pattern(Some("db"), Some("f*"))?),
        "Catalog::get_function" => |c| c.df(&c.spark.catalog().get_function("f")?),
        "Catalog::get_function_with_database" => |c| c.df(&c.spark.catalog().get_function_with_database("f", Some("db"))?),
        "Catalog::function_exists" => |c| c.spark.catalog().function_exists("f").map(drop),
        "Catalog::function_exists_with_database" => |c| c.spark.catalog().function_exists_with_database("f", Some("db")).map(drop),
        "Catalog::create_external_table" => |c| c.df(&c.spark.catalog().create_external_table("t", Some("/p"), Some("parquet"))?),
        "Catalog::create_table" => |c| c.df(&c.spark.catalog().create_table("t", Some("/p"), Some("delta"), Some("desc"))?),
        "Catalog::create_table/minimal" => |c| c.df(&c.spark.catalog().create_table("t", None, None, None)?),
        "Catalog::drop_temp_view" => |c| c.spark.catalog().drop_temp_view("v").map(drop),
        "Catalog::drop_global_temp_view" => |c| c.spark.catalog().drop_global_temp_view("v").map(drop),
        "Catalog::cache_table" => |c| c.spark.catalog().cache_table("t"),
        "Catalog::cache_table_with_storage_level" => |c| {
            let level = StorageLevel { use_disk: true, use_memory: true, use_off_heap: false, deserialized: false, replication: 1 };
            c.spark.catalog().cache_table_with_storage_level("t", Some(level))?;
            c.spark.catalog().cache_table_with_storage_level("t", None)
        },
        "Catalog::uncache_table" => |c| c.spark.catalog().uncache_table("t"),
        "Catalog::is_cached" => |c| c.spark.catalog().is_cached("t").map(drop),
        "Catalog::clear_cache" => |c| c.spark.catalog().clear_cache(),
        "Catalog::refresh_table" => |c| c.spark.catalog().refresh_table("t"),
        "Catalog::refresh_by_path" => |c| c.spark.catalog().refresh_by_path("/p"),
        "Catalog::recover_partitions" => |c| c.spark.catalog().recover_partitions("t"),
        "Catalog::create_database" => {
            |c| c.spark.catalog().create_database("db", true, HashMap::from([("k".to_string(), "v".to_string())]))
        },
        "Catalog::create_database/plain" => |c| c.spark.catalog().create_database("db", false, HashMap::new()),
        "Catalog::drop_database" => |c| c.spark.catalog().drop_database("db", true, true),
        "Catalog::drop_table" => |c| c.spark.catalog().drop_table("t", true, false),
        "Catalog::drop_view" => |c| c.spark.catalog().drop_view("v", false),
        "Catalog::truncate_table" => |c| c.spark.catalog().truncate_table("t"),
        "Catalog::analyze_table" => |c| c.spark.catalog().analyze_table("t", true),
        "Catalog::get_create_table_string" => |c| c.spark.catalog().get_create_table_string("t", false).map(drop),
        "Catalog::get_table_properties" => |c| c.spark.catalog().get_table_properties("t").map(drop),
        "Catalog::list_partitions" => |c| c.df(&c.spark.catalog().list_partitions("t")?),
        "Catalog::list_views" => |c| c.df(&c.spark.catalog().list_views(Some("db"), Some("v*"))?),
        "Catalog::list_catalogs_typed" => |c| c.spark.catalog().list_catalogs_typed(None).map(drop),
        "Catalog::list_databases_typed" => |c| c.spark.catalog().list_databases_typed(Some("d")).map(drop),
        "Catalog::get_database_typed" => |c| c.spark.catalog().get_database_typed("db").map(drop),
        "Catalog::list_tables_typed" => |c| c.spark.catalog().list_tables_typed(Some("db"), None).map(drop),
        "Catalog::get_table_typed" => |c| c.spark.catalog().get_table_typed("t").map(drop),
        "Catalog::list_functions_typed" => |c| c.spark.catalog().list_functions_typed(None, Some("f")).map(drop),
        "Catalog::get_function_typed" => |c| c.spark.catalog().get_function_typed("f").map(drop),
        "Catalog::list_columns_typed" => |c| c.spark.catalog().list_columns_typed("t", Some("db")).map(drop),
        "Catalog::list_partitions_typed" => |c| c.spark.catalog().list_partitions_typed("t").map(drop),
        "Catalog::list_views_typed" => |c| c.spark.catalog().list_views_typed(None, None).map(drop),
        "Catalog::create_external_table_with" => |c| {
            let schema = spark_connect::DataType::from_ddl("a INT")?;
            let options = HashMap::from([("k".to_string(), "v".to_string())]);
            c.spark.catalog().create_external_table_with("t", Some("/p"), Some("parquet"), Some(&schema), options).map(drop)
        },
        "Catalog::create_table_with" => |c| {
            let schema = spark_connect::DataType::from_ddl("a INT")?;
            let options = HashMap::from([("k".to_string(), "v".to_string())]);
            c.spark.catalog().create_table_with("t", Some("/p"), Some("parquet"), Some(&schema), Some("d"), options).map(drop)
        },
        "Catalog::spark_session" => |c| c.df(&c.spark.catalog().spark_session().range(1)?),
    ]
}
