# Catalog

`session.catalog()` is the catalog of the session: its catalogs, databases, tables, views and
functions, and the tables it caches. Each call is a future, and its answer is typed: a
`Table`, a `Database`, a `CatalogColumn`, a `Function`, a `bool` or a `String`. An optional
argument, such as the database to look in or a pattern of names, is a builder method of the
call.

```{.rust .async}
let catalog = session.catalog();
println!("in {}.{}", catalog.current_catalog().await?, catalog.current_database().await?);

for table in catalog.list_tables().database("default").await? {
    println!("{} ({}, temporary: {})", table.name, table.table_type, table.is_temporary);
}
for column in catalog.list_columns("people").await? {
    println!("{}: {} (nullable: {})", column.name, column.data_type, column.nullable);
}
if catalog.table_exists("people").await? {
    let people = catalog.get_table("people").await?;
    println!("{:?}", people.description);
}
let functions = catalog.list_functions().pattern("date*").await?;
```

A field that Spark may leave out, such as a description, is an `Option`.

## Creating and Dropping

```{.rust .async}
let catalog = session.catalog();
catalog.create_database("sales").if_not_exists().await?;
let events = catalog
    .create_table("sales.events")
    .source("json")
    .path("/data/events")
    .await?;
catalog.drop_table("sales.old_events").if_exists().await?;
catalog.drop_database("archive").if_exists().cascade().await?;
```

`create_table` returns a DataFrame of the new table.

## Caching

```{.rust .async}
let catalog = session.catalog();
catalog.cache_table("people").await?;
assert!(catalog.is_cached("people").await?);
catalog.uncache_table("people").await?;
catalog.clear_cache().await?;
```

`refresh_table`, `refresh_by_path` and `recover_partitions` update what Spark knows of a
table's data.
