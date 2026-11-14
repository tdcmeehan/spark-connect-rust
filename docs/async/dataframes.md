# DataFrames, Columns and SQL

A `DataFrame` is the plan of a query and the session that runs it. Transformations return a
new DataFrame and send nothing; [actions](results.md) send the plan.

## Where DataFrames Come From

```{.rust .async}
use spark_connect_async::options::RangeSpec;

let first_ten = session.range(10);
let even = session.range(RangeSpec::new(0..100).step(2).partitions(4));
let people = session.table("people");
let adults = session.sql("SELECT * FROM people WHERE age >= 18").await?;
```

`range(10)` is 0 to 9 and `range(5..10)` 5 to 9, in a `BIGINT` column `id`. `session.read()`
reads data sources, and `create_dataframe` makes a DataFrame of local rows (see
[Reading and Writing](data-sources.md)).

## SQL

Awaiting `sql` sends the statement: a command, such as `CREATE VIEW`, runs then, and a query
returns a DataFrame that runs at its actions. Arguments are Rust values, positional (`?`) or
named (`:name`), and `reference` passes a DataFrame as a relation of the statement:

```{.rust .async}
let adults = session
    .sql("SELECT name FROM people WHERE age >= :age AND city = :city")
    .named_arg("age", 18)
    .named_arg("city", "Utrecht")
    .await?;

let orders = session.table("orders");
let big = session
    .sql("SELECT * FROM orders_view WHERE amount > ?")
    .arg(100.0)
    .reference("orders_view", &orders)
    .await?;
```

A `&str` argument is a string value, never SQL text, so arguments need no quoting.

## Columns

A `Column` is an expression. Wherever a column is taken, `impl Into<Column>`, a `&str` or a
`String` is the column of that name, and `lit` makes a literal of a Rust value (see
[Types and Values](types.md#literals)):

```{.rust .async}
use spark_connect_async::{col, lit, when};

let total = col("price") * col("quantity");
let adult = col("age").ge(lit(18));
let label = when(col("age").lt(lit(13)), lit("child"))
    .when(col("age").lt(lit(18)), lit("teenager"))
    .otherwise(lit("adult"));
let typed = col("amount").cast("decimal(10, 2)");
let renamed = col("name").alias("label");
```

Comparisons and logic are methods (`eq`, `ne`, `lt`, `le`, `gt`, `ge`, `and`, `or`,
`is_null`, `is_in`, `between`, `like`, ...), since Rust's comparison operators return `bool`.
Arithmetic uses the operators `+`, `-`, `*`, `/` and `%`, `&` and `|` are `and` and `or`, and
`!` is `NOT`. `cast` and `try_cast` take a `DataType` or a type in Spark SQL's syntax.

## Functions

`spark_connect_async::functions` has Spark's SQL functions, each documented with the
description of its SQL function. A parameter takes what its SQL argument is:

- a column (`impl Into<Column>`), where a `&str` is the column of that name;
- a Rust value, such as `&str`, `i32` or `f64`, where the argument is always a literal;
- a value or a column (`IntoArgument`), where a `&str` is a string;
- an `Option` for a trailing optional argument, where `None` leaves it out.

```{.rust .async}
use spark_connect_async::functions::{concat_ws, round, split, to_date, upper};

let people = session.table("people");
let shaped = people.select([
    upper("name").alias("name"),
    concat_ws(" ", ["first_name", "last_name"]).alias("full_name"),
    split("tags", ",", None).alias("tags"),
    round("score", Some(1)).alias("score"),
    to_date("joined", Some("yyyy-MM-dd")).alias("joined"),
]);
```

`expr` parses a column from SQL text: `expr("age + 1")`.

## Transformations

```{.rust .async}
use spark_connect_async::{col, lit};

let people = session.table("people");
let shaped = people
    .filter(col("age").ge(lit(18)))
    .with_column("decade", (col("age") / lit(10)).cast("int") * lit(10))
    .with_column_renamed("name", "full_name")
    .drop(["email"])
    .sort([col("age").desc(), col("full_name").asc()])
    .limit(100);
let unique = people.select(["city"]).distinct();
let both = df1.union(&df2);
```

Others follow the same shape: `select_expr`, `with_columns`, `drop_duplicates`, `sample`,
`random_split`, `repartition`, `coalesce`, `hint`, `with_watermark`, `unpivot`, `transpose`,
`describe`, `summary`, `intersect`, `subtract` and more. The arguments of an operation with
options are a value that has their defaults, with builder methods for the rest, in
`spark_connect_async::options`.

## Joins

`join` joins on a condition, and `join_using` on columns of the same names. When both
DataFrames have a column of the same name, `DataFrame::col` says which one a column is:

```{.rust .async}
use spark_connect_async::options::JoinType;

let people = session.table("people");
let orders = session.table("orders");
let with_orders = people.join(
    &orders,
    people.col("id").eq(orders.col("person_id")),
    JoinType::LeftOuter,
);
let by_id = people.join_using(&session.table("addresses"), ["id"], JoinType::Inner);
```

`cross_join`, `lateral_join`, `as_of_join` and `nearest_by_join` are the other joins.

## Grouping and Aggregation

```{.rust .async}
use spark_connect_async::functions::{avg, count, max, sum};

let orders = session.table("orders");
let per_customer = orders.group_by(["customer"]).agg([
    sum("amount").alias("total"),
    avg("amount").alias("average"),
    max("amount").alias("largest"),
    count("id").alias("orders"),
]);
let per_day = orders.group_by(["day"]).count();
let everything = orders.agg([sum("amount").alias("total")]);
```

`rollup`, `cube` and `grouping_sets` group the same way, and `pivot` turns the values of a
column into columns. The `sum`, `avg`, `min` and `max` of `GroupedData` aggregate the numeric
columns they are given, or every numeric column, and are futures, since they read the
DataFrame's schema first:

```{.rust .async}
let sums = session.table("orders").group_by(["customer"]).sum(["amount"]).await?;
```

## Windows

A window function's column takes a `Window`: the rows of a row's partition, in an order, within
a frame.

```{.rust .async}
use spark_connect_async::col;
use spark_connect_async::functions::{lag, rank, sum};
use spark_connect_async::window::{FrameBound, Window};

let by_department = Window::new()
    .partition_by(["department"])
    .order_by([col("salary").desc()]);
let ranked = session.table("employees").select([
    col("name"),
    rank().over(by_department.clone()).alias("rank"),
    lag("salary", Some(1), None).over(by_department.clone()).alias("next_higher"),
    sum("salary")
        .over(by_department.rows_between(FrameBound::UnboundedPreceding, FrameBound::CurrentRow))
        .alias("running_total"),
]);
```

## Statistics and Missing Values

`stat()` has the statistics of a DataFrame's columns, and `na()` the handling of their missing
values:

```{.rust .async}
let measurements = session.table("measurements");
let correlation: f64 = measurements.stat().corr("height", "weight").await?;
let counts = measurements.stat().crosstab("site", "kind");
```

`corr`, `cov` and `approx_quantile` are futures, since the server answers them; `crosstab`,
`freq_items` and `sample_by` are DataFrames.
