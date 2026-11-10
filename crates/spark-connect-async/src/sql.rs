//! SQL statements with arguments.

use std::collections::HashMap;
use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::Arc;

use spark_connect_core::actions;
use spark_connect_core::expression::Expression;
use spark_connect_core::plan::{relation_with_ids, LogicalPlan};

use crate::column::IntoArgument;
use crate::dataframe::DataFrame;
use crate::error::Result;
use crate::session::Session;

/// The SQL statement that [`Session::sql`] returns: add its arguments, then await it to send
/// it.
///
/// A statement takes either positional arguments, `?` in its text, or named ones, `:name`.
/// Each argument is a Rust value, as [`lit`](crate::lit) makes it, or a column of literals,
/// such as an array.
pub struct Sql {
    session: Session,
    query: String,
    positional: Vec<Expression>,
    named: HashMap<String, Expression>,
    references: Vec<Arc<LogicalPlan>>,
}

impl Sql {
    pub(crate) fn new(session: Session, query: String) -> Self {
        Sql {
            session,
            query,
            positional: Vec::new(),
            named: HashMap::new(),
            references: Vec::new(),
        }
    }

    /// The value of the next `?` of the statement.
    pub fn arg(mut self, value: impl IntoArgument) -> Self {
        self.positional.push(argument(value));
        self
    }

    /// The value of `:name` in the statement.
    pub fn named_arg(mut self, name: impl Into<String>, value: impl IntoArgument) -> Self {
        self.named.insert(name.into(), argument(value));
        self
    }

    /// `dataframe` as the relation `name` in the statement. The server creates `name` as a
    /// temporary view while the statement runs, replacing a view of that name, and drops it
    /// after; so `name` must not be that of a view to keep.
    pub fn reference(mut self, name: impl Into<String>, dataframe: &DataFrame) -> Self {
        self.references
            .push(Arc::clone(dataframe.alias(name).plan()));
        self
    }

    fn plan(self) -> (Session, Arc<LogicalPlan>) {
        let sql = Arc::new(LogicalPlan::Sql {
            query: self.query,
            pos_args: self.positional,
            named_args: self.named,
        });
        let plan = if self.references.is_empty() {
            sql
        } else {
            Arc::new(LogicalPlan::WithRelations {
                root: sql,
                references: self.references,
            })
        };
        (self.session, plan)
    }
}

fn argument(value: impl IntoArgument) -> Expression {
    value.into_argument().expression().clone()
}

impl IntoFuture for Sql {
    type Output = Result<DataFrame>;
    type IntoFuture = Pin<Box<dyn Future<Output = Result<DataFrame>> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        let (session, plan) = self.plan();
        let relation = relation_with_ids(&plan, session.plan_ids());
        Box::pin(async move {
            let returned = actions::session::sql(
                session.client(),
                Arc::clone(session.plan_ids()),
                session.tags().to_vec(),
                relation,
                (),
            )
            .await?;
            Ok(match returned {
                Some(relation) => DataFrame::new(
                    session,
                    LogicalPlan::CachedRelation {
                        relation: Box::new(relation),
                    },
                ),
                // A server that returns no relation runs a query at its actions.
                None => DataFrame::new(session, plan),
            })
        })
    }
}
