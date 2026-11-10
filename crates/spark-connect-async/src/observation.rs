//! Observations: metrics of a query's rows, collected while an action runs it.

use spark_connect_core::observation::Observation as CoreObservation;

use crate::error::Result;
use crate::value::Row;

/// The metrics that [`DataFrame::observe`](crate::DataFrame::observe) collects, which each
/// action that runs the observed DataFrame sets. Clones share the metrics.
#[derive(Debug, Clone, Default)]
pub struct Observation(CoreObservation);

impl Observation {
    /// An observation named `name`.
    pub fn new(name: &str) -> Self {
        Observation(CoreObservation::new(name))
    }

    /// The name of the observation; one without a name gets a random one when it is first
    /// observed.
    pub fn name(&self) -> Option<String> {
        self.0.name()
    }

    /// The metrics of the latest action, as a row of a column per metric.
    ///
    /// Fails with the condition `NO_OBSERVE_BEFORE_GET` for an observation of no DataFrame.
    pub fn get(&self) -> Result<Row> {
        let metrics = self.0.get()?;
        Ok(crate::value::row_of_metrics(
            metrics.iter().map(|(name, _)| name),
            metrics.iter().map(|(_, value)| value),
        ))
    }

    pub(crate) fn into_core(self) -> CoreObservation {
        self.0
    }
}

impl From<&str> for Observation {
    fn from(name: &str) -> Self {
        Observation::new(name)
    }
}

impl From<String> for Observation {
    fn from(name: String) -> Self {
        Observation::new(&name)
    }
}

impl From<&Observation> for Observation {
    fn from(observation: &Observation) -> Self {
        observation.clone()
    }
}
