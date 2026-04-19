/// Structured error type returned by the workflow evaluator and persistence layer.
///
/// Use `WorkflowError::Evaluation` for ad-hoc evaluation errors (expected to be
/// replaced with typed variants in Stage 4/5 once the stringly-typed params are
/// gone). `Io`, `Json`, and `Other` exist so that `?` works transparently at
/// the I/O and anyhow boundaries without `.map_err(|e| e.to_string())`.
#[derive(Debug, thiserror::Error)]
pub enum WorkflowError {
    /// Ad-hoc evaluation error (e.g. missing input, type mismatch on a port).
    #[error("{0}")]
    Evaluation(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<String> for WorkflowError {
    fn from(s: String) -> Self {
        Self::Evaluation(s)
    }
}

pub type WorkflowResult<T> = Result<T, WorkflowError>;
