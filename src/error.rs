use thiserror::Error;

#[derive(Error, Debug)]
pub enum InferustError {
    #[error("insufficient data: need at least {needed} observations, got {got}")]
    InsufficientData { needed: usize, got: usize },

    #[error("dimension mismatch: X has {x_rows} rows but y has {y_len} elements")]
    DimensionMismatch { x_rows: usize, y_len: usize },

    #[error("singular matrix: the design matrix is not invertible")]
    SingularMatrix,

    #[error("invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, InferustError>;
