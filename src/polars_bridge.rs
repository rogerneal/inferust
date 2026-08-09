//! Optional Polars bridge (`inferust` feature `polars`).

/// Build an inferust [`crate::data::DataFrame`] from selected Polars columns.
#[cfg(feature = "polars")]
pub fn from_polars(
    frame: &polars::prelude::DataFrame,
    columns: &[&str],
) -> crate::Result<crate::data::DataFrame> {
    use crate::error::InferustError;
    use polars::prelude::*;
    let height = frame.height();
    let mut out = crate::data::DataFrame::new();
    for &name in columns {
        let series = frame
            .column(name)
            .map_err(|e| InferustError::InvalidInput(e.to_string()))?;
        match series.dtype() {
            DataType::Float64 => {
                let vals = series
                    .f64()
                    .map_err(|e| InferustError::InvalidInput(e.to_string()))?
                    .into_no_null_iter()
                    .collect::<Vec<_>>();
                if vals.len() != height {
                    return Err(InferustError::InvalidInput(format!(
                        "column `{name}` has nulls; call drop_nulls first"
                    )));
                }
                out = out.with_column(name, vals)?;
            }
            DataType::String => {
                let ca = series
                    .str()
                    .map_err(|e| InferustError::InvalidInput(e.to_string()))?;
                let mut vals = Vec::with_capacity(height);
                for opt in ca.iter() {
                    let Some(s) = opt else {
                        return Err(InferustError::InvalidInput(format!(
                            "column `{name}` has nulls; call drop_nulls first"
                        )));
                    };
                    vals.push(s.to_string());
                }
                out = out.with_categorical_column(name, vals)?;
            }
            other => {
                return Err(InferustError::InvalidInput(format!(
                    "unsupported Polars dtype for `{name}`: {other:?}"
                )));
            }
        }
    }
    Ok(out)
}
