use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};

use crate::error::{InferustError, Result};
use crate::regression::Ols;

#[derive(Debug, Clone)]
pub struct VifResult {
    pub variable: String,
    pub r_squared: f64,
    pub vif: f64,
}

#[derive(Debug, Clone)]
pub struct DiagnosticTest {
    pub statistic: f64,
    pub p_value: f64,
    pub f_statistic: Option<f64>,
    pub f_p_value: Option<f64>,
}

pub fn variance_inflation_factors(
    x: &[Vec<f64>],
    names: Option<&[String]>,
) -> Result<Vec<VifResult>> {
    if x.is_empty() {
        return Err(InferustError::InsufficientData { needed: 1, got: 0 });
    }
    let p = x[0].len();
    if p < 2 {
        return Err(InferustError::InvalidInput(
            "VIF requires at least two predictors".into(),
        ));
    }
    let mut results = Vec::with_capacity(p);
    for target_idx in 0..p {
        let y = x.iter().map(|row| row[target_idx]).collect::<Vec<_>>();
        let predictors = x
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter_map(|(j, value)| (j != target_idx).then_some(*value))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let fit = Ols::new().fit(&predictors, &y)?;
        let r_squared = fit.r_squared.min(1.0 - 1e-12);
        let variable = names
            .and_then(|items| items.get(target_idx))
            .cloned()
            .unwrap_or_else(|| format!("x{}", target_idx + 1));
        results.push(VifResult {
            variable,
            r_squared,
            vif: 1.0 / (1.0 - r_squared),
        });
    }
    Ok(results)
}

pub fn breusch_pagan(residuals: &[f64], x: &[Vec<f64>]) -> Result<DiagnosticTest> {
    let squared = residuals
        .iter()
        .map(|value| value.powi(2))
        .collect::<Vec<_>>();
    let aux = Ols::new().fit(x, &squared)?;
    let lm = residuals.len() as f64 * aux.r_squared;
    let df = x[0].len();
    let chi = ChiSquared::new(df as f64)
        .map_err(|_| InferustError::InvalidInput("invalid chi-squared distribution".into()))?;
    Ok(DiagnosticTest {
        statistic: lm,
        p_value: 1.0 - chi.cdf(lm),
        f_statistic: Some(aux.f_statistic),
        f_p_value: Some(aux.f_p_value),
    })
}

pub fn white_test(residuals: &[f64], x: &[Vec<f64>]) -> Result<DiagnosticTest> {
    let expanded = white_design(x);
    let squared = residuals
        .iter()
        .map(|value| value.powi(2))
        .collect::<Vec<_>>();
    let aux = Ols::new().fit(&expanded, &squared)?;
    let lm = residuals.len() as f64 * aux.r_squared;
    let df = expanded[0].len();
    let chi = ChiSquared::new(df as f64)
        .map_err(|_| InferustError::InvalidInput("invalid chi-squared distribution".into()))?;
    Ok(DiagnosticTest {
        statistic: lm,
        p_value: 1.0 - chi.cdf(lm),
        f_statistic: Some(aux.f_statistic),
        f_p_value: Some(aux.f_p_value),
    })
}

pub fn reset_test(y: &[f64], x: &[Vec<f64>], powers: &[u32]) -> Result<DiagnosticTest> {
    let base = Ols::new().fit(x, y)?;
    let mut augmented = x.to_vec();
    for (row, fitted) in augmented.iter_mut().zip(base.fitted_values.iter()) {
        for power in powers {
            row.push(fitted.powi(*power as i32));
        }
    }
    let full = Ols::new().fit(&augmented, y)?;
    let q = powers.len() as f64;
    let numerator = (base.ssr - full.ssr) / q;
    let denominator = full.ssr / full.df_resid as f64;
    let f_statistic = numerator / denominator;
    let fisher = FisherSnedecor::new(q, full.df_resid as f64)
        .map_err(|_| InferustError::InvalidInput("invalid F distribution".into()))?;
    Ok(DiagnosticTest {
        statistic: f_statistic,
        p_value: 1.0 - fisher.cdf(f_statistic),
        f_statistic: Some(f_statistic),
        f_p_value: Some(1.0 - fisher.cdf(f_statistic)),
    })
}

fn white_design(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| {
            let mut out = row.clone();
            for i in 0..row.len() {
                for j in i..row.len() {
                    out.push(row[i] * row[j]);
                }
            }
            out
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{breusch_pagan, variance_inflation_factors, white_test};
    use crate::regression::Ols;

    #[test]
    fn computes_vif_and_heteroskedasticity_tests() {
        let x = vec![
            vec![0.2, 1.3],
            vec![1.1, 0.7],
            vec![1.7, 2.9],
            vec![2.4, 1.8],
            vec![3.2, 3.7],
            vec![4.1, 2.6],
            vec![4.8, 4.4],
            vec![5.6, 3.1],
            vec![6.3, 5.2],
        ];
        let y = vec![2.4, 2.9, 5.6, 5.9, 8.9, 9.0, 12.1, 12.0, 15.5];
        let fit = Ols::new().fit(&x, &y).unwrap();
        let vif = variance_inflation_factors(&x, None).unwrap();
        assert_eq!(vif.len(), 2);
        assert!(vif[0].vif >= 1.0);
        assert!(breusch_pagan(&fit.residuals, &x).unwrap().statistic >= 0.0);
        assert!(white_test(&fit.residuals, &x).unwrap().statistic >= 0.0);
    }
}
