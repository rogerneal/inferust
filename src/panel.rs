//! Panel-data estimators.
//!
//! Currently supports entity fixed effects (within) and random effects
//! (Swamy–Arora / GLS quasi-demeaning), plus a Hausman test comparing the two.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector};
use statrs::distribution::{ChiSquared, ContinuousCDF};

use crate::error::{InferustError, Result};
use crate::regression::{Ols, OlsResult};

/// One-way panel OLS builder (entity FE and RE).
#[derive(Debug, Clone)]
pub struct PanelOls {
    feature_names: Vec<String>,
}

impl Default for PanelOls {
    fn default() -> Self {
        Self::new()
    }
}

impl PanelOls {
    pub fn new() -> Self {
        Self {
            feature_names: Vec::new(),
        }
    }

    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = names;
        self
    }

    /// Fit `y ~ x` with entity fixed effects removed by within transformation.
    ///
    /// Returns an [`OlsResult`] with **no intercept** (absorbed by the entity
    /// means). Standard errors are the OLS-within SEs (not the linearmodels
    /// within-df correction).
    pub fn fit_entity_fe(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        entities: &[usize],
    ) -> Result<OlsResult> {
        let (x_dm, y_dm) = within_transform(x, y, entities)?;
        Ols::new()
            .with_feature_names(self.feature_names.clone())
            .no_intercept()
            .fit(&x_dm, &y_dm)
    }

    /// Fit entity random-effects GLS via the Swamy–Arora estimator.
    ///
    /// Matches `linearmodels.panel.RandomEffects` with a constant term
    /// (`cov_type="unadjusted"`, `debiased=True`, default variance components).
    /// Coefficients are ordered as `[intercept, …slopes]`.
    pub fn fit_random_effects(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        entities: &[usize],
    ) -> Result<PanelReResult> {
        let (groups, p) = validate_panel(x, y, entities)?;
        let n = y.len();
        let n_entities = groups.len();
        if n_entities <= p + 1 {
            return Err(InferustError::InsufficientData {
                needed: p + 2,
                got: n_entities,
            });
        }

        // Entity means and within demeaning.
        let mut y_bar = vec![0.0; n];
        let mut x_bar = vec![vec![0.0; p]; n];
        let mut entity_y_mean = BTreeMap::new();
        let mut entity_x_mean = BTreeMap::new();
        let mut t_counts = BTreeMap::new();
        for (&e, idx) in &groups {
            let t_i = idx.len() as f64;
            t_counts.insert(e, t_i);
            let mean_y: f64 = idx.iter().map(|&i| y[i]).sum::<f64>() / t_i;
            entity_y_mean.insert(e, mean_y);
            let mut mean_x = vec![0.0; p];
            for j in 0..p {
                mean_x[j] = idx.iter().map(|&i| x[i][j]).sum::<f64>() / t_i;
            }
            entity_x_mean.insert(e, mean_x.clone());
            for &i in idx {
                y_bar[i] = mean_y;
                x_bar[i] = mean_x.clone();
            }
        }

        let y_gm: f64 = y.iter().sum::<f64>() / n as f64;
        let mut x_gm = vec![0.0; p];
        for row in x {
            for j in 0..p {
                x_gm[j] += row[j];
            }
        }
        for v in &mut x_gm {
            *v /= n as f64;
        }

        // Within regression with intercept restored via grand means (matches
        // linearmodels when has_constant=True).
        let y_w: Vec<f64> = y
            .iter()
            .zip(y_bar.iter())
            .map(|(&yi, &yb)| yi - yb + y_gm)
            .collect();
        let x_w: Vec<Vec<f64>> = x
            .iter()
            .zip(x_bar.iter())
            .map(|(xi, xb)| {
                let mut row = Vec::with_capacity(p + 1);
                row.push(1.0);
                for j in 0..p {
                    row.push(xi[j] - xb[j] + x_gm[j]);
                }
                row
            })
            .collect();
        let within = Ols::new().no_intercept().fit(&x_w, &y_w)?;
        let nvar = p + 1;
        let sigma2_e = within.ssr / (n - nvar - n_entities + 1) as f64;

        // Between regression on entity means.
        let ents: Vec<usize> = groups.keys().copied().collect();
        let y_b: Vec<f64> = ents.iter().map(|e| entity_y_mean[e]).collect();
        let x_b: Vec<Vec<f64>> = ents
            .iter()
            .map(|e| {
                let mut row = Vec::with_capacity(p + 1);
                row.push(1.0);
                row.extend_from_slice(&entity_x_mean[e]);
                row
            })
            .collect();
        let between = Ols::new().no_intercept().fit(&x_b, &y_b)?;
        let ssr_b = between.ssr;
        let t_vec: Vec<f64> = ents.iter().map(|e| t_counts[e]).collect();
        let t_bar = n_entities as f64 / t_vec.iter().map(|t| 1.0 / t).sum::<f64>();
        let sigma2_u = (ssr_b / (n_entities - nvar) as f64 - sigma2_e / t_bar).max(0.0);

        // θ_i and quasi-demeaning.
        let mut theta_by_entity = BTreeMap::new();
        let mut theta = Vec::with_capacity(n_entities);
        for (e, &t_i) in ents.iter().zip(t_vec.iter()) {
            let th = 1.0 - (sigma2_e / (t_i * sigma2_u + sigma2_e)).sqrt();
            theta_by_entity.insert(*e, th);
            theta.push(th);
        }

        let mut y_q = vec![0.0; n];
        let mut x_q = vec![vec![0.0; p + 1]; n];
        for (i, &e) in entities.iter().enumerate() {
            let th = theta_by_entity[&e];
            y_q[i] = y[i] - th * entity_y_mean[&e];
            x_q[i][0] = 1.0 - th;
            for j in 0..p {
                x_q[i][j + 1] = x[i][j] - th * entity_x_mean[&e][j];
            }
        }

        let mut names = Vec::with_capacity(p + 1);
        names.push("intercept".into());
        if self.feature_names.len() == p {
            names.extend(self.feature_names.iter().cloned());
        } else {
            names.extend((1..=p).map(|j| format!("x{j}")));
        }
        let gls = Ols::new()
            .with_feature_names(names.clone())
            .no_intercept()
            .fit(&x_q, &y_q)?;

        Ok(PanelReResult {
            coefficients: gls.coefficients.clone(),
            std_errors: gls.std_errors.clone(),
            t_statistics: gls.t_statistics.clone(),
            p_values: gls.p_values.clone(),
            covariance_matrix: gls.covariance_matrix.clone(),
            r_squared: gls.r_squared,
            adj_r_squared: gls.adj_r_squared,
            ssr: gls.ssr,
            sigma2_e,
            sigma2_u,
            theta,
            entities: ents,
            n,
            n_entities,
            k: nvar,
            feature_names: names,
            fitted_values: gls.fitted_values,
            residuals: gls.residuals,
        })
    }
}

/// Random-effects estimation result (Swamy–Arora).
#[derive(Debug, Clone)]
pub struct PanelReResult {
    /// Coefficients with intercept first.
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub t_statistics: Vec<f64>,
    pub p_values: Vec<f64>,
    pub covariance_matrix: Vec<Vec<f64>>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub ssr: f64,
    /// Idiosyncratic variance σ²_e.
    pub sigma2_e: f64,
    /// Entity-effect variance σ²_u.
    pub sigma2_u: f64,
    /// Quasi-demeaning θ per entity (same order as [`Self::entities`]).
    pub theta: Vec<f64>,
    /// Entity ids corresponding to [`Self::theta`].
    pub entities: Vec<usize>,
    pub n: usize,
    pub n_entities: usize,
    /// Number of coefficients including intercept.
    pub k: usize,
    pub feature_names: Vec<String>,
    pub fitted_values: Vec<f64>,
    pub residuals: Vec<f64>,
}

impl PanelReResult {
    pub fn print(&self) {
        println!();
        println!("── Random Effects (Swamy–Arora) ───────────────────────────────────");
        println!(
            " n = {}   entities = {}   σ²_e = {:.6}   σ²_u = {:.6}",
            self.n, self.n_entities, self.sigma2_e, self.sigma2_u
        );
        let pct = self.sigma2_u / (self.sigma2_u + self.sigma2_e).max(1e-300);
        println!(" fraction of variance due to effects = {pct:.4}");
        println!(
            " R² = {:.4}   adj R² = {:.4}",
            self.r_squared, self.adj_r_squared
        );
        println!();
        println!(
            " {:<16} {:>12} {:>12} {:>10} {:>10}",
            "coef", "est", "std err", "t", "P>|t|"
        );
        println!(" {}", "-".repeat(64));
        for i in 0..self.coefficients.len() {
            let name = self.feature_names.get(i).map(String::as_str).unwrap_or("?");
            println!(
                " {:<16} {:>12.6} {:>12.6} {:>10.3} {:>10.4}",
                name,
                self.coefficients[i],
                self.std_errors[i],
                self.t_statistics[i],
                self.p_values[i]
            );
        }
    }
}

/// Hausman test comparing entity FE slopes to RE slopes.
#[derive(Debug, Clone)]
pub struct HausmanResult {
    pub statistic: f64,
    pub df: usize,
    pub p_value: f64,
    /// FE − RE slope differences (no intercept).
    pub diff: Vec<f64>,
}

impl HausmanResult {
    pub fn print(&self) {
        println!();
        println!("── Hausman test (FE vs RE) ────────────────────────────────────────");
        println!(
            " χ²({}) = {:.6}   p = {:.6}",
            self.df, self.statistic, self.p_value
        );
    }
}

/// Hausman specification test: H₀ that RE is consistent (and efficient).
///
/// Compares slope coefficients from [`PanelOls::fit_entity_fe`] and
/// [`PanelOls::fit_random_effects`]. The RE intercept is excluded.
pub fn hausman_fe_re(fe: &OlsResult, re: &PanelReResult) -> Result<HausmanResult> {
    let k = fe.coefficients.len();
    if re.coefficients.len() != k + 1 {
        return Err(InferustError::InvalidInput(
            "RE result must have intercept + same slopes as FE".into(),
        ));
    }
    if k == 0 {
        return Err(InferustError::InvalidInput(
            "Hausman test needs at least one slope".into(),
        ));
    }
    let mut diff = Vec::with_capacity(k);
    for i in 0..k {
        diff.push(fe.coefficients[i] - re.coefficients[i + 1]);
    }
    // V = V_fe − V_re[slopes, slopes]
    let mut v = DMatrix::<f64>::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            v[(i, j)] = fe.covariance_matrix[i][j] - re.covariance_matrix[i + 1][j + 1];
        }
    }
    // Symmetrize and stabilize for the inverse.
    for i in 0..k {
        for j in 0..i {
            let s = 0.5 * (v[(i, j)] + v[(j, i)]);
            v[(i, j)] = s;
            v[(j, i)] = s;
        }
        v[(i, i)] += 1e-14;
    }
    let d = DVector::from_vec(diff.clone());
    let v_inv = v.clone().try_inverse().ok_or_else(|| {
        InferustError::InvalidInput(
            "Hausman covariance difference is singular; try a different sample".into(),
        )
    })?;
    let statistic = (d.transpose() * v_inv * d)[(0, 0)];
    let statistic = statistic.max(0.0);
    let chi2 = ChiSquared::new(k as f64)
        .map_err(|_| InferustError::InvalidInput("invalid chi-square df".into()))?;
    let p_value = 1.0 - chi2.cdf(statistic);
    Ok(HausmanResult {
        statistic,
        df: k,
        p_value,
        diff,
    })
}

/// Build composite cluster ids for two-way clustering (entity × time).
pub fn two_way_cluster_ids(entities: &[usize], times: &[usize]) -> Vec<usize> {
    entities
        .iter()
        .zip(times)
        .map(|(&e, &t)| e.wrapping_mul(1_000_003).wrapping_add(t))
        .collect()
}

fn validate_panel<'a>(
    x: &'a [Vec<f64>],
    y: &'a [f64],
    entities: &'a [usize],
) -> Result<(BTreeMap<usize, Vec<usize>>, usize)> {
    if y.is_empty() {
        return Err(InferustError::InsufficientData { needed: 1, got: 0 });
    }
    if entities.len() != y.len() || x.len() != y.len() {
        return Err(InferustError::DimensionMismatch {
            x_rows: x.len().max(entities.len()),
            y_len: y.len(),
        });
    }
    let p = x[0].len();
    for row in x {
        if row.len() != p {
            return Err(InferustError::InvalidInput(
                "all covariate rows must have the same width".into(),
            ));
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err(InferustError::InvalidInput(
                "covariates must be finite".into(),
            ));
        }
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(InferustError::InvalidInput(
            "response values must be finite".into(),
        ));
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, &e) in entities.iter().enumerate() {
        groups.entry(e).or_default().push(i);
    }
    if groups.len() < 2 {
        return Err(InferustError::InsufficientData {
            needed: 2,
            got: groups.len(),
        });
    }
    Ok((groups, p))
}

fn within_transform(
    x: &[Vec<f64>],
    y: &[f64],
    entities: &[usize],
) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    let (groups, p) = validate_panel(x, y, entities)?;
    let mut y_dm = y.to_vec();
    let mut x_dm = x.to_vec();
    for idx in groups.values() {
        let n_i = idx.len() as f64;
        let mean_y: f64 = idx.iter().map(|&i| y[i]).sum::<f64>() / n_i;
        for &i in idx {
            y_dm[i] -= mean_y;
        }
        for j in 0..p {
            let mean_x: f64 = idx.iter().map(|&i| x[i][j]).sum::<f64>() / n_i;
            for &i in idx {
                x_dm[i][j] -= mean_x;
            }
        }
    }
    Ok((x_dm, y_dm))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_panel() -> (Vec<Vec<f64>>, Vec<f64>, Vec<usize>) {
        // Balanced 4 entities × 3 periods, strong entity effect.
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut entities = Vec::new();
        for e in 0..4 {
            let ae = e as f64;
            for t in 0..3 {
                let x1 = (t + 1) as f64 + 0.1 * e as f64;
                let x2 = (t as f64) - 0.2 * e as f64;
                x.push(vec![x1, x2]);
                y.push(1.0 + ae + 0.5 * x1 - 0.25 * x2);
                entities.push(e);
            }
        }
        (x, y, entities)
    }

    #[test]
    fn random_effects_estimates_slopes() {
        let (x, y, entities) = toy_panel();
        let re = PanelOls::new()
            .fit_random_effects(&x, &y, &entities)
            .unwrap();
        assert_eq!(re.coefficients.len(), 3);
        assert!(re.sigma2_e.is_finite());
        assert!(re.sigma2_u.is_finite());
        assert!(re.theta.iter().all(|t| t.is_finite()));
        assert!(re.std_errors.iter().all(|s| s.is_finite() && *s >= 0.0));
    }

    #[test]
    fn hausman_runs_on_fe_and_re() {
        let (x, y, entities) = toy_panel();
        let fe = PanelOls::new().fit_entity_fe(&x, &y, &entities).unwrap();
        let re = PanelOls::new()
            .fit_random_effects(&x, &y, &entities)
            .unwrap();
        let h = hausman_fe_re(&fe, &re).unwrap();
        assert_eq!(h.df, 2);
        assert!(h.statistic.is_finite());
        assert!((0.0..=1.0).contains(&h.p_value));
    }
}
