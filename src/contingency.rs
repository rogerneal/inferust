//! Contingency table statistics.

use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

use crate::error::{InferustError, Result};

/// 2x2 table effect estimates.
#[derive(Debug, Clone)]
pub struct Table2x2 {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub odds_ratio: f64,
    pub risk_ratio: f64,
    pub risk_difference: f64,
    pub log_odds_ratio_se: f64,
}

/// McNemar paired-binary test.
#[derive(Debug, Clone)]
pub struct McNemarResult {
    pub statistic: f64,
    pub p_value: f64,
    pub discordant_pairs: f64,
}

/// Cochran-Mantel-Haenszel common odds-ratio test.
#[derive(Debug, Clone)]
pub struct CmhResult {
    pub statistic: f64,
    pub p_value: f64,
    pub common_odds_ratio: f64,
    pub strata: usize,
}

/// Analyze a 2x2 table `[[a, b], [c, d]]`.
pub fn table2x2(table: [[f64; 2]; 2]) -> Result<Table2x2> {
    let [a, b, c, d] = validate_2x2(table)?;
    let aa = if a == 0.0 { 0.5 } else { a };
    let bb = if b == 0.0 { 0.5 } else { b };
    let cc = if c == 0.0 { 0.5 } else { c };
    let dd = if d == 0.0 { 0.5 } else { d };
    let odds_ratio = (aa * dd) / (bb * cc);
    let risk_treated = aa / (aa + bb);
    let risk_control = cc / (cc + dd);
    let risk_ratio = risk_treated / risk_control.max(1e-12);
    let risk_difference = risk_treated - risk_control;
    let log_odds_ratio_se = (1.0 / aa + 1.0 / bb + 1.0 / cc + 1.0 / dd).sqrt();
    Ok(Table2x2 {
        a,
        b,
        c,
        d,
        odds_ratio,
        risk_ratio,
        risk_difference,
        log_odds_ratio_se,
    })
}

/// Wald confidence interval for the odds ratio.
pub fn odds_ratio_ci(table: [[f64; 2]; 2], alpha: f64) -> Result<(f64, f64)> {
    if !(0.0..1.0).contains(&alpha) {
        return Err(InferustError::InvalidInput(
            "alpha must be between 0 and 1".into(),
        ));
    }
    let result = table2x2(table)?;
    let normal = Normal::new(0.0, 1.0)
        .map_err(|_| InferustError::InvalidInput("invalid normal distribution".into()))?;
    let critical = normal.inverse_cdf(1.0 - alpha / 2.0);
    let center = result.odds_ratio.ln();
    Ok((
        (center - critical * result.log_odds_ratio_se).exp(),
        (center + critical * result.log_odds_ratio_se).exp(),
    ))
}

/// McNemar test for paired 2x2 tables.
pub fn mcnemar(table: [[f64; 2]; 2], correction: bool) -> Result<McNemarResult> {
    let [_a, b, c, _d] = validate_2x2(table)?;
    let discordant = b + c;
    if discordant == 0.0 {
        return Ok(McNemarResult {
            statistic: 0.0,
            p_value: 1.0,
            discordant_pairs: 0.0,
        });
    }
    let numerator = if correction {
        (b - c).abs().max(1.0) - 1.0
    } else {
        b - c
    };
    let statistic = numerator.powi(2) / discordant;
    let chi2 = ChiSquared::new(1.0)
        .map_err(|_| InferustError::InvalidInput("invalid chi-squared df".into()))?;
    Ok(McNemarResult {
        statistic,
        p_value: 1.0 - chi2.cdf(statistic),
        discordant_pairs: discordant,
    })
}

/// Cochran-Mantel-Haenszel test across 2x2 strata.
pub fn cochran_mantel_haenszel(strata: &[[[f64; 2]; 2]]) -> Result<CmhResult> {
    if strata.is_empty() {
        return Err(InferustError::InsufficientData { needed: 1, got: 0 });
    }
    let mut numerator = 0.0;
    let mut variance = 0.0;
    let mut or_num = 0.0;
    let mut or_den = 0.0;
    for &table in strata {
        let [a, b, c, d] = validate_2x2(table)?;
        let n = a + b + c + d;
        if n <= 1.0 {
            continue;
        }
        let r1 = a + b;
        let r2 = c + d;
        let c1 = a + c;
        let c2 = b + d;
        let expected = r1 * c1 / n;
        let var = r1 * r2 * c1 * c2 / (n.powi(2) * (n - 1.0));
        numerator += a - expected;
        variance += var;
        or_num += a * d / n;
        or_den += b * c / n;
    }
    let statistic = numerator.powi(2) / variance.max(1e-12);
    let chi2 = ChiSquared::new(1.0)
        .map_err(|_| InferustError::InvalidInput("invalid chi-squared df".into()))?;
    Ok(CmhResult {
        statistic,
        p_value: 1.0 - chi2.cdf(statistic),
        common_odds_ratio: or_num / or_den.max(1e-12),
        strata: strata.len(),
    })
}

fn validate_2x2(table: [[f64; 2]; 2]) -> Result<[f64; 4]> {
    let values = [table[0][0], table[0][1], table[1][0], table[1][1]];
    if values.iter().any(|v| *v < 0.0 || !v.is_finite()) {
        return Err(InferustError::InvalidInput(
            "contingency counts must be finite and non-negative".into(),
        ));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::{cochran_mantel_haenszel, mcnemar, odds_ratio_ci, table2x2};

    #[test]
    fn table2x2_reports_effect_sizes() {
        let result = table2x2([[12.0, 5.0], [4.0, 20.0]]).unwrap();
        assert!(result.odds_ratio > 10.0);
        assert!(result.risk_ratio > 1.0);
        let ci = odds_ratio_ci([[12.0, 5.0], [4.0, 20.0]], 0.05).unwrap();
        assert!(ci.0 < result.odds_ratio && ci.1 > result.odds_ratio);
    }

    #[test]
    fn mcnemar_detects_discordance() {
        let result = mcnemar([[20.0, 12.0], [3.0, 25.0]], true).unwrap();
        assert!(result.statistic > 0.0);
        assert_eq!(result.discordant_pairs, 15.0);
    }

    #[test]
    fn cmh_combines_strata() {
        let result =
            cochran_mantel_haenszel(&[[[8.0, 2.0], [3.0, 9.0]], [[10.0, 4.0], [5.0, 12.0]]])
                .unwrap();
        assert_eq!(result.strata, 2);
        assert!(result.common_odds_ratio > 1.0);
    }
}
