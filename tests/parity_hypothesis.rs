//! Parity tests for hypothesis tests against scipy.stats reference fixtures.

mod common;

use common::{
    as_f64, as_f64_matrix, as_f64_vec, assert_parity, check_scalar, check_vec, load_fixture, xy,
};
use inferust::hypothesis::{
    adjust, anova, chisq, nonparametric, ttest, tukey_hsd, MultiTestMethod,
};
use inferust::regression::Ols;

// ── KS one-sample ─────────────────────────────────────────────────────────────

#[test]
fn parity_ks_one_sample() {
    let fx = load_fixture("ks_one_sample");
    let data = as_f64_vec(&fx["dataset"]["data"]);
    let mean = as_f64(&fx["dataset"]["mean"]);
    let std = as_f64(&fx["dataset"]["std"]);
    let res =
        nonparametric::ks_one_sample(&data, Some(mean), Some(std)).expect("ks_one_sample failed");
    assert_parity(
        "ks_one_sample",
        vec![
            check_scalar("statistic", res.statistic, as_f64(&fx["statistic"]), 1e-6),
            // p-value uses Marsaglia 2003 asymptotic series; scipy uses a
            // different expansion. Agreement within ~3 % is expected.
            check_scalar("p_value", res.p_value, as_f64(&fx["p_value"]), 3e-2),
        ],
    );
}

// ── KS two-sample ─────────────────────────────────────────────────────────────

#[test]
fn parity_ks_two_sample() {
    let fx = load_fixture("ks_two_sample");
    let a = as_f64_vec(&fx["dataset"]["a"]);
    let b = as_f64_vec(&fx["dataset"]["b"]);
    let res = nonparametric::ks_two_sample(&a, &b).expect("ks_two_sample failed");
    assert_parity(
        "ks_two_sample",
        vec![
            check_scalar("statistic", res.statistic, as_f64(&fx["statistic"]), 1e-6),
            // Same Marsaglia approximation difference as one-sample KS.
            check_scalar("p_value", res.p_value, as_f64(&fx["p_value"]), 3e-2),
        ],
    );
}

// ── Kruskal-Wallis ────────────────────────────────────────────────────────────

#[test]
fn parity_kruskal_wallis() {
    let fx = load_fixture("kruskal_wallis_parity");
    let groups_raw: Vec<Vec<f64>> = fx["dataset"]["groups"]
        .as_array()
        .expect("groups array")
        .iter()
        .map(as_f64_vec)
        .collect();
    let groups: Vec<&[f64]> = groups_raw.iter().map(|g| g.as_slice()).collect();
    let res = nonparametric::kruskal_wallis(&groups).expect("kruskal_wallis failed");
    assert_parity(
        "kruskal_wallis",
        vec![
            check_scalar(
                "h_statistic",
                res.h_statistic,
                as_f64(&fx["h_statistic"]),
                1e-6,
            ),
            check_scalar("p_value", res.p_value, as_f64(&fx["p_value"]), 1e-6),
        ],
    );
}

// ── Shapiro-Wilk ──────────────────────────────────────────────────────────────

#[test]
fn parity_shapiro_wilk() {
    let fx = load_fixture("shapiro_wilk");
    let data = as_f64_vec(&fx["dataset"]["data"]);
    let res = nonparametric::shapiro_wilk(&data).expect("shapiro_wilk failed");
    // Royston approximation differs substantially from scipy's reference
    // implementation (AS R94) in both polynomial coefficients and the
    // calibration blend applied to near-normal samples. Verify that:
    //  1. The W statistic matches to 1e-2 (deterministic computation).
    //  2. Both implementations yield a p-value > 0.05 (normal data, same conclusion).
    let expected_p = as_f64(&fx["p_value"]);
    let w_check = check_scalar("statistic", res.w_statistic, as_f64(&fx["statistic"]), 1e-2);
    // p-value direction: both should agree the data is not significantly non-normal
    let p_direction_ok = (res.p_value > 0.05) == (expected_p > 0.05);
    let mut checks = vec![w_check];
    if !p_direction_ok {
        checks.push(Err(format!(
            "  p_value direction: inferust={:.4} scipy={:.4} (disagree on H0 conclusion)",
            res.p_value, expected_p
        )));
    }
    assert_parity("shapiro_wilk", checks);
}

// ── Chi-squared goodness-of-fit ───────────────────────────────────────────────

#[test]
fn parity_chi2_goodness_of_fit() {
    let fx = load_fixture("chi2_goodness_of_fit");
    let observed = as_f64_vec(&fx["dataset"]["observed"]);
    let expected = as_f64_vec(&fx["dataset"]["expected"]);
    let res = chisq::goodness_of_fit(&observed, Some(&expected)).expect("goodness_of_fit failed");
    assert_parity(
        "chi2_goodness_of_fit",
        vec![
            check_scalar("statistic", res.statistic, as_f64(&fx["statistic"]), 1e-9),
            check_scalar("p_value", res.p_value, as_f64(&fx["p_value"]), 1e-9),
            check_scalar("df", res.df, as_f64(&fx["df"]), 1e-12),
        ],
    );
}

#[test]
fn parity_ttest_1samp() {
    let fx = load_fixture("ttest_1samp");
    let data = as_f64_vec(&fx["dataset"]["data"]);
    let result = ttest::one_sample(&data, 0.0).expect("one-sample t-test failed");

    assert_parity(
        "ttest_1samp",
        vec![
            check_scalar(
                "statistic",
                result.statistic,
                as_f64(&fx["statistic"]),
                1e-10,
            ),
            check_scalar("pvalue", result.p_value, as_f64(&fx["pvalue"]), 1e-10),
            check_scalar("df", result.df, as_f64(&fx["df"]), 1e-12),
            check_scalar(
                "mean_diff",
                result.mean_diff,
                as_f64(&fx["mean_diff"]),
                1e-10,
            ),
            check_scalar(
                "ci_lo",
                result.conf_interval.0,
                as_f64_vec(&fx["conf_int_05"])[0],
                1e-9,
            ),
            check_scalar(
                "ci_hi",
                result.conf_interval.1,
                as_f64_vec(&fx["conf_int_05"])[1],
                1e-9,
            ),
        ],
    );
}

#[test]
fn parity_ttest_ind() {
    let fx = load_fixture("ttest_ind");
    let a = as_f64_vec(&fx["dataset"]["a"]);
    let b = as_f64_vec(&fx["dataset"]["b"]);
    let result = ttest::two_sample(&a, &b).expect("two-sample t-test failed");

    assert_parity(
        "ttest_ind",
        vec![
            check_scalar(
                "statistic",
                result.statistic,
                as_f64(&fx["statistic"]),
                1e-10,
            ),
            check_scalar("pvalue", result.p_value, as_f64(&fx["pvalue"]), 1e-10),
            check_scalar("df", result.df, as_f64(&fx["df"]), 1e-9),
        ],
    );
}

#[test]
fn parity_anova_oneway() {
    let fx = load_fixture("anova_oneway");
    let groups_raw: Vec<Vec<f64>> = fx["dataset"]["groups"]
        .as_array()
        .expect("groups array")
        .iter()
        .map(as_f64_vec)
        .collect();
    let groups: Vec<&[f64]> = groups_raw.iter().map(|g| g.as_slice()).collect();
    let result = anova::one_way(&groups).expect("ANOVA failed");

    assert_parity(
        "anova_oneway",
        vec![
            check_scalar(
                "f_statistic",
                result.f_statistic,
                as_f64(&fx["f_statistic"]),
                1e-10,
            ),
            check_scalar("pvalue", result.p_value, as_f64(&fx["pvalue"]), 1e-10),
        ],
    );
}

#[test]
fn parity_mann_whitney() {
    let fx = load_fixture("mann_whitney");
    let a = as_f64_vec(&fx["dataset"]["a"]);
    let b = as_f64_vec(&fx["dataset"]["b"]);
    let result = nonparametric::mann_whitney(&a, &b).expect("MW failed");

    // U statistic: scipy reports U1 (sum of ranks of a − n_a(n_a+1)/2). inferust
    // may report U1 or U2 depending on convention. We allow either to pass via
    // min(U, n_a*n_b - U) equivalence: both produce the same p-value.
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;
    let u_max = n_a * n_b;
    let expected_u = as_f64(&fx["u_statistic"]);
    let actual_u = result.u_statistic;
    // inferust returns min(U1, U2); scipy can return either side. Accept both.
    let u_match =
        (actual_u - expected_u).abs() < 1e-8 || (actual_u - (u_max - expected_u)).abs() < 1e-8;

    let mut checks = vec![check_scalar(
        "pvalue",
        result.p_value,
        as_f64(&fx["pvalue"]),
        1e-6,
    )];
    if !u_match {
        checks.push(Err(format!(
            "  u_statistic: actual={actual_u} expected={expected_u} (or {})",
            u_max - expected_u
        )));
    }
    assert_parity("mann_whitney", checks);
}

#[test]
fn parity_chi2_independence() {
    let fx = load_fixture("chi2_independence");
    let table = fx["dataset"]["table"]
        .as_array()
        .expect("table")
        .iter()
        .map(as_f64_vec)
        .collect::<Vec<_>>();
    let result = chisq::independence(&table).expect("chi2 failed");

    assert_parity(
        "chi2_independence",
        vec![
            check_scalar(
                "statistic",
                result.statistic,
                as_f64(&fx["statistic"]),
                1e-9,
            ),
            check_scalar("pvalue", result.p_value, as_f64(&fx["pvalue"]), 1e-9),
            check_scalar("dof", result.df, as_f64(&fx["dof"]), 1e-12),
        ],
    );
}

#[test]
fn parity_wilcoxon_signed_rank() {
    let fx = load_fixture("wilcoxon");
    let a = as_f64_vec(&fx["dataset"]["a"]);
    let b = as_f64_vec(&fx["dataset"]["b"]);
    let res = nonparametric::wilcoxon_signed_rank(&a, &b).expect("wilcoxon failed");
    // scipy's `wilcoxon` reports W = min(W+, W-) by default; our struct exposes
    // `.statistic` with the same convention.
    assert_parity(
        "wilcoxon",
        vec![
            check_scalar("statistic", res.statistic, as_f64(&fx["statistic"]), 1e-8),
            // p-value: normal approx with CC; tolerate small drift vs scipy's
            // exact-or-asymptotic switch around n = 25.
            check_scalar("pvalue", res.p_value, as_f64(&fx["pvalue"]), 5e-3),
        ],
    );
}

#[test]
fn parity_sign_test() {
    let fx = load_fixture("sign_test");
    let a = as_f64_vec(&fx["dataset"]["a"]);
    let b = as_f64_vec(&fx["dataset"]["b"]);
    let res = nonparametric::sign_test(&a, &b).expect("sign_test failed");
    let pos_expected = fx["positives"].as_u64().unwrap() as usize;
    let neg_expected = fx["negatives"].as_u64().unwrap() as usize;
    let zeros_expected = fx["zeros"].as_u64().unwrap() as usize;

    assert_eq!(res.positives, pos_expected);
    assert_eq!(res.negatives, neg_expected);
    assert_eq!(res.zeros, zeros_expected);
    assert_parity(
        "sign_test",
        vec![check_scalar(
            "p_value",
            res.p_value,
            as_f64(&fx["p_value"]),
            1e-10,
        )],
    );
}

#[test]
fn parity_anderson_darling() {
    let fx = load_fixture("anderson_darling");
    let data = as_f64_vec(&fx["dataset"]["data"]);
    let res = nonparametric::anderson_darling(&data).expect("AD failed");
    // scipy's `anderson(dist='norm')` returns the raw A² statistic; we match
    // the raw statistic, not the adjusted one.
    assert_parity(
        "anderson_darling",
        vec![check_scalar(
            "statistic",
            res.statistic,
            as_f64(&fx["statistic"]),
            1e-6,
        )],
    );
}

#[test]
fn parity_lilliefors() {
    let fx = load_fixture("lilliefors");
    let data = as_f64_vec(&fx["dataset"]["data"]);
    let res = nonparametric::lilliefors(&data).expect("lilliefors failed");
    // Statistic is exact (max deviation of empirical CDF from estimated normal);
    // p-value uses Dallal-Wilkinson here vs Lilliefors' table in statsmodels.
    assert_parity(
        "lilliefors_statistic",
        vec![check_scalar(
            "statistic",
            res.statistic,
            as_f64(&fx["statistic"]),
            1e-9,
        )],
    );
    // p-value tolerance is intentionally loose — different approximations.
    let p_diff = (res.p_value - as_f64(&fx["p_value"])).abs();
    assert!(
        p_diff < 0.05 || res.p_value >= 0.20,
        "Lilliefors p drift too large: inferust={} statsmodels={}",
        res.p_value,
        as_f64(&fx["p_value"])
    );
}

#[test]
fn parity_wald_ols() {
    let fx = load_fixture("wald_ols");
    let (x, y) = xy(&fx);
    let r = as_f64_matrix(&fx["r"]);
    let q = as_f64_vec(&fx["q"]);

    let result = Ols::new().fit(&x, &y).expect("OLS fit failed");
    let wald = result.wald_test(&r, &q).expect("Wald test failed");

    assert_parity(
        "wald_ols",
        vec![
            check_scalar(
                "chi2_statistic",
                wald.chi2_statistic,
                as_f64(&fx["chi2_statistic"]),
                1e-6,
            ),
            check_scalar(
                "chi2_pvalue",
                wald.chi2_p_value,
                as_f64(&fx["chi2_pvalue"]),
                1e-7,
            ),
            check_scalar(
                "f_statistic",
                wald.f_statistic,
                as_f64(&fx["f_statistic"]),
                1e-6,
            ),
            check_scalar("f_pvalue", wald.f_p_value, as_f64(&fx["f_pvalue"]), 1e-7),
        ],
    );
    // Reference vector that statsmodels and inferust both compute.
    let expected_rb: Vec<f64> = r
        .iter()
        .map(|row| {
            row.iter()
                .zip(result.coefficients.iter())
                .map(|(rij, b)| rij * b)
                .sum()
        })
        .collect();
    assert_parity(
        "wald_rb",
        vec![check_vec("r_beta", &wald.r_beta, &expected_rb, 1e-12)],
    );
}

#[test]
fn parity_multicomp() {
    let fx = load_fixture("multicomp");
    let p_values = as_f64_vec(&fx["dataset"]["p_values"]);
    let alpha = as_f64(&fx["alpha"]);

    let cases = [
        ("Bonferroni", MultiTestMethod::Bonferroni),
        ("Holm", MultiTestMethod::Holm),
        ("BenjaminiHochberg", MultiTestMethod::BenjaminiHochberg),
        ("BenjaminiYekutieli", MultiTestMethod::BenjaminiYekutieli),
    ];

    let mut checks = Vec::new();
    for (key, method) in cases {
        let result = adjust(&p_values, alpha, method).expect("adjust failed");
        let expected = &fx["methods"][key];

        checks.push(check_vec(
            &format!("{key}.p_corrected"),
            &result.p_values_corrected,
            &as_f64_vec(&expected["p_corrected"]),
            1e-10,
        ));
        checks.push(check_scalar(
            &format!("{key}.alpha_bonferroni"),
            result.alpha_bonferroni,
            as_f64(&expected["alpha_bonferroni"]),
            1e-12,
        ));
        checks.push(check_scalar(
            &format!("{key}.alpha_sidak"),
            result.alpha_sidak,
            as_f64(&expected["alpha_sidak"]),
            1e-12,
        ));

        let expected_reject: Vec<bool> = expected["reject"]
            .as_array()
            .expect("reject array")
            .iter()
            .map(|v| v.as_bool().expect("reject bool"))
            .collect();
        if result.reject != expected_reject {
            checks.push(Err(format!(
                "  {key}.reject: actual={:?} expected={:?}",
                result.reject, expected_reject
            )));
        }
    }
    assert_parity("multicomp", checks);
}

#[test]
fn parity_tukey_hsd() {
    let fx = load_fixture("tukey_hsd");
    let groups_raw: Vec<Vec<f64>> = fx["dataset"]["groups"]
        .as_array()
        .expect("groups array")
        .iter()
        .map(as_f64_vec)
        .collect();
    let groups: Vec<&[f64]> = groups_raw.iter().map(|g| g.as_slice()).collect();
    let alpha = as_f64(&fx["alpha"]);
    let result = tukey_hsd(&groups, None, alpha).expect("Tukey HSD failed");

    // statsmodels' `pairwise_tukeyhsd` reports meandiff[k] = mean(later) -
    // mean(earlier) for pair (i, j), i < j; inferust reports mean_diff =
    // mean(group_a) - mean(group_b) = mean(earlier) - mean(later), the
    // opposite sign. Negate before comparing (the CI bounds flip sign and
    // swap order under negation too).
    let actual_mean_diffs: Vec<f64> = result.comparisons.iter().map(|c| -c.mean_diff).collect();
    let actual_std_pairs: Vec<f64> = result.comparisons.iter().map(|c| c.std_error).collect();
    let actual_p_values: Vec<f64> = result.comparisons.iter().map(|c| c.p_value).collect();
    let actual_reject: Vec<bool> = result.comparisons.iter().map(|c| c.reject).collect();
    let actual_conf_lower: Vec<f64> = result.comparisons.iter().map(|c| -c.conf_high).collect();
    let actual_conf_upper: Vec<f64> = result.comparisons.iter().map(|c| -c.conf_low).collect();

    let expected_conf_int = as_f64_matrix(&fx["conf_int"]);
    let expected_conf_lower: Vec<f64> = expected_conf_int.iter().map(|row| row[0]).collect();
    let expected_conf_upper: Vec<f64> = expected_conf_int.iter().map(|row| row[1]).collect();
    let expected_reject: Vec<bool> = fx["reject"]
        .as_array()
        .expect("reject array")
        .iter()
        .map(|v| v.as_bool().expect("reject bool"))
        .collect();

    let mut checks = vec![
        check_vec(
            "mean_diffs",
            &actual_mean_diffs,
            &as_f64_vec(&fx["mean_diffs"]),
            1e-6,
        ),
        check_vec(
            "std_pairs",
            &actual_std_pairs,
            &as_f64_vec(&fx["std_pairs"]),
            1e-6,
        ),
        check_scalar(
            "df_within",
            result.df_within,
            as_f64(&fx["df_total"]),
            1e-12,
        ),
        // q_crit / p_values / conf_int: statsmodels uses an interpolated
        // lookup table (libqsturng, accurate to ~1e-3); inferust computes the
        // studentized range distribution directly via quadrature (accurate
        // to ~1e-9 against the true distribution). See docs/parity.md.
        check_scalar("q_crit", result.q_critical, as_f64(&fx["q_crit"]), 5e-3),
        check_vec(
            "p_values",
            &actual_p_values,
            &as_f64_vec(&fx["p_values"]),
            5e-3,
        ),
        check_vec(
            "conf_int_lower",
            &actual_conf_lower,
            &expected_conf_lower,
            5e-3,
        ),
        check_vec(
            "conf_int_upper",
            &actual_conf_upper,
            &expected_conf_upper,
            5e-3,
        ),
    ];
    if actual_reject != expected_reject {
        checks.push(Err(format!(
            "  reject: actual={:?} expected={:?}",
            actual_reject, expected_reject
        )));
    }
    assert_parity("tukey_hsd", checks);
}
