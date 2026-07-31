/// Case Studies for inferust v0.1.22
///
/// 1. Clinical trial — survival analysis + hypothesis testing
/// 2. Credit risk — logistic regression + diagnostics
/// 3. Macroeconomics — time series VAR + ARIMA
/// 4. Epidemiology — Poisson GLM + GEE for clustered counts
///
/// Each case uses deterministic synthetic data that mimics realistic structure.
use inferust::gee::{Gee, GeeFamily, WorkingCorrelation};
use inferust::glm::{Logistic, Poisson};
use inferust::hypothesis::{nonparametric, ttest};
use inferust::regression::Ols;
use inferust::survival::{CoxPh, KaplanMeier};
use inferust::time_series::{Arima, Var};

// ── LCG random number generator (deterministic, no dep needed) ────────────────

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / (1u64 << 53) as f64
}
fn norm(s: &mut u64) -> f64 {
    let u1 = lcg(s).max(1e-15);
    let u2 = lcg(s);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ─────────────────────────────────────────────────────────────────────────────
// CASE STUDY 1: Clinical trial — overall survival
//
// Scenario: 400-patient RCT comparing two cancer treatments.
// Covariates: treatment arm (0/1), age (centred), ECOG score (0/1).
// Endpoint: time-to-event with ~25% censoring.
// Questions:
//   a) Does treatment significantly extend survival? (Kaplan-Meier + log-rank)
//   b) Adjusted hazard ratio after controlling for age and ECOG (Cox PH).
//   c) Is the age distribution balanced between arms? (t-test).
// ─────────────────────────────────────────────────────────────────────────────
fn case_clinical_trial() {
    println!("\n=== CASE STUDY 1: Clinical Trial — Overall Survival ===\n");

    let mut s = 42u64;
    let n = 400_usize;

    // Generate patients
    let treatment: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();
    let age: Vec<f64> = (0..n).map(|_| 60.0 + 10.0 * norm(&mut s)).collect();
    let ecog: Vec<f64> = (0..n)
        .map(|_| if lcg(&mut s) < 0.3 { 1.0 } else { 0.0 })
        .collect();

    // True log-hazard: treatment reduces hazard by 0.5, age +0.03/yr, ECOG +0.4
    let times: Vec<f64> = (0..n)
        .map(|i| {
            let log_h = -0.5 * treatment[i] + 0.03 * (age[i] - 60.0) + 0.4 * ecog[i];
            let lam = log_h.exp();
            -lcg(&mut s).max(1e-15).ln() / lam
        })
        .collect();
    let events: Vec<usize> = (0..n)
        .map(|_| if lcg(&mut s) < 0.75 { 1 } else { 0 })
        .collect();

    // a) Kaplan-Meier per arm
    let (t0, e0): (Vec<f64>, Vec<usize>) = times
        .iter()
        .zip(events.iter())
        .zip(treatment.iter())
        .filter(|(_, &trt)| trt == 0.0)
        .map(|((t, e), _)| (*t, *e))
        .unzip();
    let (t1, e1): (Vec<f64>, Vec<usize>) = times
        .iter()
        .zip(events.iter())
        .zip(treatment.iter())
        .filter(|(_, &trt)| trt == 1.0)
        .map(|((t, e), _)| (*t, *e))
        .unzip();

    let km0 = KaplanMeier::new().fit(&t0, &e0).expect("KM control");
    let km1 = KaplanMeier::new().fit(&t1, &e1).expect("KM treatment");

    // Median survival (first t where S(t) <= 0.5)
    let med0 = km0.median_survival.unwrap_or(f64::INFINITY);
    let med1 = km1.median_survival.unwrap_or(f64::INFINITY);

    println!("Kaplan-Meier:");
    println!(
        "  Control   arm  n={}, events={}, median survival = {:.2}",
        t0.len(),
        e0.iter().sum::<usize>(),
        med0
    );
    println!(
        "  Treatment arm  n={}, events={}, median survival = {:.2}",
        t1.len(),
        e1.iter().sum::<usize>(),
        med1
    );

    // b) Cox PH
    let x_cox: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![treatment[i], (age[i] - 60.0) / 10.0, ecog[i]])
        .collect();
    let cox = CoxPh::new().fit(&times, &events, &x_cox).expect("Cox PH");
    println!("\nCox Proportional Hazards (treatment, age_std, ecog):");
    let labels = ["treatment", "age_std", "ecog"];
    for (i, (&coef, (&se, (&z, &p)))) in cox
        .coefficients
        .iter()
        .zip(
            cox.std_errors
                .iter()
                .zip(cox.z_statistics.iter().zip(cox.p_values.iter())),
        )
        .enumerate()
    {
        let hr = coef.exp();
        let sig = if p < 0.001 {
            "***"
        } else if p < 0.01 {
            "**"
        } else if p < 0.05 {
            "*"
        } else {
            ""
        };
        println!(
            "  {:<12}  coef={:+.3}  HR={:.3}  SE={:.3}  z={:+.2}  p={:.4} {}",
            labels[i], coef, hr, se, z, p, sig
        );
    }
    println!("  Log-likelihood: {:.3}", cox.log_likelihood);

    // c) Age balance t-test
    let age0: Vec<f64> = age
        .iter()
        .zip(treatment.iter())
        .filter(|(_, &t)| t == 0.0)
        .map(|(&a, _)| a)
        .collect();
    let age1: Vec<f64> = age
        .iter()
        .zip(treatment.iter())
        .filter(|(_, &t)| t == 1.0)
        .map(|(&a, _)| a)
        .collect();
    let tt = ttest::two_sample(&age0, &age1).expect("t-test");
    println!("\nAge balance (two-sample t-test):");
    println!(
        "  mean control={:.1}  mean treatment={:.1}  t={:.3}  p={:.3}",
        age0.iter().sum::<f64>() / age0.len() as f64,
        age1.iter().sum::<f64>() / age1.len() as f64,
        tt.statistic,
        tt.p_value
    );
    println!(
        "  → Arms are {}balanced (p {}0.05)",
        if tt.p_value > 0.05 { "" } else { "NOT " },
        if tt.p_value > 0.05 { ">" } else { "<" }
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// CASE STUDY 2: Credit Risk — Probability of Default
//
// Scenario: 2000 retail loan applications.
// Features: log_income, debt_to_income, credit_score_std, loan_age.
// Target: default (binary).
// Questions:
//   a) Logistic regression PD model with Wald test on debt-to-income.
//   b) OLS diagnostics on a credit-score linear sub-model.
//   c) Mann-Whitney test: do defaulters have lower credit scores?
// ─────────────────────────────────────────────────────────────────────────────
fn case_credit_risk() {
    println!("\n=== CASE STUDY 2: Credit Risk — Probability of Default ===\n");

    let mut s = 99u64;
    let n = 2000_usize;

    let log_income: Vec<f64> = (0..n).map(|_| 10.5 + 0.5 * norm(&mut s)).collect();
    let dti: Vec<f64> = (0..n)
        .map(|_| (0.35 + 0.15 * norm(&mut s)).clamp(0.05, 0.95))
        .collect();
    let credit_std: Vec<f64> = (0..n).map(|_| norm(&mut s)).collect();
    let loan_age: Vec<f64> = (0..n).map(|_| 12.0 * lcg(&mut s)).collect();

    // True model: higher dti and lower credit → higher PD
    let x: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![log_income[i], dti[i], credit_std[i], loan_age[i]])
        .collect();

    let logit_true: Vec<f64> = (0..n)
        .map(|i| {
            -4.0 + 0.2 * log_income[i] + 2.5 * dti[i] - 1.2 * credit_std[i] - 0.05 * loan_age[i]
        })
        .collect();
    let default: Vec<f64> = logit_true
        .iter()
        .map(|&eta| {
            if lcg(&mut s) < 1.0 / (1.0 + (-eta).exp()) {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    // a) Logistic regression
    let logit = Logistic::new().fit(&x, &default).expect("logistic");
    let default_rate = default.iter().sum::<f64>() / n as f64;
    println!("Default rate: {:.1}%", default_rate * 100.0);
    println!("\nLogistic Regression (intercept, log_income, dti, credit_std, loan_age):");
    let feat_names = ["intercept", "log_income", "dti", "credit_std", "loan_age"];
    for (i, (&coef, (&se, (&z, &p)))) in logit
        .coefficients
        .iter()
        .zip(
            logit
                .std_errors
                .iter()
                .zip(logit.z_statistics.iter().zip(logit.p_values.iter())),
        )
        .enumerate()
    {
        let sig = if p < 0.001 {
            "***"
        } else if p < 0.01 {
            "**"
        } else if p < 0.05 {
            "*"
        } else {
            ""
        };
        println!(
            "  {:<12}  coef={:+.3}  OR={:.3}  SE={:.3}  z={:+.2}  p={:.4} {}",
            feat_names[i],
            coef,
            coef.exp(),
            se,
            z,
            p,
            sig
        );
    }
    println!(
        "  Pseudo-R²={:.4}  AIC={:.1}  BIC={:.1}",
        logit.pseudo_r_squared, logit.aic, logit.bic
    );

    // b) OLS on credit score ~ income + dti
    let x_ols: Vec<Vec<f64>> = (0..n).map(|i| vec![log_income[i], dti[i]]).collect();
    let ols = Ols::new().fit(&x_ols, &credit_std).expect("OLS");
    println!("\nOLS sub-model (credit_score ~ log_income + dti):");
    println!(
        "  R²={:.4}  adj-R²={:.4}  F={:.2}  p(F)={:.4}",
        ols.r_squared, ols.adj_r_squared, ols.f_statistic, ols.f_p_value
    );

    // c) Mann-Whitney: defaulters vs non-defaulters on credit score
    let cs_default: Vec<f64> = credit_std
        .iter()
        .zip(default.iter())
        .filter(|(_, &d)| d == 1.0)
        .map(|(&c, _)| c)
        .collect();
    let cs_ok: Vec<f64> = credit_std
        .iter()
        .zip(default.iter())
        .filter(|(_, &d)| d == 0.0)
        .map(|(&c, _)| c)
        .collect();
    let mw = nonparametric::mann_whitney(&cs_default, &cs_ok).expect("Mann-Whitney");
    println!("\nMann-Whitney (credit score: defaulters vs non-defaulters):");
    println!(
        "  mean defaulters={:.3}  mean non-defaulters={:.3}",
        cs_default.iter().sum::<f64>() / cs_default.len() as f64,
        cs_ok.iter().sum::<f64>() / cs_ok.len() as f64
    );
    println!(
        "  U={:.0}  p={:.4e}  → {}",
        mw.u_statistic,
        mw.p_value,
        if mw.p_value < 0.001 {
            "Highly significant difference ✓"
        } else {
            "Not significant"
        }
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// CASE STUDY 3: Macroeconomics — GDP & Unemployment VAR
//
// Scenario: 200 quarterly observations of GDP growth and unemployment change.
// Questions:
//   a) VAR(2) model — do past unemployment changes Granger-cause GDP growth?
//   b) ARIMA(1,1,1) on GDP growth alone.
// ─────────────────────────────────────────────────────────────────────────────
fn case_macroeconomics() {
    println!("\n=== CASE STUDY 3: Macroeconomics — GDP & Unemployment VAR ===\n");

    let mut s = 777u64;
    let t = 200_usize;

    // Simulate bivariate VAR(2): GDP growth (y1) and unemployment change (y2)
    // y1_t = 0.5*y1_{t-1} - 0.2*y2_{t-1} + 0.1*y1_{t-2} + eps1
    // y2_t = 0.1*y1_{t-1} + 0.4*y2_{t-1} - 0.1*y2_{t-2} + eps2
    let mut y1 = vec![0.0_f64; t];
    let mut y2 = vec![0.0_f64; t];
    for i in 2..t {
        let e1 = 0.5 * norm(&mut s);
        let e2 = 0.3 * norm(&mut s);
        y1[i] = 0.5 * y1[i - 1] - 0.2 * y2[i - 1] + 0.1 * y1[i - 2] + e1;
        y2[i] = 0.1 * y1[i - 1] + 0.4 * y2[i - 1] - 0.1 * y2[i - 2] + e2;
    }

    // a) VAR(2)
    let series: Vec<Vec<f64>> = vec![y1.clone(), y2.clone()];
    let var = Var::new(2).fit(&series).expect("VAR(2)");
    println!("VAR(2) — GDP growth and unemployment change (T={}):", t);
    println!("  Equation 1 (GDP growth):");
    if let Some(coefs) = var.coefficients.first() {
        for (j, c) in coefs.iter().enumerate() {
            println!("    lag-coef[{}] = {:.4}", j, c);
        }
    }
    println!("  AIC={:.2}  BIC={:.2}", var.aic, var.bic);

    // b) ARIMA(1,1,1) on GDP growth
    let arima = Arima::new(1, 1, 1).fit(&y1).expect("ARIMA");
    println!("\nARIMA(1,1,1) on GDP growth series:");
    println!(
        "  AR coefs: {:?}",
        arima
            .ar_coefficients
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "  MA coefs: {:?}",
        arima
            .ma_coefficients
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "  Sigma²={:.5}  Log-lik={:.3}",
        arima.sigma2, arima.log_likelihood
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// CASE STUDY 4: Epidemiology — Disease Incidence (Poisson / GEE)
//
// Scenario: 50 counties × 20 years = 1000 observations.
//   Each county is a cluster. Covariates: log_population, pollution_index,
//   vaccination_rate. Outcome: disease count.
// Questions:
//   a) Poisson GLM (naive, ignores clustering).
//   b) GEE with exchangeable working correlation (accounts for county clusters).
//   c) Compare SE inflation — GEE vs GLM.
// ─────────────────────────────────────────────────────────────────────────────
fn case_epidemiology() {
    println!("\n=== CASE STUDY 4: Epidemiology — Disease Incidence ===\n");

    let mut s = 314u64;
    let n_counties = 50_usize;
    let n_years = 20_usize;
    let n = n_counties * n_years;

    // County-level random intercepts
    let county_re: Vec<f64> = (0..n_counties).map(|_| 0.3 * norm(&mut s)).collect();

    let mut x: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    let mut clusters: Vec<usize> = Vec::with_capacity(n);

    for (county, &county_effect) in county_re.iter().enumerate() {
        for _year in 0..n_years {
            let log_pop = 10.0 + 0.5 * norm(&mut s);
            let pollution = (0.5 + 0.2 * norm(&mut s)).clamp(0.0, 1.0);
            let vacc_rate = (0.7 + 0.1 * norm(&mut s)).clamp(0.0, 1.0);

            // Use offset so expected counts stay in 1–30 range
            let log_mu = -8.0 + 0.8 * log_pop + 1.2 * pollution - 2.0 * vacc_rate + county_effect;
            let mu = log_mu.exp();

            // Poisson draw
            let mut k = 0u64;
            let mut p = (-mu).exp();
            let mut cdf = p;
            let u = lcg(&mut s);
            while cdf < u && k < 5_000 {
                k += 1;
                p *= mu / k as f64;
                cdf += p;
            }

            x.push(vec![log_pop, pollution, vacc_rate]);
            y.push(k as f64);
            clusters.push(county);
        }
    }

    // a) Naive Poisson GLM
    let poi = Poisson::new().fit(&x, &y).expect("Poisson GLM");
    println!("Poisson GLM (intercept, log_pop, pollution, vacc_rate):");
    let names = ["intercept", "log_pop", "pollution", "vacc_rate"];
    for (i, (&coef, (&se, &p))) in poi
        .coefficients
        .iter()
        .zip(poi.std_errors.iter().zip(poi.p_values.iter()))
        .enumerate()
    {
        println!(
            "  {:<12}  coef={:+.3}  RR={:.3}  SE(GLM)={:.4}  p={:.4}",
            names[i],
            coef,
            coef.exp(),
            se,
            p
        );
    }
    println!("  LLf={:.2}  AIC={:.1}", poi.log_likelihood, poi.aic);

    // b) GEE with exchangeable correlation
    let gee = Gee::new(GeeFamily::Poisson)
        .with_working_correlation(WorkingCorrelation::Exchangeable)
        .fit(&x, &y, &clusters)
        .expect("GEE");
    println!("\nGEE Poisson — Exchangeable working correlation (50 county clusters):");
    println!("  Estimated within-cluster correlation rho={:.4}", gee.rho);
    for (i, (&coef, &se)) in gee
        .coefficients
        .iter()
        .zip(gee.robust_std_errors.iter())
        .enumerate()
    {
        println!("  {:<12}  coef={:+.3}  SE(GEE)={:.4}", names[i], coef, se);
    }

    // c) SE inflation ratio
    println!("\nSE inflation (GEE robust / GLM naive):");
    for (i, (&se_gee, &se_glm)) in gee
        .robust_std_errors
        .iter()
        .zip(poi.std_errors.iter())
        .enumerate()
    {
        println!(
            "  {:<12}  ratio={:.3}{}",
            names[i],
            se_gee / se_glm,
            if se_gee / se_glm > 1.3 {
                "  ← clustering inflates SE"
            } else {
                ""
            }
        );
    }
    println!("\n  → GEE accounts for within-county correlation; naive Poisson SEs are anti-conservative.");
}

fn main() {
    case_clinical_trial();
    case_credit_risk();
    case_macroeconomics();
    case_epidemiology();
    println!("\n=== All case studies complete ===");
}
