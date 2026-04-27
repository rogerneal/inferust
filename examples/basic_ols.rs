use inferust::{
    correlation,
    descriptive::Summary,
    hypothesis::{anova, chisq, ttest},
    regression::Ols,
};

fn main() {
    // ── 1. Descriptive statistics ────────────────────────────────────────────
    println!("\n╔══════════════════════════════╗");
    println!("║   Descriptive Statistics      ║");
    println!("╚══════════════════════════════╝");

    let data = vec![4.2, 7.8, 5.1, 9.3, 3.6, 8.4, 6.7, 5.9, 7.2, 6.0];
    Summary::new(&data).unwrap().print();

    // ── 2. OLS regression ────────────────────────────────────────────────────
    //      Predict exam score from hours studied + prior GPA
    println!("\n╔══════════════════════════════╗");
    println!("║     OLS Regression            ║");
    println!("╚══════════════════════════════╝");

    let x: Vec<Vec<f64>> = vec![
        vec![2.0, 3.1],
        vec![3.0, 3.4],
        vec![4.0, 2.8],
        vec![5.0, 3.7],
        vec![6.0, 3.2],
        vec![7.0, 3.9],
        vec![8.0, 3.5],
        vec![9.0, 4.0],
        vec![10.0, 2.9],
        vec![11.0, 3.6],
    ];
    let y = vec![55.0, 60.0, 62.0, 70.0, 72.0, 78.0, 80.0, 85.0, 84.0, 90.0];

    let model = Ols::new()
        .with_feature_names(vec!["hours_studied".to_string(), "prior_gpa".to_string()])
        .fit(&x, &y)
        .unwrap();
    model.print_summary();

    println!(
        " Predictions for first 3 obs: {:?}",
        &model.predict(&x[..3])
    );

    // ── 3. Hypothesis tests ──────────────────────────────────────────────────
    println!("\n╔══════════════════════════════╗");
    println!("║     Hypothesis Tests          ║");
    println!("╚══════════════════════════════╝");

    // Paired t-test: effect of an intervention
    let before = vec![72.0, 68.0, 75.0, 80.0, 65.0, 70.0];
    let after = vec![78.0, 74.0, 80.0, 85.0, 72.0, 77.0];
    ttest::paired(&before, &after).unwrap().print();

    // Two-sample Welch t-test: two independent groups
    let group_a = vec![85.0, 90.0, 88.0, 92.0, 87.0];
    let group_b = vec![78.0, 82.0, 75.0, 80.0, 79.0];
    ttest::two_sample(&group_a, &group_b).unwrap().print();

    // One-sample t-test: is the population mean different from 70?
    ttest::one_sample(&y, 70.0).unwrap().print();

    // ── 4. Chi-squared test ──────────────────────────────────────────────────
    println!("\n╔══════════════════════════════╗");
    println!("║     Chi-Squared Test          ║");
    println!("╚══════════════════════════════╝");

    // Goodness-of-fit: are die rolls uniform?
    let die_rolls = vec![18.0, 22.0, 15.0, 25.0, 17.0, 23.0];
    chisq::goodness_of_fit(&die_rolls, None).unwrap().print();

    // Independence: gender × preference contingency table
    let table = vec![vec![30.0, 10.0], vec![15.0, 25.0]];
    chisq::independence(&table).unwrap().print();

    // ── 5. One-way ANOVA ─────────────────────────────────────────────────────
    println!("\n╔══════════════════════════════╗");
    println!("║       One-Way ANOVA           ║");
    println!("╚══════════════════════════════╝");

    let g1 = [23.0, 25.0, 28.0, 24.0, 26.0];
    let g2 = [31.0, 33.0, 30.0, 35.0, 29.0];
    let g3 = [18.0, 21.0, 19.0, 22.0, 20.0];
    anova::one_way(&[&g1, &g2, &g3]).unwrap().print();

    // ── 6. Correlation matrix ────────────────────────────────────────────────
    println!("\n╔══════════════════════════════╗");
    println!("║    Correlation Matrix          ║");
    println!("╚══════════════════════════════╝\n");

    let hours: Vec<f64> = x.iter().map(|r| r[0]).collect();
    let gpa: Vec<f64> = x.iter().map(|r| r[1]).collect();
    let matrix = correlation::correlation_matrix(&[hours.clone(), gpa.clone(), y.clone()]).unwrap();
    correlation::print_correlation_matrix(&matrix, &["hours", "gpa", "score"]);

    let r = correlation::spearman(&hours, &y).unwrap();
    println!("\n Spearman(hours, score) = {:.4}", r);
}
