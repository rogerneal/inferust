use inferust::diagnostics::{breusch_pagan, variance_inflation_factors, white_test};
use inferust::regression::Ols;

fn main() -> inferust::Result<()> {
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
    let fit = Ols::new()
        .with_feature_names(vec!["x1".into(), "x2".into()])
        .fit(&x, &y)?;
    let names = vec!["x1".to_string(), "x2".to_string()];
    let vif = variance_inflation_factors(&x, Some(&names))?;
    let bp = breusch_pagan(&fit.residuals, &x)?;
    let white = white_test(&fit.residuals, &x)?;

    println!("VIF: {:?}", vif);
    println!("Breusch-Pagan LM: {:.4}, p={:.4}", bp.statistic, bp.p_value);
    println!("White LM: {:.4}, p={:.4}", white.statistic, white.p_value);
    Ok(())
}
