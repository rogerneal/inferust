use inferust::discrete::{MultinomialLogit, NegativeBinomial, Probit};

fn main() -> inferust::Result<()> {
    let x = vec![
        vec![0.0],
        vec![1.0],
        vec![2.0],
        vec![3.0],
        vec![4.0],
        vec![5.0],
        vec![6.0],
        vec![7.0],
        vec![8.0],
    ];

    let binary = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let probit = Probit::new()
        .with_feature_names(vec!["dose".into()])
        .fit(&x, &binary)?;
    println!("Probit coefficients: {:?}", probit.coefficients);

    let counts = vec![1.0, 2.0, 2.0, 6.0, 9.0, 14.0, 18.0, 24.0, 31.0];
    let nb = NegativeBinomial::new()
        .with_feature_names(vec!["dose".into()])
        .fit(&x, &counts)?;
    println!("Estimated NB alpha: {:.4}", nb.alpha);

    let classes = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let mnlogit = MultinomialLogit::new().fit(&x, &classes)?;
    println!(
        "Class probabilities for first row: {:?}",
        mnlogit.predict_proba(&x)[0]
    );
    Ok(())
}
