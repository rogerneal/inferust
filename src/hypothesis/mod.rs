pub mod anova;
pub mod chisq;
pub mod multicomp;
pub mod nonparametric;
pub mod ttest;
pub mod tukey;
pub mod wald;

pub use multicomp::{adjust, bonferroni, fdr_bh, fdr_by, holm, MultiTestMethod, MultiTestResult};
pub use tukey::{tukey_hsd, TukeyComparison, TukeyHsdResult};
pub use wald::{wald_linear, WaldTestResult};
