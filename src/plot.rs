//! Plotting utilities — SVG file output and ASCII terminal charts.
//!
//! Every plot is built with a fluent builder, then either:
//! - saved to an SVG file with [`Plot::save`], or
//! - written as a string with [`Plot::to_svg`], or
//! - printed to the terminal with [`Plot::print_ascii`].
//!
//! # Quick start
//! ```rust
//! use inferust::plot::Plot;
//!
//! let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
//! let y: Vec<f64> = x.iter().map(|xi| xi * 0.5 + (xi * 0.8).sin()).collect();
//!
//! Plot::new()
//!     .title("My series")
//!     .xlabel("time")
//!     .ylabel("value")
//!     .line(&x, &y, "y")
//!     .print_ascii();
//! ```
//!
//! # Convenience constructors
//! - [`Plot::acf`] — ACF / PACF bar chart.
//! - [`Plot::survival`] — Kaplan-Meier step curve with confidence bands.
//! - [`Plot::residuals`] — Residuals-vs-fitted scatter.

use std::fmt::Write as FmtWrite;
use crate::error::{InferustError, Result};

// ── Palette ───────────────────────────────────────────────────────────────────

const COLORS: &[&str] = &[
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
];

fn color(idx: usize) -> &'static str { COLORS[idx % COLORS.len()] }

// ── Series types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Series {
    Line { x: Vec<f64>, y: Vec<f64>, label: String, color: String },
    Scatter { x: Vec<f64>, y: Vec<f64>, label: String, color: String },
    Bar { x: Vec<f64>, heights: Vec<f64>, label: String, color: String },
    Band { x: Vec<f64>, lo: Vec<f64>, hi: Vec<f64>, color: String },
    Step { x: Vec<f64>, y: Vec<f64>, label: String, color: String },
    HLine { y: f64, color: String, dash: bool },
}

// ── Plot builder ──────────────────────────────────────────────────────────────

/// A plot builder that can render to SVG or ASCII.
///
/// Add data with `.line()`, `.scatter()`, `.bar()`, `.step()`, or `.band()`,
/// then call `.to_svg()`, `.save()`, or `.print_ascii()`.
#[derive(Debug, Clone, Default)]
pub struct Plot {
    title: String,
    xlabel: String,
    ylabel: String,
    series: Vec<Series>,
    width: f64,
    height: f64,
}

impl Plot {
    /// Create an empty plot with default 600 × 350 SVG dimensions.
    pub fn new() -> Self {
        Self { width: 600.0, height: 350.0, ..Default::default() }
    }

    /// Set the plot title.
    pub fn title(mut self, t: impl Into<String>) -> Self { self.title = t.into(); self }
    /// Set the x-axis label.
    pub fn xlabel(mut self, l: impl Into<String>) -> Self { self.xlabel = l.into(); self }
    /// Set the y-axis label.
    pub fn ylabel(mut self, l: impl Into<String>) -> Self { self.ylabel = l.into(); self }
    /// Override SVG canvas width (default 600).
    pub fn width(mut self, w: f64) -> Self { self.width = w; self }
    /// Override SVG canvas height (default 350).
    pub fn height(mut self, h: f64) -> Self { self.height = h; self }

    /// Add a line series.
    pub fn line(mut self, x: &[f64], y: &[f64], label: impl Into<String>) -> Self {
        let idx = self.series.len();
        self.series.push(Series::Line {
            x: x.to_vec(), y: y.to_vec(),
            label: label.into(), color: color(idx).into(),
        });
        self
    }

    /// Add a scatter series.
    pub fn scatter(mut self, x: &[f64], y: &[f64], label: impl Into<String>) -> Self {
        let idx = self.series.len();
        self.series.push(Series::Scatter {
            x: x.to_vec(), y: y.to_vec(),
            label: label.into(), color: color(idx).into(),
        });
        self
    }

    /// Add a bar chart series.  `x` are the bar centres, `heights` are the values.
    pub fn bar(mut self, x: &[f64], heights: &[f64], label: impl Into<String>) -> Self {
        let idx = self.series.len();
        self.series.push(Series::Bar {
            x: x.to_vec(), heights: heights.to_vec(),
            label: label.into(), color: color(idx).into(),
        });
        self
    }

    /// Add a step-function series (e.g. a survival curve).
    pub fn step(mut self, x: &[f64], y: &[f64], label: impl Into<String>) -> Self {
        let idx = self.series.len();
        self.series.push(Series::Step {
            x: x.to_vec(), y: y.to_vec(),
            label: label.into(), color: color(idx).into(),
        });
        self
    }

    /// Add a shaded confidence band between `lo` and `hi`.
    pub fn band(mut self, x: &[f64], lo: &[f64], hi: &[f64]) -> Self {
        let idx = self.series.len();
        self.series.push(Series::Band {
            x: x.to_vec(), lo: lo.to_vec(), hi: hi.to_vec(),
            color: color(idx).into(),
        });
        self
    }

    /// Add a horizontal reference line.
    pub fn hline(mut self, y: f64, dashed: bool) -> Self {
        self.series.push(Series::HLine { y, color: "#aaaaaa".into(), dash: dashed });
        self
    }

    // ── Convenience constructors ──────────────────────────────────────────────

    /// Build an ACF or PACF bar chart.
    ///
    /// * `lags` — lag values (typically 0, 1, 2, …).
    /// * `values` — autocorrelation or partial autocorrelation at each lag.
    /// * `conf_bound` — ±1.96/√n significance threshold (drawn as dashed lines).
    pub fn acf(lags: &[usize], values: &[f64], conf_bound: f64) -> Self {
        let x: Vec<f64> = lags.iter().map(|&l| l as f64).collect();
        Plot::new()
            .title("ACF")
            .xlabel("lag")
            .ylabel("autocorrelation")
            .bar(&x, values, "acf")
            .hline(conf_bound, true)
            .hline(-conf_bound, true)
            .hline(0.0, false)
    }

    /// Build a Kaplan-Meier step curve with Greenwood confidence band.
    ///
    /// Accepts the `curve` slice from a [`crate::survival::KaplanMeierResult`].
    pub fn survival(curve: &[crate::survival::KmStep]) -> Self {
        let times: Vec<f64> = curve.iter().map(|s| s.time).collect();
        let surv: Vec<f64> = curve.iter().map(|s| s.survival).collect();
        let lo: Vec<f64> = curve.iter().map(|s| s.ci_lower).collect();
        let hi: Vec<f64> = curve.iter().map(|s| s.ci_upper).collect();
        Plot::new()
            .title("Kaplan-Meier Survival Curve")
            .xlabel("time")
            .ylabel("S(t)")
            .band(&times, &lo, &hi)
            .step(&times, &surv, "survival")
    }

    /// Build a residuals-vs-fitted scatter plot.
    pub fn residuals(fitted: &[f64], residuals: &[f64]) -> Self {
        Plot::new()
            .title("Residuals vs Fitted")
            .xlabel("fitted values")
            .ylabel("residuals")
            .scatter(fitted, residuals, "residuals")
            .hline(0.0, true)
    }

    // ── Render ────────────────────────────────────────────────────────────────

    /// Return the plot as an SVG string.
    pub fn to_svg(&self) -> String {
        let w = self.width;
        let h = self.height;
        // Margins: top, right, bottom, left
        let (mt, mr, mb, ml) = (
            if self.title.is_empty() { 15.0 } else { 35.0 },
            20.0,
            if self.xlabel.is_empty() { 40.0 } else { 55.0 },
            if self.ylabel.is_empty() { 45.0 } else { 60.0 },
        );
        let pw = w - ml - mr; // plot width
        let ph = h - mt - mb; // plot height

        // Compute data bounds
        let (xmin, xmax, ymin, ymax) = self.data_bounds();
        let xrange = (xmax - xmin).max(f64::EPSILON);
        let yrange = (ymax - ymin).max(f64::EPSILON);

        // Scale helpers (data → SVG coordinates)
        let sx = |xv: f64| ml + (xv - xmin) / xrange * pw;
        let sy = |yv: f64| mt + ph - (yv - ymin) / yrange * ph;

        let mut svg = String::new();
        let _ = write!(svg,
            r#"<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">"#
        );

        // Background
        let _ = write!(svg,
            r#"<rect width="{w}" height="{h}" fill="white"/>"#
        );

        // Grid lines
        let n_yticks = 5usize;
        for i in 0..=n_yticks {
            let yv = ymin + yrange * i as f64 / n_yticks as f64;
            let yp = sy(yv);
            let _ = write!(svg,
                r#"<line x1="{ml:.1}" y1="{yp:.1}" x2="{:.1}" y2="{yp:.1}" stroke="#e0e0e0" stroke-width="0.5"/>"#,
                ml + pw
            );
            let _ = write!(svg,
                r#"<text x="{:.1}" y="{:.1}" text-anchor="end" font-size="11" fill="#666">{:.2}</text>"#,
                ml - 4.0, yp + 4.0, yv
            );
        }
        let n_xticks = 6usize;
        for i in 0..=n_xticks {
            let xv = xmin + xrange * i as f64 / n_xticks as f64;
            let xp = sx(xv);
            let _ = write!(svg,
                r#"<line x1="{xp:.1}" y1="{mt:.1}" x2="{xp:.1}" y2="{:.1}" stroke="#e0e0e0" stroke-width="0.5"/>"#,
                mt + ph
            );
            let _ = write!(svg,
                r#"<text x="{xp:.1}" y="{:.1}" text-anchor="middle" font-size="11" fill="#666">{:.2}</text>"#,
                mt + ph + 14.0, xv
            );
        }

        // Plot border
        let _ = write!(svg,
            r#"<rect x="{ml:.1}" y="{mt:.1}" width="{pw:.1}" height="{ph:.1}" fill="none" stroke="#cccccc" stroke-width="0.8"/>"#
        );

        // Clipping region
        let _ = write!(svg,
            r#"<clipPath id="plot-area"><rect x="{ml:.1}" y="{mt:.1}" width="{pw:.1}" height="{ph:.1}"/></clipPath>"#
        );

        // Draw series
        for s in &self.series {
            self.render_series(&mut svg, s, &sx, &sy, mt, ph, ml, pw);
        }

        // Axes labels
        if !self.xlabel.is_empty() {
            let _ = write!(svg,
                r#"<text x="{:.1}" y="{:.1}" text-anchor="middle" font-size="12" fill="#444">{}</text>"#,
                ml + pw / 2.0, h - 5.0, escape_xml(&self.xlabel)
            );
        }
        if !self.ylabel.is_empty() {
            let _ = write!(svg,
                r#"<text transform="rotate(-90)" x="{:.1}" y="{:.1}" text-anchor="middle" font-size="12" fill="#444">{}</text>"#,
                -(mt + ph / 2.0), 13.0, escape_xml(&self.ylabel)
            );
        }
        if !self.title.is_empty() {
            let _ = write!(svg,
                r#"<text x="{:.1}" y="{:.1}" text-anchor="middle" font-size="14" font-weight="500" fill="#222">{}</text>"#,
                w / 2.0, mt - 8.0, escape_xml(&self.title)
            );
        }

        // Legend
        let legend_series: Vec<(&str, &str)> = self.series.iter().filter_map(|s| match s {
            Series::Line { label, color, .. } => Some((label.as_str(), color.as_str())),
            Series::Scatter { label, color, .. } => Some((label.as_str(), color.as_str())),
            Series::Bar { label, color, .. } => Some((label.as_str(), color.as_str())),
            Series::Step { label, color, .. } => Some((label.as_str(), color.as_str())),
            _ => None,
        }).collect();
        if !legend_series.is_empty() {
            let lx = ml + pw - 10.0;
            let mut ly = mt + 10.0;
            for (label, col) in &legend_series {
                let _ = write!(svg,
                    r#"<rect x="{:.1}" y="{:.1}" width="12" height="3" fill="{col}" rx="1"/>"#,
                    lx - 18.0, ly + 4.0
                );
                let _ = write!(svg,
                    r#"<text x="{:.1}" y="{:.1}" text-anchor="end" font-size="11" fill="#444">{}</text>"#,
                    lx - 22.0, ly + 10.0, escape_xml(label)
                );
                ly += 16.0;
            }
        }

        svg.push_str("</svg>");
        svg
    }

    /// Save the plot to an SVG file at `path`.
    pub fn save(&self, path: &str) -> Result<()> {
        std::fs::write(path, self.to_svg())
            .map_err(|e| InferustError::InvalidInput(format!("failed to write {path}: {e}")))
    }

    /// Print an ASCII representation of the plot to stdout.
    ///
    /// Uses a 70-column × 20-row grid.
    pub fn print_ascii(&self) {
        const COLS: usize = 70;
        const ROWS: usize = 20;

        let (xmin, xmax, ymin, ymax) = self.data_bounds();
        let xrange = (xmax - xmin).max(f64::EPSILON);
        let yrange = (ymax - ymin).max(f64::EPSILON);

        let mut grid = vec![vec![' '; COLS]; ROWS];

        let col_of = |x: f64| ((x - xmin) / xrange * (COLS - 1) as f64).round().clamp(0.0, (COLS - 1) as f64) as usize;
        let row_of = |y: f64| (ROWS - 1) - ((y - ymin) / yrange * (ROWS - 1) as f64).round().clamp(0.0, (ROWS - 1) as f64) as usize;

        // Horizontal zero line
        if ymin <= 0.0 && ymax >= 0.0 {
            let r = row_of(0.0);
            for c in 0..COLS { grid[r][c] = '-'; }
        }

        for s in &self.series {
            match s {
                Series::Line { x, y, .. } | Series::Step { x, y, .. } => {
                    for (xv, yv) in x.iter().zip(y.iter()) {
                        let r = row_of(*yv);
                        let c = col_of(*xv);
                        grid[r][c] = '*';
                    }
                }
                Series::Scatter { x, y, .. } => {
                    for (xv, yv) in x.iter().zip(y.iter()) {
                        let r = row_of(*yv);
                        let c = col_of(*xv);
                        grid[r][c] = 'o';
                    }
                }
                Series::Bar { x, heights, .. } => {
                    let baseline = row_of(0.0_f64.clamp(ymin, ymax));
                    for (xv, hv) in x.iter().zip(heights.iter()) {
                        let top = row_of(*hv);
                        let c = col_of(*xv);
                        let (r0, r1) = if top <= baseline { (top, baseline) } else { (baseline, top) };
                        for r in r0..=r1.min(ROWS - 1) { grid[r][c] = '|'; }
                        if top < ROWS { grid[top][c] = '#'; }
                    }
                }
                Series::HLine { y, .. } => {
                    if *y >= ymin && *y <= ymax {
                        let r = row_of(*y);
                        for c in 0..COLS { if grid[r][c] == ' ' { grid[r][c] = '.'; } }
                    }
                }
                Series::Band { .. } => {} // skip in ASCII
            }
        }

        // Print
        if !self.title.is_empty() { println!("{:^width$}", self.title, width = COLS + 8); }
        for (i, row) in grid.iter().enumerate() {
            // y-axis tick every ~5 rows
            let yv = ymax - (i as f64 / (ROWS - 1) as f64) * yrange;
            if i % 5 == 0 {
                print!("{:>7.2} |", yv);
            } else {
                print!("        |");
            }
            let line: String = row.iter().collect();
            println!("{line}");
        }
        // x-axis
        println!("        +{}", "-".repeat(COLS));
        // x-axis ticks
        let ticks = 7usize;
        print!("         ");
        for i in 0..=ticks {
            let xv = xmin + xrange * i as f64 / ticks as f64;
            let pos = (i * (COLS / ticks)).min(COLS - 1);
            let s = format!("{xv:.1}");
            print!("{:<width$}", s, width = if i < ticks { COLS / ticks } else { s.len() });
        }
        println!();
        if !self.xlabel.is_empty() {
            println!("{:^width$}", self.xlabel, width = COLS + 9);
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn data_bounds(&self) -> (f64, f64, f64, f64) {
        let mut xmin = f64::INFINITY;
        let mut xmax = f64::NEG_INFINITY;
        let mut ymin = f64::INFINITY;
        let mut ymax = f64::NEG_INFINITY;

        for s in &self.series {
            let (xs, ys): (Option<&[f64]>, Option<&[f64]>) = match s {
                Series::Line { x, y, .. } => (Some(x), Some(y)),
                Series::Scatter { x, y, .. } => (Some(x), Some(y)),
                Series::Bar { x, heights, .. } => (Some(x), Some(heights)),
                Series::Step { x, y, .. } => (Some(x), Some(y)),
                Series::Band { x, lo, hi, .. } => {
                    for v in x.iter() { xmin = xmin.min(*v); xmax = xmax.max(*v); }
                    for v in lo.iter() { ymin = ymin.min(*v); }
                    for v in hi.iter() { ymax = ymax.max(*v); }
                    (None, None)
                }
                Series::HLine { y, .. } => {
                    ymin = ymin.min(*y); ymax = ymax.max(*y);
                    (None, None)
                }
            };
            if let Some(xs) = xs {
                for v in xs { xmin = xmin.min(*v); xmax = xmax.max(*v); }
            }
            if let Some(ys) = ys {
                for v in ys { ymin = ymin.min(*v); ymax = ymax.max(*v); }
            }
        }
        // Fallback for empty plots
        if !xmin.is_finite() { xmin = 0.0; xmax = 1.0; }
        if !ymin.is_finite() { ymin = 0.0; ymax = 1.0; }
        if (xmax - xmin).abs() < f64::EPSILON { xmax = xmin + 1.0; }
        if (ymax - ymin).abs() < f64::EPSILON { ymin -= 0.5; ymax += 0.5; }
        // Pad y by 5%
        let ypad = (ymax - ymin) * 0.05;
        (xmin, xmax, ymin - ypad, ymax + ypad)
    }

    fn render_series(
        &self, svg: &mut String, s: &Series,
        sx: &impl Fn(f64) -> f64, sy: &impl Fn(f64) -> f64,
        mt: f64, ph: f64, ml: f64, pw: f64,
    ) {
        match s {
            Series::Line { x, y, color, .. } => {
                if x.is_empty() { return; }
                let _ = write!(svg, r#"<polyline clip-path="url(#plot-area)" fill="none" stroke="{color}" stroke-width="1.8" points=""#);
                for (xv, yv) in x.iter().zip(y.iter()) {
                    let _ = write!(svg, "{:.1},{:.1} ", sx(*xv), sy(*yv));
                }
                svg.push_str(r#""/>"#);
            }
            Series::Step { x, y, color, .. } => {
                if x.is_empty() { return; }
                let _ = write!(svg, r#"<path clip-path="url(#plot-area)" fill="none" stroke="{color}" stroke-width="1.8" d=""#);
                let _ = write!(svg, "M {:.1},{:.1} ", sx(x[0]), sy(y[0]));
                for i in 1..x.len() {
                    // Step: go horizontal first, then vertical
                    let _ = write!(svg, "H {:.1} V {:.1} ", sx(x[i]), sy(y[i]));
                }
                svg.push_str(r#""/>"#);
            }
            Series::Scatter { x, y, color, .. } => {
                for (xv, yv) in x.iter().zip(y.iter()) {
                    let _ = write!(svg,
                        r#"<circle clip-path="url(#plot-area)" cx="{:.1}" cy="{:.1}" r="3" fill="{color}" fill-opacity="0.7"/>"#,
                        sx(*xv), sy(*yv)
                    );
                }
            }
            Series::Bar { x, heights, color, .. } => {
                let n = x.len();
                let bar_w = if n > 1 {
                    (pw / n as f64 * 0.7).max(2.0)
                } else { 20.0 };
                let (_, _, ymin, _) = self.data_bounds();
                let baseline_y = sy(0.0_f64.max(ymin));
                for (xv, hv) in x.iter().zip(heights.iter()) {
                    let cx = sx(*xv);
                    let top_y = sy(*hv);
                    let bh = (baseline_y - top_y).abs().max(1.0);
                    let rect_y = top_y.min(baseline_y);
                    let _ = write!(svg,
                        r#"<rect clip-path="url(#plot-area)" x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{color}" fill-opacity="0.8"/>"#,
                        cx - bar_w / 2.0, rect_y, bar_w, bh
                    );
                }
            }
            Series::Band { x, lo, hi, color, .. } => {
                if x.is_empty() { return; }
                // Upper path + lower path reversed = filled polygon
                let _ = write!(svg, r#"<polygon clip-path="url(#plot-area)" fill="{color}" fill-opacity="0.15" points=""#);
                for (xv, hv) in x.iter().zip(hi.iter()) {
                    let _ = write!(svg, "{:.1},{:.1} ", sx(*xv), sy(*hv));
                }
                for (xv, lv) in x.iter().rev().zip(lo.iter().rev()) {
                    let _ = write!(svg, "{:.1},{:.1} ", sx(*xv), sy(*lv));
                }
                svg.push_str(r#""/>"#);
            }
            Series::HLine { y, color, dash } => {
                let (xmin, xmax, _, _) = self.data_bounds();
                let yp = sy(*y);
                let dash_attr = if *dash { r#" stroke-dasharray="4,3""# } else { "" };
                let _ = write!(svg,
                    r#"<line clip-path="url(#plot-area)" x1="{:.1}" y1="{yp:.1}" x2="{:.1}" y2="{yp:.1}" stroke="{color}" stroke-width="1"{dash_attr}/>"#,
                    sx(xmin), sx(xmax)
                );
            }
        }
    }
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::Plot;

    #[test]
    fn svg_contains_polyline_for_line_series() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 0.5, 2.0];
        let svg = Plot::new().line(&x, &y, "test").to_svg();
        assert!(svg.contains("polyline"), "expected polyline in SVG: {}", &svg[..200]);
    }

    #[test]
    fn svg_contains_title() {
        let svg = Plot::new().title("My Plot").line(&[1.0], &[1.0], "s").to_svg();
        assert!(svg.contains("My Plot"), "title missing from SVG");
    }

    #[test]
    fn svg_contains_circle_for_scatter() {
        let svg = Plot::new().scatter(&[1.0, 2.0], &[3.0, 4.0], "pts").to_svg();
        assert!(svg.contains("circle"), "expected circles in SVG");
    }

    #[test]
    fn svg_contains_rect_for_bar() {
        let svg = Plot::new().bar(&[1.0, 2.0, 3.0], &[0.5, -0.3, 0.8], "acf").to_svg();
        assert!(svg.contains("<rect"), "expected rects in bar chart SVG");
    }

    #[test]
    fn acf_convenience_constructor() {
        let lags: Vec<usize> = (0..8).collect();
        let vals = vec![1.0, 0.7, 0.5, 0.3, 0.1, 0.0, -0.1, -0.2];
        let svg = Plot::acf(&lags, &vals, 0.31).to_svg();
        assert!(svg.contains("ACF"));
    }

    #[test]
    fn save_writes_file() {
        let path = "/tmp/inferust_test_plot.svg";
        Plot::new().line(&[0.0, 1.0], &[0.0, 1.0], "l").save(path).unwrap();
        assert!(std::path::Path::new(path).exists());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn print_ascii_does_not_panic() {
        let x: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|xi| xi.sin()).collect();
        // Just check it doesn't panic
        let p = Plot::new().line(&x, &y, "sin");
        let _ = std::panic::catch_unwind(|| p.print_ascii());
    }
}
