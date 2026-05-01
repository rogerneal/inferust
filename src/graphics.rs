//! Lightweight plotting helpers that render statistical graphics as SVG.
//!
//! The functions in this module deliberately return `String` values instead of
//! writing files or depending on a plotting backend. This keeps `inferust`
//! dependency-light while still giving regression and time-series workflows a
//! portable graphics surface.
//!
//! # Example
//! ```rust
//! use inferust::graphics::{line_plot_svg, PlotOptions};
//!
//! let y = vec![1.0, 1.8, 1.4, 2.2, 2.7];
//! let svg = line_plot_svg(&y, PlotOptions::default()).unwrap();
//! assert!(svg.starts_with("<svg"));
//! ```

use crate::error::{InferustError, Result};

/// Options shared by SVG plot helpers.
#[derive(Debug, Clone)]
pub struct PlotOptions {
    /// SVG width in pixels.
    pub width: usize,
    /// SVG height in pixels.
    pub height: usize,
    /// Plot title.
    pub title: String,
    /// X-axis label.
    pub x_label: String,
    /// Y-axis label.
    pub y_label: String,
    /// Stroke/fill color for primary marks.
    pub color: String,
    /// Background color.
    pub background: String,
}

impl Default for PlotOptions {
    fn default() -> Self {
        Self {
            width: 720,
            height: 420,
            title: String::new(),
            x_label: String::new(),
            y_label: String::new(),
            color: "#2563eb".to_string(),
            background: "#ffffff".to_string(),
        }
    }
}

/// Render a line plot for values observed at integer positions `0..n`.
pub fn line_plot_svg(y: &[f64], options: PlotOptions) -> Result<String> {
    if y.len() < 2 {
        return Err(InferustError::InsufficientData {
            needed: 2,
            got: y.len(),
        });
    }
    ensure_finite(y, "line plot values")?;
    let x: Vec<f64> = (0..y.len()).map(|i| i as f64).collect();
    render_xy_svg(&x, y, &options, PlotKind::Line)
}

/// Render an x/y scatter plot.
pub fn scatter_plot_svg(x: &[f64], y: &[f64], options: PlotOptions) -> Result<String> {
    validate_xy(x, y)?;
    render_xy_svg(x, y, &options, PlotKind::Scatter)
}

/// Render residuals against fitted values with a horizontal zero line.
pub fn residual_plot_svg(
    fitted: &[f64],
    residuals: &[f64],
    options: PlotOptions,
) -> Result<String> {
    validate_xy(fitted, residuals)?;
    render_xy_svg(fitted, residuals, &options, PlotKind::Residual)
}

/// Render an autocorrelation bar chart for ACF values.
pub fn acf_plot_svg(acf: &[f64], options: PlotOptions) -> Result<String> {
    if acf.is_empty() {
        return Err(InferustError::InsufficientData { needed: 1, got: 0 });
    }
    ensure_finite(acf, "ACF values")?;
    let width = options.width.max(240);
    let height = options.height.max(180);
    let margin = Margins::default();
    let plot_w = (width - margin.left - margin.right) as f64;
    let plot_h = (height - margin.top - margin.bottom) as f64;
    let max_abs = acf.iter().map(|v| v.abs()).fold(1.0_f64, f64::max);
    let y_min = -max_abs;
    let y_max = max_abs;
    let x_max = (acf.len() - 1).max(1) as f64;
    let zero_y = scale_y(0.0, y_min, y_max, height, &margin);

    let mut svg = svg_header(width, height, &options);
    push_axes(&mut svg, width, height, &margin, &options);
    svg.push_str(&format!(
        r##"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="#64748b" stroke-width="1"/>"##,
        margin.left as f64,
        zero_y,
        (width - margin.right) as f64,
        zero_y
    ));

    for (lag, &value) in acf.iter().enumerate() {
        let x = margin.left as f64 + (lag as f64 / x_max) * plot_w;
        let y = scale_y(value, y_min, y_max, height, &margin);
        svg.push_str(&format!(
            r#"<line x1="{x:.2}" y1="{zero_y:.2}" x2="{x:.2}" y2="{y:.2}" stroke="{}" stroke-width="3" stroke-linecap="round"/>"#,
            escape_attr(&options.color)
        ));
    }
    finish_svg(&mut svg, &options, width, height, &margin, plot_w, plot_h);
    Ok(svg)
}

#[derive(Debug, Clone, Copy)]
enum PlotKind {
    Line,
    Scatter,
    Residual,
}

#[derive(Debug, Clone, Copy)]
struct Margins {
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
}

impl Default for Margins {
    fn default() -> Self {
        Self {
            left: 64,
            right: 28,
            top: 48,
            bottom: 56,
        }
    }
}

fn render_xy_svg(x: &[f64], y: &[f64], options: &PlotOptions, kind: PlotKind) -> Result<String> {
    validate_xy(x, y)?;
    let width = options.width.max(240);
    let height = options.height.max(180);
    let margin = Margins::default();
    let plot_w = (width - margin.left - margin.right) as f64;
    let plot_h = (height - margin.top - margin.bottom) as f64;
    let (x_min, x_max) = extent(x);
    let (mut y_min, mut y_max) = extent(y);
    if matches!(kind, PlotKind::Residual) {
        y_min = y_min.min(0.0);
        y_max = y_max.max(0.0);
    }
    let x_span = expand_if_flat(x_min, x_max);
    let y_span = expand_if_flat(y_min, y_max);

    let points: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| {
            (
                margin.left as f64 + ((xi - x_span.0) / (x_span.1 - x_span.0)) * plot_w,
                scale_y(yi, y_span.0, y_span.1, height, &margin),
            )
        })
        .collect();

    let mut svg = svg_header(width, height, options);
    push_axes(&mut svg, width, height, &margin, options);
    if matches!(kind, PlotKind::Residual) {
        let zero_y = scale_y(0.0, y_span.0, y_span.1, height, &margin);
        svg.push_str(&format!(
            r##"<line x1="{:.2}" y1="{zero_y:.2}" x2="{:.2}" y2="{zero_y:.2}" stroke="#64748b" stroke-width="1" stroke-dasharray="4 4"/>"##,
            margin.left as f64,
            (width - margin.right) as f64
        ));
    }

    match kind {
        PlotKind::Line => {
            let path = points
                .iter()
                .enumerate()
                .map(|(i, (px, py))| {
                    if i == 0 {
                        format!("M {px:.2} {py:.2}")
                    } else {
                        format!("L {px:.2} {py:.2}")
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            svg.push_str(&format!(
                r#"<path d="{path}" fill="none" stroke="{}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>"#,
                escape_attr(&options.color)
            ));
        }
        PlotKind::Scatter | PlotKind::Residual => {
            for (px, py) in points {
                svg.push_str(&format!(
                    r#"<circle cx="{px:.2}" cy="{py:.2}" r="3.5" fill="{}" opacity="0.85"/>"#,
                    escape_attr(&options.color)
                ));
            }
        }
    }

    finish_svg(&mut svg, options, width, height, &margin, plot_w, plot_h);
    Ok(svg)
}

fn svg_header(width: usize, height: usize, options: &PlotOptions) -> String {
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{}"><rect width="100%" height="100%" fill="{}"/>"#,
        escape_attr(&options.title),
        escape_attr(&options.background)
    )
}

fn push_axes(
    svg: &mut String,
    width: usize,
    height: usize,
    margin: &Margins,
    options: &PlotOptions,
) {
    let left = margin.left;
    let right = width - margin.right;
    let top = margin.top;
    let bottom = height - margin.bottom;
    svg.push_str(&format!(
        r##"<g stroke="#334155" stroke-width="1.2"><line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/><line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/></g>"##
    ));
    if !options.title.is_empty() {
        svg.push_str(&format!(
            r##"<text x="{}" y="26" text-anchor="middle" font-family="system-ui, sans-serif" font-size="18" font-weight="600" fill="#0f172a">{}</text>"##,
            width / 2,
            escape_text(&options.title)
        ));
    }
    if !options.x_label.is_empty() {
        svg.push_str(&format!(
            r##"<text x="{}" y="{}" text-anchor="middle" font-family="system-ui, sans-serif" font-size="13" fill="#334155">{}</text>"##,
            width / 2,
            height - 16,
            escape_text(&options.x_label)
        ));
    }
    if !options.y_label.is_empty() {
        svg.push_str(&format!(
            r##"<text transform="translate(18 {}) rotate(-90)" text-anchor="middle" font-family="system-ui, sans-serif" font-size="13" fill="#334155">{}</text>"##,
            height / 2,
            escape_text(&options.y_label)
        ));
    }
}

fn finish_svg(
    svg: &mut String,
    options: &PlotOptions,
    width: usize,
    height: usize,
    margin: &Margins,
    plot_w: f64,
    plot_h: f64,
) {
    svg.push_str(&format!(
        r##"<rect x="{}" y="{}" width="{plot_w:.2}" height="{plot_h:.2}" fill="none" stroke="#cbd5e1" stroke-width="1"/>"##,
        margin.left,
        margin.top
    ));
    svg.push_str(&format!(
        r#"<metadata>{{"width":{width},"height":{height},"title":"{}"}}</metadata></svg>"#,
        escape_attr(&options.title)
    ));
}

fn validate_xy(x: &[f64], y: &[f64]) -> Result<()> {
    if x.len() != y.len() {
        return Err(InferustError::DimensionMismatch {
            x_rows: x.len(),
            y_len: y.len(),
        });
    }
    if x.len() < 2 {
        return Err(InferustError::InsufficientData {
            needed: 2,
            got: x.len(),
        });
    }
    ensure_finite(x, "x values")?;
    ensure_finite(y, "y values")
}

fn ensure_finite(values: &[f64], label: &str) -> Result<()> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(InferustError::InvalidInput(format!(
            "{label} must be finite"
        )))
    }
}

fn extent(values: &[f64]) -> (f64, f64) {
    values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        })
}

fn expand_if_flat(min: f64, max: f64) -> (f64, f64) {
    if (max - min).abs() <= f64::EPSILON {
        (min - 1.0, max + 1.0)
    } else {
        let pad = (max - min) * 0.05;
        (min - pad, max + pad)
    }
}

fn scale_y(y: f64, y_min: f64, y_max: f64, height: usize, margin: &Margins) -> f64 {
    let plot_h = (height - margin.top - margin.bottom) as f64;
    let (lo, hi) = expand_if_flat(y_min, y_max);
    (height - margin.bottom) as f64 - ((y - lo) / (hi - lo)) * plot_h
}

fn escape_text(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn escape_attr(text: &str) -> String {
    escape_text(text).replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::{acf_plot_svg, line_plot_svg, residual_plot_svg, scatter_plot_svg, PlotOptions};

    #[test]
    fn line_plot_returns_svg() {
        let svg = line_plot_svg(&[1.0, 2.0, 1.5], PlotOptions::default()).unwrap();
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("<path"));
    }

    #[test]
    fn scatter_plot_checks_dimensions() {
        let err = scatter_plot_svg(&[1.0], &[1.0, 2.0], PlotOptions::default()).unwrap_err();
        assert!(format!("{err}").contains("dimension mismatch"));
    }

    #[test]
    fn residual_plot_has_zero_line() {
        let svg =
            residual_plot_svg(&[1.0, 2.0, 3.0], &[-0.5, 0.0, 0.5], PlotOptions::default()).unwrap();
        assert!(svg.contains("stroke-dasharray"));
    }

    #[test]
    fn acf_plot_returns_bars() {
        let svg = acf_plot_svg(&[1.0, 0.4, -0.1], PlotOptions::default()).unwrap();
        assert!(svg.matches("<line").count() >= 3);
    }
}
