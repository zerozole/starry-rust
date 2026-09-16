//! Surface intensity and rendering for emission, reflection, RV and oblateness.
use crate::{Map, Result, basis, map, oblate, reflected, require, rotation, surface::Projection};
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct View {
    pub upstream_grid: bool,
    pub inclination: f64,
    pub obliquity: f64,
    pub phase: f64,
    pub limb: Vec<f64>,
    pub reflection: Option<Reflection>,
    pub velocity: Option<(f64, f64)>,
    pub oblateness: Option<(f64, Option<oblate::GravityDarkening>, bool)>,
}
#[derive(Clone, Copy, Debug)]
pub struct Reflection {
    pub source: [f64; 3],
    pub radius: f64,
    pub samples: usize,
    pub roughness: f64,
    pub illuminate: bool,
    pub exact: bool,
}
impl Default for View {
    fn default() -> Self {
        Self {
            upstream_grid: false,
            inclination: PI / 2.,
            obliquity: 0.,
            phase: 0.,
            limb: vec![],
            reflection: None,
            velocity: None,
            oblateness: None,
        }
    }
}
struct Lamp {
    unit: [f64; 3],
    distance2: f64,
    b: f64,
    rotation: [[f64; 3]; 3],
    degree: usize,
    polynomial: Vec<f64>,
}
struct Field {
    map: Map,
    lamps: Vec<Lamp>,
    reflection: Option<Reflection>,
    velocity: Option<Vec<f64>>,
    axis_ratio: f64,
    obliquity: f64,
}
impl Field {
    fn new(map: &Map, view: &View, project: bool, filters: bool) -> Result<Self> {
        let mut ratio = 1.;
        let mut obliquity = 0.;
        let u = if filters { view.limb.as_slice() } else { &[] };
        let mapped = if let Some((f, g, normalized)) = view.oblateness.filter(|_| filters) {
            if project {
                ratio = (1. - f * (2. - f) * view.inclination.sin().powi(2)).sqrt();
                obliquity = view.obliquity;
            }
            oblate::observed_map(
                map,
                f,
                if project { view.inclination } else { PI / 2. },
                if project { view.phase } else { 0. },
                u,
                g,
                normalized,
            )?
        } else if project {
            map.projected(view.inclination, view.obliquity, view.phase)?
                .limb_filtered(u)?
        } else {
            map.limb_filtered(u)?
        };
        let mut lamps = vec![];
        if let Some(r) = view.reflection {
            require(
                r.roughness.is_finite() && r.roughness >= 0.,
                "invalid roughness",
            )?;
            if r.illuminate {
                for source in reflected::source_points(r.source, r.radius, r.samples)? {
                    let distance = source[0].hypot(source[1]).hypot(source[2]);
                    require(
                        distance > 0. && distance.is_finite(),
                        "invalid illuminating source",
                    )?;
                    let b = (-source[2] / distance).clamp(-1., 1.);
                    let (degree, polynomial) = reflected::illumination_polynomial(b, r.roughness)?;
                    lamps.push(Lamp {
                        unit: source.map(|x| x / distance),
                        distance2: distance * distance,
                        b,
                        rotation: rotation::axis_angle([0., 0., 1.], source[0].atan2(source[1]))?,
                        degree,
                        polynomial,
                    });
                }
            }
        }
        let velocity = if filters {
            view.velocity
                .map(|(veq, alpha)| {
                    crate::rv::velocity_polynomial(view.inclination, view.obliquity, veq, alpha)
                })
                .transpose()?
        } else {
            None
        };
        Ok(Self {
            map: mapped,
            lamps,
            reflection: view.reflection,
            velocity,
            axis_ratio: ratio,
            obliquity,
        })
    }
    fn value(&self, p: [f64; 3]) -> Result<f64> {
        let mut value = self.map.intensity_xyz(p)?;
        if let Some(velocity) = &self.velocity {
            value *= map::dot(velocity, &basis::polynomial(3, p)?);
        }
        if let Some(r) = self.reflection {
            if !r.illuminate {
                return Ok(PI * value);
            }
            let mut illumination = 0.;
            for lamp in &self.lamps {
                let mu = map::dot(&lamp.unit, &p);
                if mu <= 0. {
                    continue;
                }
                let scatter = if r.exact && r.roughness > 0. {
                    let sig2 = r.roughness * r.roughness;
                    let a = 1. - 0.5 * sig2 / (sig2 + 0.33);
                    let b = 0.45 * sig2 / (sig2 + 0.09);
                    let correction = if p[2] == 0. {
                        (-lamp.b / mu - p[2]).max(0.)
                    } else {
                        (-lamp.b / p[2] - mu).min(-lamp.b / mu - p[2]).max(0.)
                    };
                    a * mu + b * mu * correction
                } else {
                    let q = rotation::apply(lamp.rotation, p);
                    map::dot(&lamp.polynomial, &basis::polynomial(lamp.degree, q)?)
                };
                illumination += scatter / lamp.distance2;
            }
            value *= illumination / self.lamps.len() as f64;
        }
        Ok(value)
    }
}

/// Derivatives of projected flux with respect to occultor x, y and radius.
pub fn occultor_gradient(
    map: &Map,
    view: &View,
    occultor: crate::occultation::Occultor,
) -> Result<[f64; 3]> {
    let field = Field::new(map, view, true, true)?;
    let (s, c) = field.obliquity.sin_cos();
    let aligned = crate::occultation::Occultor::new(
        c * occultor.x + s * occultor.y,
        -s * occultor.x + c * occultor.y,
        occultor.radius,
    )?;
    let g = crate::quadrature::circular_boundary(field.axis_ratio, aligned, |p| field.value(p))?;
    Ok([c * g[0] - s * g[1], s * g[0] + c * g[1], g[2]])
}

/// Per-silhouette translation/axes/angle derivatives with overlapping foregrounds.
pub fn silhouette_gradients(
    map: &Map,
    view: &View,
    ellipses: &[crate::conic::Ellipse],
) -> Result<Vec<[f64; 5]>> {
    let field = Field::new(map, view, true, true)?;
    let aligned: Vec<_> = ellipses
        .iter()
        .map(|e| e.rotated(-field.obliquity))
        .collect();
    let (s, c) = field.obliquity.sin_cos();
    let mut result = vec![];
    for (i, e) in aligned.iter().enumerate() {
        let others: Vec<_> = aligned
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, e)| *e)
            .collect();
        let g =
            crate::quadrature::ellipse_boundary(field.axis_ratio, *e, &others, |p| field.value(p))?;
        result.push([c * g[0] - s * g[1], s * g[0] + c * g[1], g[2], g[3], g[4]]);
    }
    Ok(result)
}

/// Stable projected flux for an arbitrary union of foreground ellipses.
pub fn flux(map: &Map, view: &View, ellipses: &[crate::conic::Ellipse]) -> Result<f64> {
    if let Some(lamp) = view.reflection {
        return reflected::flux_extended_ellipses(
            &map.projected(view.inclination, view.obliquity, view.phase)?
                .limb_filtered(&view.limb)?,
            lamp.source,
            lamp.radius,
            lamp.samples,
            lamp.roughness,
            ellipses,
        );
    }
    let field = Field::new(map, view, true, true)?;
    let aligned: Vec<_> = ellipses
        .iter()
        .map(|e| e.rotated(-field.obliquity))
        .collect();
    crate::quadrature::surface(field.axis_ratio, -1., &aligned, 2e-11, |p| field.value(p))
}

/// Intrinsic latitude/longitude intensity, with optional local filters.
pub fn intensity(
    map: &Map,
    view: &View,
    points: &[[f64; 3]],
    limbdarken: bool,
) -> Result<Vec<f64>> {
    let mut view = view.clone();
    if !limbdarken {
        view.limb.clear();
    }
    let field = Field::new(map, &view, false, true)?;
    points.iter().map(|p| field.value(*p)).collect()
}

/// Pixel-centered rendering. Rectangular/Mollweide projections are intrinsic
/// maps; inclination, rotation and limb/velocity filters apply to orthographic.
pub fn render(
    map: &Map,
    view: &View,
    width: usize,
    height: usize,
    projection: Projection,
) -> Result<Vec<f64>> {
    require(
        width >= 2 && height >= 2 && width.checked_mul(height).is_some_and(|n| n <= 16_000_000),
        "invalid image size",
    )?;
    let orthographic = matches!(projection, Projection::Orthographic);
    let field = Field::new(map, view, orthographic, orthographic)?;
    let mut image = vec![f64::NAN; width * height];
    for iy in 0..height {
        for ix in 0..width {
            let (x, y) = if view.upstream_grid {
                if orthographic {
                    (
                        2. * ix as f64 / (width as f64 - 0.01) - 1.,
                        2. * iy as f64 / (height as f64 - 0.01) - 1.,
                    )
                } else {
                    (
                        2. * ix as f64 / (width - 1) as f64 - 1.,
                        2. * iy as f64 / (height - 1) as f64 - 1.,
                    )
                }
            } else {
                (
                    2. * (ix as f64 + 0.5) / width as f64 - 1.,
                    1. - 2. * (iy as f64 + 0.5) / height as f64,
                )
            };
            let point = match projection {
                Projection::Orthographic => {
                    let (s, c) = field.obliquity.sin_cos();
                    let px = c * x + s * y;
                    let py = (-s * x + c * y) / field.axis_ratio;
                    let r2 = px * px + py * py;
                    if r2 > 1. {
                        continue;
                    }
                    [px, py, (1. - r2).sqrt()]
                }
                Projection::Rectangular => {
                    let lat = y * PI / 2.;
                    let lon = x * PI;
                    [lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos()]
                }
                Projection::Mollweide => {
                    if x * x + y * y > 1. {
                        continue;
                    }
                    let t = y.asin();
                    let lat = ((2. * t + (2. * t).sin()) / PI).clamp(-1., 1.).asin();
                    let lon = PI * x / t.cos();
                    if lon.abs() > PI {
                        continue;
                    }
                    [lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos()]
                }
            };
            image[iy * width + ix] = field.value(point)?;
        }
    }
    Ok(image)
}

/// Intrinsic coordinates of render pixels. Off-surface pixels contain NaNs.
/// Phase and viewing angles affect orthographic grids only.
pub fn coordinate_grid(
    view: &View,
    width: usize,
    height: usize,
    projection: Projection,
) -> Result<Vec<[f64; 2]>> {
    require(
        width >= 2 && height >= 2 && width.checked_mul(height).is_some_and(|n| n <= 16_000_000),
        "invalid grid size",
    )?;
    require(
        view.inclination.is_finite() && (0. ..=PI).contains(&view.inclination),
        "invalid inclination",
    )?;
    let orthographic = matches!(projection, Projection::Orthographic);
    let f = view.oblateness.map_or(0., |v| v.0);
    require(
        f.is_finite() && (0. ..1.).contains(&f),
        "invalid flattening",
    )?;
    let q = (1. - f * (2. - f) * view.inclination.sin().powi(2)).sqrt();
    let rz = rotation::axis_angle([0., 0., 1.], -view.obliquity)?;
    let rx = rotation::axis_angle([1., 0., 0.], view.inclination - PI / 2.)?;
    let ry = rotation::axis_angle([0., 1., 0.], -view.phase)?;
    let mut points = vec![[f64::NAN; 2]; width * height];
    for iy in 0..height {
        for ix in 0..width {
            let (x, y) = if view.upstream_grid {
                let correction = if orthographic { 0.01 } else { 1. };
                (
                    2. * ix as f64 / (width as f64 - correction) - 1.,
                    2. * iy as f64 / (height as f64 - correction) - 1.,
                )
            } else {
                (
                    2. * (ix as f64 + 0.5) / width as f64 - 1.,
                    1. - 2. * (iy as f64 + 0.5) / height as f64,
                )
            };
            let latlon = match projection {
                Projection::Orthographic => {
                    let p = rotation::apply(rz, [x, y, 0.]);
                    let (x, y) = (p[0], p[1] / q);
                    if x * x + y * y > 1. {
                        continue;
                    }
                    let p = rotation::apply(
                        ry,
                        rotation::apply(rx, [x, y, (1. - x * x - y * y).sqrt()]),
                    );
                    [p[1].clamp(-1., 1.).asin(), p[0].atan2(p[2])]
                }
                Projection::Rectangular => [y * PI / 2., x * PI],
                Projection::Mollweide => {
                    if x * x + y * y > 1. {
                        continue;
                    }
                    let theta = y.asin();
                    let lon = PI * x / theta.cos();
                    if lon.abs() > PI {
                        continue;
                    }
                    [
                        ((2. * theta + (2. * theta).sin()) / PI)
                            .clamp(-1., 1.)
                            .asin(),
                        lon,
                    ]
                }
            };
            points[iy * width + ix] = latlon;
        }
    }
    Ok(points)
}
