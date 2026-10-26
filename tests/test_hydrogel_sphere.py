"""Run directly: python tests/test_hydrogel_sphere.py."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.pca_vca.hydrogel_sphere import SphereModel, concentration, radial_curve, rim_fraction, scale_pixels, load_scan, ROOT, STEM, NAMES
from simulator.core.physics import lorentzian_unit_area
from analysis.pca_vca.hydrogel_flexible_mixture import fit_mixture


def test_sphere():
    nu = np.linspace(4, 8, 81)
    bg = lorentzian_unit_area(nu, 5.5, .3)
    axis = np.arange(-6, 7, 2)
    x, y = np.meshgrid(axis, axis)
    xy = np.column_stack((x.ravel(), y.ravel()))
    model = SphereModel(nu, bg, xy, 100., xy_order=5, z_order=48)
    p = np.array([.04, -.06, .18, .2, .2, 5.9, 5.6, 2, .9, .6, 2,
                  1500, 1000, 1, 1000, 1., 0.])
    np.testing.assert_allclose(concentration([0, .5, 1, 2]), [1, .75, 0, 0])
    np.testing.assert_allclose(radial_curve([0, 1], 5.9, 5.6, 2), [5.9, 5.6])
    np.testing.assert_allclose(rim_fraction([0, .8, .9, 1, 1.1], .2), [1, 1, .5, 0, 0])
    np.testing.assert_array_equal(rim_fraction([0, .9, 1, 2], 0), [1, 1, 0, 0])
    predicted = model.remodel(p)
    hydrogel, water = model.remodel(p, return_components=True)
    np.testing.assert_allclose(hydrogel+water, predicted)
    assert np.all(hydrogel >= 0) and np.all(water >= 0)
    assert predicted.shape == (49, 81) and np.isfinite(predicted).all()
    assert np.all(predicted >= 0)
    # Same spectrum/intensity everywhere must survive mixing unchanged.
    uniform = p.copy()
    uniform[5:7] = 5.5
    uniform[8:10] = .6
    uniform[[11, 12, 14]] = 1000
    expected = np.broadcast_to(1000*bg/bg.sum(), predicted.shape)
    np.testing.assert_allclose(model.remodel(uniform), expected, rtol=1e-12)
    uniform[16] = .4
    np.testing.assert_allclose(model.remodel(uniform), expected, rtol=1e-12)
    # A sharp intensity edge must not depend on the spectral lookup resolution.
    sharp = uniform.copy()
    sharp[11:14] = [800000, 0, .25]
    finer_radial = SphereModel(nu, bg, xy, 100., xy_order=5, z_order=48, radial_nodes=257)
    np.testing.assert_allclose(model.remodel(sharp), finer_radial.remodel(sharp), rtol=1e-12)
    outside = p.copy()
    outside[0] = 10
    np.testing.assert_allclose(model.remodel(outside), expected, rtol=1e-12)
    outside_gel, outside_water = model.remodel(outside, return_components=True)
    np.testing.assert_array_equal(outside_gel, np.zeros_like(expected))
    np.testing.assert_allclose(outside_water, expected)
    mirrored = p.copy()
    mirrored[2] *= -1
    np.testing.assert_allclose(model.remodel(mirrored), predicted, atol=1e-10)
    # Direct high-order continuum evaluation checks the radial lookup approximation.
    fine = SphereModel(nu, bg, xy, 100., xy_order=9, z_order=96, radial_nodes=513)
    assert np.sqrt(np.mean((predicted-fine.remodel(p))**2)) / predicted.max() < .001
    # Recover several coupled spectral parameters from a deliberately perturbed start.
    initial = p.copy()
    initial[[5, 8, 11]] = [5.8, .75, 1200]
    lower, upper = np.full(len(NAMES), -np.inf), np.full(len(NAMES), np.inf)
    lower[[5, 8, 11]] = [4, .03, 0]
    upper[[5, 8, 11]] = [8, 3, 10000]
    result = model.fit(predicted, initial, (lower, upper), max_nfev=40,
                       fit_names=('bfs_center', 'lw_center', 'int_center'))
    assert result.success and result.mse < 1e-12
    np.testing.assert_allclose(result.parameters, p, rtol=1e-6, atol=1e-6)
    gains = np.linspace(0, 1.8, len(predicted))
    bright = predicted*gains[:, None]
    scaled, recovered = scale_pixels(predicted, bright)
    np.testing.assert_allclose(scaled, bright, atol=1e-12)
    np.testing.assert_allclose(recovered, gains, atol=1e-12)
    # Zero model has no identifiable gain; a negative dot product clips to zero.
    zero, zero_gains = scale_pixels(np.array([[0., 0.], [1., 2.]]), np.array([[2., 3.], [-1., -2.]]))
    np.testing.assert_array_equal(zero, np.zeros((2, 2)))
    np.testing.assert_array_equal(zero_gains, [0, 0])
    initial = p.copy()
    initial[[5, 8]] = [5.8, .75]
    result = model.fit(bright, initial, (lower, upper), max_nfev=40,
                       fit_names=('bfs_center', 'lw_center', 'water_intensity'), pixel_scaling=True)
    assert result.success and result.mse < 1e-12
    assert 'water_intensity' not in result.fit_names and result.parameters[14] == p[14]
    np.testing.assert_allclose(result.parameters, p, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.pixel_scale, gains, atol=1e-6)
    # Radial intensity must remain adjustable while per-pixel gains are active.
    initial = p.copy()
    initial[11] *= .6
    result = model.fit(bright, initial, (lower, upper), max_nfev=40,
                       fit_names=('int_center',), pixel_scaling=True)
    assert result.success and result.mse < 1e-12
    assert 'int_center' in result.fit_names
    np.testing.assert_allclose(result.parameters[11], p[11], rtol=1e-5)
    tapered = p.copy()
    tapered[16] = .35
    target = model.remodel(tapered)*gains[:, None]
    initial = tapered.copy()
    initial[16] = .15
    lower[16], upper[16] = 0, 1
    result = model.fit(target, initial, (lower, upper), max_nfev=40,
                       fit_names=('rim_width',), pixel_scaling=True)
    assert result.success and result.mse < 1e-10
    np.testing.assert_allclose(result.parameters[16], .35, atol=1e-6)
    # Recover spectral percentages independently of overall brightness.
    h = lorentzian_unit_area(nu, 5.9, .4)
    h, b = h/h.sum(), bg/bg.sum()
    truth = np.array([[400., 600.], [800., 0.], [0., 300.]])
    mix = truth[:, :1]*h + truth[:, 1:]*b
    recovered, areas, shares = fit_mixture(np.tile(h, (3, 1)), b, mix)
    np.testing.assert_allclose(recovered, mix, atol=1e-10)
    np.testing.assert_allclose(areas, truth, atol=1e-8)
    np.testing.assert_allclose(shares, [.4, 1., 0.], atol=1e-10)
    _, _, absent = fit_mixture(np.zeros((1, len(nu))), b, b[None, :]*100)
    np.testing.assert_allclose(absent, [0.])
    _, _, ambiguous = fit_mixture(b[None, :], b, b[None, :]*100)
    assert np.isnan(ambiguous[0])
    _, _, empty = fit_mixture(h[None, :], b, np.zeros((1, len(nu))))
    assert np.isnan(empty[0])
    for bad in (np.zeros(81), np.full(81, np.nan), -np.ones(81)):
        try:
            SphereModel(nu, bad, xy, 100)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid background accepted')
    try:
        model.loss(p, np.zeros((1, 81)))
    except ValueError:
        pass
    else:
        raise AssertionError('Broadcastable but incorrect observed shape accepted')
    scan = ROOT/'data/csv'/f'{STEM}.csv'
    em = ROOT/'outputs/background_endmember'/f'{STEM}_background_endmember.csv'
    if scan.exists() and em.exists():
        # Saved decimal frequencies can round just below their in-memory values.
        freq, spectra, background, positions, table = load_scan(scan, em)
        assert spectra.shape == (len(table), len(freq))
        assert len(background) == len(freq) and positions.shape == (len(table), 2)


if __name__ == '__main__':
    test_sphere()
    print('Sphere checks passed: concentration, limits, symmetry, integration, recovery, validation.')
