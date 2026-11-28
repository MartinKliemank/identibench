"""Tests for the IAS (instantaneous angular speed) datasets and benchmarks."""

import dataclasses
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

import identibench as idb
from identibench.datasets.ias import GridwiseEstimation, WindowedEstimation, _common, harmonics
from identibench.datasets.ias.harmonics import HarmonicSettings
from identibench.datasets.ias import planetary_gearbox as planetary
from identibench.datasets.ias.ball_bearing import (
    BenchmarkBallBearing_Estimation,
    BenchmarkBallBearing_Simulation,
    ball_bearing_dataset,
)


# ───────────────────────── encoder → IAS ─────────────────────────


def _pulse_train(ias_hz, fs, ppr):
    """An analog encoder pulse train whose instantaneous speed follows `ias_hz(t)`."""
    t = np.arange(int(fs * 2)) / fs
    phase = np.cumsum(2 * np.pi * ppr * ias_hz(t) / fs)
    return t, np.sin(phase)


def test_encoder_pulse_to_ias_recovers_constant_frequency():
    fs, f_rot, ppr = 20_000.0, 20.0, 64
    t, pulse = _pulse_train(lambda t: np.full_like(t, f_rot), fs, ppr)

    ias, sl = _common.encoder_pulse_to_ias(pulse, fs, pulses_per_revolution=ppr, cutoff_order=4.0)

    assert ias.shape == (sl.stop - sl.start,)
    # The IAS is defined only where the encoder measures; that span is nearly the whole
    # recording here (one pulse period at the head), and is never extrapolated past.
    assert sl.stop - sl.start > 0.99 * len(pulse)
    np.testing.assert_allclose(ias, f_rot, rtol=0.02)


def test_encoder_pulse_to_ias_tracks_a_speed_sweep():
    """Bandwidth is the point of the order-domain reference; a constant target cannot show it."""
    fs, ppr = 20_000.0, 64
    # 20 Hz mean with a 2 Hz peak-to-peak wobble at 2 orders -- inside a 4-order cutoff.
    profile = lambda t: 20.0 + 1.0 * np.sin(2 * np.pi * 2 * 20.0 * t)  # noqa: E731
    t, pulse = _pulse_train(profile, fs, ppr)
    ias, sl = _common.encoder_pulse_to_ias(pulse, fs, pulses_per_revolution=ppr, cutoff_order=4.0)

    expected = profile(t[sl])
    interior = slice(len(ias) // 10, -len(ias) // 10)  # filter transients at the ends
    np.testing.assert_allclose(ias[interior], expected[interior], rtol=0.05)
    # and it really is tracking, not just sitting at the mean
    assert np.ptp(ias[interior]) > 1.5


def test_encoder_pulse_to_ias_slice_keeps_channels_aligned():
    fs, ppr = 20_000.0, 64
    t, pulse = _pulse_train(lambda t: np.full_like(t, 20.0), fs, ppr)
    vibration = np.sin(2 * np.pi * 137.0 * t)

    ias, sl = _common.encoder_pulse_to_ias(pulse, fs, pulses_per_revolution=ppr, cutoff_order=4.0)

    assert len(vibration[sl]) == len(ias)
    assert 0 <= sl.start < sl.stop <= len(pulse)


def test_order_domain_lowpass_passes_and_stops():
    ppr = 128
    theta = np.arange(ppr * 200) / ppr  # 200 revolutions, uniform in angle
    for order, expect_pass in ((1.0, True), (40.0, False)):
        x = np.sin(2 * np.pi * order * theta)
        y = _common.order_domain_lowpass(x, cutoff_order=4.0, pulses_per_revolution=ppr)
        interior = y[len(y) // 10 : -len(y) // 10]
        gain = np.ptp(interior) / np.ptp(x)
        if expect_pass:
            assert gain > 0.9, f"order {order} should pass, gain={gain}"
        else:
            assert gain < 0.01, f"order {order} should be stopped, gain={gain}"


def test_order_domain_lowpass_rejects_impossible_cutoff():
    x = np.zeros(1000)
    with pytest.raises(ValueError, match="Nyquist"):
        _common.order_domain_lowpass(x, cutoff_order=15.0, pulses_per_revolution=20)
    with pytest.raises(ValueError, match="Nyquist"):
        _common.order_domain_lowpass(x, cutoff_order=0.0, pulses_per_revolution=1024)


def test_rising_edge_times_does_not_double_count_exact_zeros():
    """A sample landing exactly on zero is one edge, not two.

    The symmetric ramp is already normalized, so sample 5 of every repeat is exactly 0.0.
    A `np.diff(np.sign(...)) > 0` detector steps -1 -> 0 and 0 -> +1 separately and reports
    two rising edges there, halving the apparent period.
    """
    fs = 1000.0
    ramp = np.linspace(-1.0, 1.0, 11)  # ..., -0.2, 0.0, 0.2, ...
    signal = np.tile(ramp, 10)

    edges = _common.rising_edge_times(signal, fs)

    assert len(edges) == 10  # one upward crossing per repeat
    np.testing.assert_allclose(np.diff(edges) * fs, len(ramp), atol=1e-9)
    # the refined crossing lands exactly on the zero sample
    np.testing.assert_allclose(edges[0] * fs, 5.0, atol=1e-9)
    # the discarded alternative would have found twice as many
    assert len(np.argwhere(np.diff(np.sign(signal)) > 0).ravel()) == 20


# ───────────────────────── zebra tape reconstruction ─────────────────────────


def _synthetic_zebra(
    centres, ias_profile, fs=5_000.0, duration=200.0, drop=0.15, jitter_s=2e-5, seed=0, bar_spread=0.06, merge=0.0
):
    """A synthetic planetary recording: 1PR carrier reference + irregular sun-shaft zebra tape.

    Returns ``(recording, true_idx, ias_profile_on_grid, n_samples)``: ``recording`` is what
    :func:`calibrate_zebra` takes (``t_ref``, ``t_rise``, ``t_fall``), and ``true_idx`` is the true
    global stripe index of every rising edge, counting stripes in angular order from ``centres``.
    Each dark bar is centred on its stripe and 0.35 of a spacing wide, give or take a per-stripe
    ``bar_spread`` drawn per recording (as a remounted sensor reads the same tape). Bars are randomly
    dropped, a fraction ``merge`` run into the next one, and edge times are jittered.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * fs)) / fs
    ias = ias_profile(t)
    theta = np.cumsum(ias) / fs  # sun revolutions
    t_ref = np.interp(np.arange(1.0, np.floor(theta[-1] / planetary._SUN_PER_CARRIER_REV)), theta / planetary._SUN_PER_CARRIER_REV, t)

    centres = np.sort(np.asarray(centres) % 1.0)
    n = len(centres)
    revs = np.arange(0.0, np.floor(theta[-1]))
    centre_angle = (revs[:, None] + centres[None, :]).ravel()
    half_width = np.tile(0.35 + rng.uniform(-bar_spread, bar_spread, n), len(revs)) / (2 * n)
    true_idx = (revs[:, None] * n + np.arange(n)[None, :]).ravel().astype(np.int64)
    rise_angle, fall_angle = centre_angle + half_width, centre_angle - half_width
    keep = (fall_angle > theta[0]) & (rise_angle < theta[-1]) & (rng.random(len(centre_angle)) > drop)
    merged = keep & np.r_[keep[1:], False] & (rng.random(len(keep)) < merge)  # bar k runs into bar k+1
    has_rise, has_fall = keep & ~merged, keep & ~np.r_[False, merged[:-1]]
    recording = dict(
        t_ref=t_ref,
        t_rise=np.interp(rise_angle[has_rise], theta, t) + rng.normal(0, jitter_s, has_rise.sum()),
        t_fall=np.interp(fall_angle[has_fall], theta, t) + rng.normal(0, jitter_s, has_fall.sum()),
    )
    return recording, true_idx[has_rise], ias, len(t)


def _irregular_tape(n_stripes, seed):
    """Stripe angles nudged off a perfect grid, the way a real tape is. Kept within +-0.12 of a
    spacing: the bootstrap fitter cannot resolve two stripes closer together than half the mean
    spacing (see `_fit_template_from_phase`)."""
    rng = np.random.default_rng(seed)
    return np.sort((np.arange(n_stripes) + rng.uniform(-0.12, 0.12, n_stripes)) / n_stripes) % 1.0


def _circular_spacings(angles):
    """Gap to the next stripe, wrapping around -- invariant to how the tape is clocked."""
    return np.sort(np.diff(np.concatenate([angles, [angles[0] + 1.0]])))


def test_zebra_template_handles_detections_across_phase_wrap():
    """A stripe at phase zero must not move to half a turn or duplicate another stripe."""
    n_stripes, n_revs = 76, 200
    planted = np.arange(n_stripes) / n_stripes
    jitter = np.tile([-0.0002, 0.0002], n_revs // 2)
    phases = ((planted[None, :] + jitter[:, None]) % 1.0).ravel()

    template, counts = planetary._fit_template_from_phase(phases, n_revs)

    assert len(template) == n_stripes
    distances = np.abs((template[:, None] - planted[None, :] + 0.5) % 1.0 - 0.5)
    np.testing.assert_allclose(distances.min(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(_circular_spacings(template), 1 / n_stripes, atol=1e-12)
    np.testing.assert_array_equal(counts, n_revs)

    # Duplicate template angles used to make this fail inside PchipInterpolator.
    indices = np.arange(n_stripes * 100)
    times = indices / (n_stripes * 10.0)
    ias, sl = planetary.reconstruct_ias(times, indices, template, fs=5000, signal_len=50000)
    assert len(ias) == sl.stop - sl.start
    np.testing.assert_allclose(ias, 10.0, rtol=1e-6)


def test_zebra_round_trip_recovers_labels_template_and_speed(monkeypatch):
    fs, n_stripes = 5_000.0, 76
    planted = _irregular_tape(n_stripes, seed=7)

    def fast(t):
        return 10.0 + 0.5 * np.sin(2 * np.pi * 0.02 * t)

    def slow(t):
        return 8.0 + 0.4 * np.sin(2 * np.pi * 0.015 * t)

    # The second recording is clocked 0.31 rev around from the first, as a rig reassembled between
    # test phases would be, and reads every bar with a different width; the bootstrap must align the
    # clocking so that stripe k is the same stripe in both, and calibrating at the centres must see
    # through the widths. 1 % of bars merge into their neighbour.
    recordings, truth = {}, {}
    for stem, centres, profile, seed in (("a", planted, fast, 1), ("b", (planted + 0.31) % 1.0, slow, 2)):
        recordings[stem], true_idx, ias, n_samples = _synthetic_zebra(centres, profile, fs=fs, seed=seed, merge=0.01)
        truth[stem] = (true_idx, ias, n_samples)
    monkeypatch.setattr(planetary, "_TEMPLATE_GROUPS", (("a", "b"),))  # calibrate the two together

    labels = planetary.calibrate_zebra(recordings)

    for stem, (t, idx, template) in labels.items():
        true_idx, ias_true, n_samples = truth[stem]
        # Every labelled centre carries its true stripe index, up to where the numbering starts.
        offset = idx - true_idx[np.searchsorted(recordings[stem]["t_rise"], t)]
        assert np.all(offset == offset[0])
        assert len(t) > 0.97 * len(true_idx)

        # The calibrated stripe centres are the planted ones, up to which stripe is numbered 0 -- to
        # far better than the rising edges, which differ from the centres by ~0.03 of a spacing.
        misfit = [np.std((template - np.roll(planted, -s) + 0.5) % 1.0 - 0.5) for s in range(n_stripes)]
        assert min(misfit) < 2e-5  # rev, i.e. 0.0015 of a spacing

        # ...and the reconstructed IAS tracks the planted speed.
        ias, sl = planetary.reconstruct_ias(t, idx, template, fs, n_samples)
        assert len(ias) == sl.stop - sl.start
        interior = slice(len(ias) // 20, -len(ias) // 20)  # filter transients at the ends
        got, want = ias[interior], ias_true[sl][interior]

        # Judged on robust statistics, not sample-wise: stripe timing jitter propagates into
        # the finite-difference rate, and a 15-order cutoff against a 38-order Nyquist leaves
        # a few percent of it in by design -- that is the bandwidth/noise trade being made.
        rel_err = np.abs(got - want) / want
        assert np.median(rel_err) < 0.01
        assert np.percentile(rel_err, 99) < 0.03
        # unbiased, and actually following the profile rather than sitting at its mean
        assert abs(got.mean() - want.mean()) / want.mean() < 0.005
        assert np.corrcoef(got, want)[0, 1] > 0.95


def test_count_zebra_stripes_survives_a_near_stop():
    """Stripe labels stay exact through a stop and restart with a quarter of the stripes missing,
    which labelling each pulse from the 1PR interpolated between its pulses gets wrong for well over
    a thousand pulses."""
    rec, true_idx, _, _ = _synthetic_zebra(
        _irregular_tape(76, seed=11), lambda t: 10.0 - 8.5 * np.exp(-0.5 * ((t - 100.0) / 1.5) ** 2), seed=5, drop=0.25
    )
    phase, n_revs = planetary._zebra_phase(rec["t_ref"], rec["t_rise"])
    template, _, pooled_hist = planetary.pool_zebra_templates([phase], [n_revs])

    t, idx, n_dropped = planetary.count_zebra_stripes(
        rec["t_rise"], rec["t_ref"], template, planetary.zebra_phase_offset(phase, pooled_hist)
    )

    offset = idx - true_idx[np.searchsorted(rec["t_rise"], t)]
    assert np.all(offset == offset[0])
    assert len(t) > 0.95 * len(true_idx)


def test_reference_events_time_the_dip_midpoint():
    """A 1PR pulse is timed at the middle of its dip, moved earlier by the channel delay, whatever
    the dip's width; a dip cut off by the start of the recording is skipped, not half-timed."""
    fs = 10_000.0
    t = np.arange(int(5 * fs)) / fs
    centres = np.array([0.5, 1.7, 2.9, 4.1])
    signal = np.where(t < 0.02, 0.5, 10.4)  # the recording starts inside a dip
    for centre, half_width in zip(centres, (0.015, 0.03, 0.02, 0.025)):
        # flat 0.5 V bottom, 4 ms linear flanks, clipped at 10.4 V like the real pickup
        signal = np.minimum(signal, np.clip(0.5 + (np.abs(t - centre) - half_width) / 0.004 * 9.9, 0.5, 10.4))

    events = planetary.reference_events(signal, fs)

    np.testing.assert_allclose(events, centres - planetary._REF_DELAY_S, atol=1e-6)


def test_zebra_reconstruction_is_truncated_not_extrapolated():
    """The span before the first labelled stripe is dropped, never filled in."""
    fs = 5_000.0
    rec, _, _, n_samples = _synthetic_zebra(_irregular_tape(76, seed=3), lambda t: np.full_like(t, 9.0), fs=fs, seed=4)

    ((t, idx, template),) = planetary.calibrate_zebra({"a": rec}).values()
    _, sl = planetary.reconstruct_ias(t, idx, template, fs, n_samples)

    # Counting starts at the first 1PR pulse, so a real head is cut off.
    assert sl.start / fs >= t[0] - 1 / fs
    assert sl.stop / fs <= t[-1] + 1 / fs
    assert sl.start > 0 and sl.stop < n_samples


# The raw planetary recordings (the folder holding the archive's g1_crack/ and g2_crack/); the check
# of the label pipeline against them runs only where this is set.
_PLANETARY_RAW = os.environ.get("IDENTIBENCH_PLANETARY_RAW")

# Per recording, as dataset version 4 was built: stripes labelled, and the rms timing residual of each
# labelled stripe centre against its neighbours on the calibrated template (in stripe spacings).
_PLANETARY_EXPECTED = {
    "G1_P0_38400Hz": (602197, 0.0070),
    "G1_P1_38400Hz": (449295, 0.0076),
    "G1_P1_slow_38400Hz": (237779, 0.0214),
    "G1_P2_38400Hz": (606805, 0.0052),
    "G1_P3_38400Hz": (580061, 0.0072),
    "G1_P4_38400Hz": (592069, 0.0053),
    "G1_P5_38400Hz": (549604, 0.0070),
    "G1_P6_38400Hz": (674792, 0.0053),
    "G1_P7_38400Hz": (505329, 0.0099),
    "G2_P0_38400Hz": (1068166, 0.0037),
    "G2_P1_38400Hz": (917700, 0.0028),
    "G2_P2_38400Hz": (814991, 0.0027),
    "G2_P3_38400Hz": (891098, 0.0028),
    "G2_P4_38400Hz": (711623, 0.0030),
    "G2_P5_38400Hz": (753277, 0.0027),
}


@pytest.mark.skipif(_PLANETARY_RAW is None, reason="set IDENTIBENCH_PLANETARY_RAW to the raw planetary recordings")
def test_planetary_labels_hold_up_on_the_real_recordings():
    """At least as many stripes labelled, fitting the calibrated stripe positions at least as tightly,
    as when version 4 was built. A mislabelled run both drops revolutions and fits worse."""
    import scipy.io

    recordings = {}
    for f in sorted(Path(_PLANETARY_RAW).glob("*_crack/*.MAT")):
        mat = scipy.io.loadmat(f, variable_names=["Channel_5_Data", "Channel_6_Data", "File_Header"])
        fs = planetary._parse_fs(mat)
        zebra = mat["Channel_5_Data"].squeeze().astype(float)
        recordings[f.stem] = dict(
            t_ref=planetary.reference_events(mat["Channel_6_Data"].squeeze(), fs),
            t_rise=_common.rising_edge_times(zebra, fs),
            t_fall=_common.rising_edge_times(-zebra, fs),
        )

    labels = planetary.calibrate_zebra(recordings)

    assert labels.keys() == _PLANETARY_EXPECTED.keys()
    for stem, (t, idx, template) in labels.items():
        n_labelled, residual = _PLANETARY_EXPECTED[stem]
        a = planetary._theta(idx, template)
        between = t[:-2] + (t[2:] - t[:-2]) * (a[1:-1] - a[:-2]) / (a[2:] - a[:-2])
        r = (t[1:-1] - between) / ((t[2:] - t[:-2]) / (idx[2:] - idx[:-2]))
        assert len(t) >= 0.998 * n_labelled, stem
        assert np.sqrt(np.mean(r**2)) <= 1.05 * residual, stem


# ───────────────────────── speed-locked harmonics ─────────────────────────

# 16-revolution frames: fine enough bins that measuring a one-order band takes little from its neighbours
_SETTINGS = HarmonicSettings(segment_revs=16)


def _machine(speed_hz, fs, orders=(1.0, 2.0, 3.0), amplitudes=(1.0, 0.5, 0.3), noise=0.05, seed=0):
    """Vibration of a machine turning at `speed_hz` (one value per sample): sines locked to the shaft
    angle at `orders`, plus white noise."""
    rng = np.random.default_rng(seed)
    angle = np.cumsum(speed_hz) / fs
    locked = sum(a * np.sin(2 * np.pi * o * angle + rng.uniform(0, 2 * np.pi)) for o, a in zip(orders, amplitudes))
    return locked + noise * rng.standard_normal(len(speed_hz))


def _reference(sig, speed, fs, orders=(1.0, 2.0, 3.0)):
    grid = harmonics.AngleGrid.from_ias(speed, fs, _SETTINGS)
    return harmonics.HarmonicReference.from_signal(sig, grid, np.asarray(orders))


def test_find_lines_keeps_locked_lines_and_drops_fixed_tones():
    """Under a speed sweep a fixed-Hz tone drifts through the orders and loses its phase relation."""
    fs = 2000.0
    t = np.arange(int(fs * 20)) / fs
    speed = 15 + 15 * t / t[-1]
    sig = _machine(speed, fs, orders=(1.0, 2.5), amplitudes=(1.0, 0.5)) + 0.5 * np.sin(2 * np.pi * 150 * t)

    lines = harmonics.find_lines(harmonics.angle_spectra(sig, harmonics.AngleGrid.from_ias(speed, fs, _SETTINGS)))

    assert np.isclose(lines, 1.0, atol=0.25).any() and np.isclose(lines, 2.5, atol=0.25).any()
    # the 150 Hz tone sweeps through orders 10..5; where the sweep starts and ends it lingers at one order
    # for a few frames and does look locked there (that is what `dataset_lines`' consensus removes)
    assert not ((lines > 5.25) & (lines < 9.75)).any()


def test_find_lines_raises_no_false_alarms_on_noise():
    fs = 2000.0
    t = np.arange(int(fs * 20)) / fs
    speed = 15 + 15 * t / t[-1]
    noise = np.random.default_rng(0).standard_normal(len(t))
    assert len(harmonics.find_lines(harmonics.angle_spectra(noise, harmonics.AngleGrid.from_ias(speed, fs, _SETTINGS)))) == 0


def test_dataset_lines_drops_fixed_hz_content_by_consensus(tmp_path, monkeypatch):
    """At constant speed a fixed tone looks locked, but at a different order in every recording."""
    monkeypatch.setenv("IDENTIBENCH_WORKERS", "1")
    fs, n = 2000.0, 20_000
    t = np.arange(n) / fs
    speeds = (12.0, 15.0, 19.0, 23.0, 27.0, 31.0)
    for i, speed in enumerate(speeds):
        sig = _machine(np.full(n, speed), fs, orders=(2.02,), amplitudes=(1.0,), seed=i) + np.sin(2 * np.pi * 120 * t)
        _write_ias_file(tmp_path / "train" / f"rec_{i}.hdf5", np.full(n, speed), {"Acc_x": sig}, fs=fs)

    lines = harmonics.dataset_lines(tmp_path, ["Acc_x"], _SETTINGS)

    assert lines.n_recordings == len(speeds)
    assert lines.strongest(1) == pytest.approx([2.0])  # the 2.02 line on the 1/16-order bin grid
    assert lines.refined_orders[0] == pytest.approx(2.02, abs=0.005)
    assert lines.orders[lines.highest_snr(1)] == pytest.approx([2.0])
    assert lines.median_snr_db[0] > 10  # the unit sine stands far out of the 0.05 white noise
    for speed in speeds:
        assert not np.isclose(lines.orders, 120 / speed, atol=0.3).any(), speed
    lines.save(tmp_path / "lines.json")
    back = harmonics.LineList.load(tmp_path / "lines.json")
    np.testing.assert_allclose(back.orders, lines.orders)
    np.testing.assert_allclose(back.median_snr_db, lines.median_snr_db)
    np.testing.assert_allclose(back.refined_orders, lines.refined_orders)
    assert back.settings == lines.settings and back.splits == lines.splits


@pytest.mark.parametrize("kind", ["gaussian", "impulsive"])
def test_shaped_disturbances_put_every_line_at_the_target_snr(kind):
    """At constant speed every line's one-order band holds exactly its own power over 10**(S/10)."""
    fs = 2000.0
    speed = np.full(int(fs * 60), 20.0)
    sig = _machine(speed, fs)
    ref = _reference(sig, speed, fs)

    for target_snr_db in (15.0, -7.5):
        noisy = _common.add_disturbances(sig, target_snr_db, np.random.default_rng(1), ref, speed, None, (kind,))
        snr = ref.line_snr(noisy - sig).long_term_snr_db()
        np.testing.assert_allclose(snr, target_snr_db, atol=1.0, err_msg=f"{kind} at {target_snr_db} dB")


def test_strongest_line_stays_near_the_target_under_a_sweep():
    """A stationary noise cannot follow sweeping lines exactly, but the strongest one stays close."""
    fs = 2000.0
    t = np.arange(int(fs * 60)) / fs
    speed = 15 + 15 * t / t[-1]
    sig = _machine(speed, fs)
    ref = _reference(sig, speed, fs)
    noisy = _common.add_disturbances(sig, 0.0, np.random.default_rng(1), ref, speed, None, ("gaussian",))
    snr = ref.line_snr(noisy - sig)
    assert snr.long_term_snr_db()[snr.strongest_line()] == pytest.approx(0.0, abs=2.0)


def test_line_snr_reports_effective_snr_and_the_strongest_line_in_band():
    fs = 2000.0
    speed = np.full(int(fs * 30), 20.0)
    orders = np.array([1.0, 2.0, 3.0])
    sig = _machine(speed, fs, amplitudes=(0.3, 1.0, 0.5), noise=0.3)  # order 2 strongest
    noisy = _common.add_disturbances(sig, 0.0, np.random.default_rng(1), _reference(sig, speed, fs), speed, None, ("gaussian",))

    snr = harmonics.line_snr(sig, noisy, speed, fs, orders, _SETTINGS)

    assert snr.orders[snr.strongest_line()] == 2.0
    assert snr.orders[snr.strongest_line(max_freq_hz=45.0)] == 1.0  # (2 + 0.5) x 20 Hz is above 45 Hz
    assert (snr.long_term_snr_db(effective=True) < snr.long_term_snr_db()).all()  # the clean background counts too
    assert snr.snr_db().shape == (len(snr.frame_time), len(snr.orders))
    assert (np.diff(snr.frame_time) > 0).all() and 0 < snr.frame_time[0] < snr.frame_time[-1] < len(sig) / fs


def test_reference_needs_a_line_above_the_background():
    fs = 2000.0
    speed = np.full(int(fs * 20), 20.0)
    noise = np.random.default_rng(0).standard_normal(len(speed))
    with pytest.raises(ValueError, match="no line"):
        _reference(noise, speed, fs)


def test_frame_speed_is_exact_at_constant_speed():
    """Every frame -- the last one too, whatever the grid's tail -- reports the recording's speed."""
    for n in (20_000, 20_500, 21_000, 21_500):
        grid = harmonics.AngleGrid.from_ias(np.full(n, 20.0), 2000.0, _SETTINGS)
        np.testing.assert_allclose(grid.frame_speed, 20.0, rtol=1e-9, err_msg=f"{n} samples")


def test_bins_only_a_silent_line_covers_are_interpolated(monkeypatch):
    """A band covered only by a line at zero power is not occupied: the reference continues there at the
    level of the highest line that has power, instead of taking the float sums' rounding residue (which
    punched 80-140 dB holes into some real references)."""
    fs = 2000.0
    t = np.arange(int(fs * 30)) / fs
    speed = 15 + 15 * t / t[-1]
    original = harmonics._frame_band_power

    def silence_the_top_line(x_angle, grid, bins_per_band, frames):
        power = original(x_angle, grid, bins_per_band, frames)
        if bins_per_band.shape == (3, 2 * harmonics.MAIN_LOBE_BINS + 1):  # the reference's line powers
            power[:, -1] = 0.0
        return power

    monkeypatch.setattr(harmonics, "_frame_band_power", silence_the_top_line)
    ref = _reference(_machine(speed, fs), speed, fs)

    only_top_line = ref.freqs > 2.5 * speed.max()  # above every band of orders 1 and 2
    assert not ref.occupied[only_top_line].any()
    duration = (ref.grid.hop_len / ref.grid.samples_per_rev) / ref.grid.frame_speed[ref.frames]
    line_2_density = np.nansum(duration * ref.line_power[:, 1] / ref.grid.frame_speed[ref.frames]) / duration.sum()
    np.testing.assert_allclose(ref.density[only_top_line], line_2_density, rtol=1e-9)


def test_flat_extension_follows_the_outermost_lines():
    """Below the lowest and above the highest line the reference holds their time-weighted mean density."""
    fs = 2000.0
    speed = np.full(int(fs * 30), 20.0)
    ref = _reference(_machine(speed, fs), speed, fs)
    assert ref.density[1] == pytest.approx(np.nanmean(ref.line_power[:, 0]) / 20.0, rel=1e-9)
    assert ref.density[-1] == pytest.approx(np.nanmean(ref.line_power[:, -1]) / 20.0, rel=1e-9)


# ───────────────────────── seeded disturbances ─────────────────────────


def _disturb(sig, speed, fs, target_snr_db, seed, disturbance_types=("harmonic", "gaussian", "impulsive"), interferer=None):
    """`add_disturbances` against the harmonic reference of `sig`, with a 27 Hz interferer by default."""
    return _common.add_disturbances(
        sig,
        target_snr_db,
        np.random.default_rng(seed),
        _reference(sig, speed, fs),
        speed,
        interferer_ias_hz=np.full_like(sig, 27.0) if interferer is None else interferer,
        disturbance_types=disturbance_types,
    )


def test_add_disturbances_deterministic_and_seed_sensitive():
    fs = 2000.0
    speed = np.full(int(fs * 10), 20.0)
    sig = _machine(speed, fs)

    out_a = _disturb(sig, speed, fs, 15, seed=7)
    np.testing.assert_array_equal(out_a, _disturb(sig, speed, fs, 15, seed=7))
    assert not np.array_equal(out_a, _disturb(sig, speed, fs, 15, seed=8))


def test_combined_splits_the_noise_power_evenly():
    """Two shaped components at half the power each put the lines at the target together."""
    fs = 2000.0
    speed = np.full(int(fs * 60), 20.0)
    sig = _machine(speed, fs)
    ref = _reference(sig, speed, fs)

    noisy = _common.add_disturbances(sig, 0.0, np.random.default_rng(1), ref, speed, None, ("gaussian", "impulsive"))
    np.testing.assert_allclose(ref.line_snr(noisy - sig).long_term_snr_db(), 0.0, atol=1.0)


def test_impulsive_disturbance_stays_bounded_across_sample_rates():
    """No single Levy sample may dominate the noise energy, which the power normalisation would hide."""
    for fs in (12_800.0, 51_200.0):
        speed = np.full(int(fs * 60.0), 50.0)
        sig = _machine(speed, fs)
        noise = _disturb(sig, speed, fs, 0, seed=0, disturbance_types=("impulsive",)) - sig
        energy = noise**2
        share = energy.max() / energy.sum()
        assert share < 0.03, f"impulsive at fs={fs}: largest sample holds {share:.2%} of the noise energy"


def test_harmonic_disturbance_follows_the_interferer():
    """The comb runs at the interferer's speed, which rises here while the machine runs at constant speed."""
    fs = 2000.0
    n = int(fs * 20)
    speed = np.full(n, 20.0)
    sig = _machine(speed, fs)
    t = np.arange(n) / fs
    interferer = 22.0 + 6.0 * (t / t[-1])

    noise = _disturb(sig, speed, fs, 0, seed=0, disturbance_types=("harmonic",), interferer=interferer) - sig

    half = n // 2
    freqs = np.fft.rfftfreq(half, d=1 / fs)
    window = (freqs > 15.0) & (freqs < 35.0)
    first_half_peak = freqs[window][np.argmax(np.abs(np.fft.rfft(noise[:half]))[window])]
    second_half_peak = freqs[window][np.argmax(np.abs(np.fft.rfft(noise[half:]))[window])]
    assert first_half_peak == pytest.approx(_common.HARMONIC_IAS_RATIO * np.mean(interferer[:half]), rel=0.05)
    assert second_half_peak == pytest.approx(_common.HARMONIC_IAS_RATIO * np.mean(interferer[half:]), rel=0.05)


def test_harmonic_lines_sit_at_the_target_below_an_own_harmonic():
    """Each comb line carries 10**(-S/10) x the power an own harmonic would have at its frequency."""
    fs, target_snr_db = 2000.0, -3.0
    n = int(fs * 60)
    speed = np.full(n, 20.0)
    sig = _machine(speed, fs)
    ref = _reference(sig, speed, fs)
    noise = _disturb(sig, speed, fs, target_snr_db, seed=1, disturbance_types=("harmonic",)) - sig

    t = np.arange(n) / fs
    for k in (1, 2, 3):
        f_k = k * _common.HARMONIC_IAS_RATIO * 27.0
        line_power = np.abs(2 * np.mean(noise * np.exp(-2j * np.pi * f_k * t))) ** 2 / 2
        own_line_power = np.interp(f_k, ref.freqs, ref.density) * 20.0  # reference density x one own order
        assert 10 * np.log10(own_line_power / line_power) == pytest.approx(target_snr_db, abs=0.3), f"line {k}"


def test_comb_vanishes_while_the_interferer_stands_still():
    """A stopped interfering machine emits nothing, instead of piling all its lines up at 0 Hz."""
    fs = 2000.0
    n = int(fs * 20)
    speed = np.full(n, 20.0)
    sig = _machine(speed, fs)
    interferer = np.where(np.arange(n) < n // 2, 0.0, 25.0)

    comb = _disturb(sig, speed, fs, 0, seed=0, disturbance_types=("harmonic",), interferer=interferer) - sig

    assert np.sqrt(np.mean(comb[: n // 2] ** 2)) < 1e-9 * np.sqrt(np.mean(comb[n // 2 :] ** 2))


def test_interferer_does_not_follow_the_own_speed_profile():
    """Even when the other recording runs the same periodic profile, the chosen offset decorrelates it."""
    fs = 100.0
    t = np.arange(int(fs * 20)) / fs
    profile = 300 + 200 * np.abs(((t / 5.0) % 1) * 2 - 1)  # 5 s up/down triangle, as in the GFB runs

    interferer = _common.interferer_ias(profile, fs, profile, fs, np.random.default_rng(0))

    assert interferer.shape == profile.shape
    assert abs(np.corrcoef(interferer, profile)[0, 1]) < 0.3


def _write_ias_file(path, ias, vib_channels, fs=1000.0):
    # Write through the production writer so the fixtures track its layout.
    path.parent.mkdir(parents=True, exist_ok=True)
    _common.save_signals_hdf5({"IAS": ias, **vib_channels}, path, fs=fs, gear_ratio=1, ias_bandwidth_orders=20.0)


def _write_machine_dataset(ds, speeds, fs=2000.0, seconds=10.0):
    """One recording per (split, constant speed) with locked orders 1-3, as `_machine` builds them."""
    n = int(fs * seconds)
    for split, split_speeds in speeds.items():
        for i, speed in enumerate(split_speeds):
            ias = np.full(n, speed)
            _write_ias_file(ds / split / f"rec_{i}.hdf5", ias, {"Acc_x": _machine(ias, fs, seed=i)}, fs=fs)


def test_write_disturbed_test_sets_reproducible(tmp_path, monkeypatch):
    """Byte-identical whatever the order of the levels and the number of worker processes."""
    speeds = {"train": [18.0, 25.0], "test": [20.0, 30.0]}
    for run_dir, workers, levels in (("a", "1", [15, 0]), ("b", "2", [0, 15])):
        monkeypatch.setenv("IDENTIBENCH_WORKERS", workers)
        _write_machine_dataset(tmp_path / run_dir, speeds)
        _common.write_disturbed_test_sets(tmp_path / run_dir, vib_keys=["Acc_x"], settings=_SETTINGS, noise_levels=levels)

    for level in (15, 0):
        for variant in _common.DISTURBANCE_VARIANTS:
            for i in range(2):
                with (
                    h5py.File(tmp_path / "a" / f"test_disturbed_{level}dB_{variant}" / f"rec_{i}.hdf5") as fa,
                    h5py.File(tmp_path / "b" / f"test_disturbed_{level}dB_{variant}" / f"rec_{i}.hdf5") as fb,
                ):
                    np.testing.assert_array_equal(fa["Acc_x"][:], fb["Acc_x"][:])
                    np.testing.assert_array_equal(fa["IAS"][:], fb["IAS"][:])  # the target is untouched


def test_writer_runs_the_comb_at_another_recordings_speed(tmp_path, monkeypatch):
    """The harmonic disturbance of each file sits at the other file's speed, never at its own."""
    monkeypatch.setenv("IDENTIBENCH_WORKERS", "1")
    fs = 2000.0
    speeds = {"rec_0": 10.0, "rec_1": 37.0}
    _write_machine_dataset(tmp_path, {"train": [15.0], "test": list(speeds.values())}, fs=fs, seconds=20.0)

    _common.write_disturbed_test_sets(tmp_path, vib_keys=["Acc_x"], settings=_SETTINGS, noise_levels=[0])

    for stem, other in (("rec_0", "rec_1"), ("rec_1", "rec_0")):
        with (
            h5py.File(tmp_path / "test" / f"{stem}.hdf5") as clean,
            h5py.File(tmp_path / "test_disturbed_0dB_harmonic" / f"{stem}.hdf5") as disturbed,
        ):
            comb = disturbed["Acc_x"][:].astype(np.float64) - clean["Acc_x"][:]
        freqs = np.fft.rfftfreq(len(comb), 1 / fs)
        spectrum = np.abs(np.fft.rfft(comb)) ** 2

        def power_near(f, spectrum=spectrum, freqs=freqs):
            return spectrum[np.abs(freqs - f) < 0.5].sum()

        assert power_near(_common.HARMONIC_IAS_RATIO * speeds[other]) > 100 * power_near(
            _common.HARMONIC_IAS_RATIO * speeds[stem]
        ), stem


def test_disturbed_copies_differ_per_file_and_scale_per_level(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_WORKERS", "1")
    _write_machine_dataset(tmp_path, {"train": [25.0], "test": [20.0, 30.0]})
    _common.write_disturbed_test_sets(tmp_path, vib_keys=["Acc_x"], settings=_SETTINGS, noise_levels=[15, 7.5])

    def noise(level, stem):
        with (
            h5py.File(tmp_path / "test" / f"{stem}.hdf5") as clean,
            h5py.File(tmp_path / f"test_disturbed_{level}dB_combined" / f"{stem}.hdf5") as disturbed,
        ):
            return disturbed["Acc_x"][:].astype(np.float64) - clean["Acc_x"][:]

    assert not np.allclose(noise(15, "rec_0")[:1000], noise(15, "rec_1")[:1000])  # each file its own noise
    # the levels of a variant scale one realization (up to float32 storage)
    np.testing.assert_allclose(noise(15, "rec_0") * 10 ** (7.5 / 20), noise(7.5, "rec_0"), atol=1e-5)


def test_writer_stores_the_line_list_and_hits_the_level(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_WORKERS", "1")
    fs = 2000.0
    _write_machine_dataset(tmp_path, {"train": [18.0, 25.0], "test": [20.0, 30.0]}, fs=fs, seconds=30.0)
    _common.write_disturbed_test_sets(tmp_path, vib_keys=["Acc_x"], settings=_SETTINGS, noise_levels=[0])

    lines = harmonics.load_dataset_lines(tmp_path)
    assert lines.settings == _SETTINGS and lines.n_recordings == 4
    np.testing.assert_allclose(np.sort(lines.orders), [1.0, 2.0, 3.0])
    with (
        h5py.File(tmp_path / "test" / "rec_0.hdf5") as clean,
        h5py.File(tmp_path / "test_disturbed_0dB_gaussian" / "rec_0.hdf5") as disturbed,
    ):
        snr = harmonics.line_snr(clean["Acc_x"][:], disturbed["Acc_x"][:], clean["IAS"][:], fs, lines.orders, lines.settings)
    np.testing.assert_allclose(snr.long_term_snr_db(), 0.0, atol=1.0)


# Opt-in check on the real prepared datasets: set IDENTIBENCH_IAS_DATA_ROOT to the directory holding the
# prepared ball_bearing/, gas_foil_bearing/, ... (prepared with this version, i.e. with harmonic_lines.json).
_IAS_DATA_ROOT = os.environ.get("IDENTIBENCH_IAS_DATA_ROOT")
# strongest line of each dataset (all splits, all channels; measured 2026-10-19) and the channel checked
_REAL_STRONGEST_LINE = {
    "ball_bearing": (1.0, "Acc_x"),
    "gas_foil_bearing": (1.0, "Acc_x"),
    "parallel_gearbox": (95.0, "gearbox_vibration_x"),
    "planetary_gearbox": (21.6875, "Acc_Carrier"),
}


@pytest.mark.skipif(_IAS_DATA_ROOT is None, reason="set IDENTIBENCH_IAS_DATA_ROOT to the prepared IAS datasets")
@pytest.mark.parametrize("name", list(_REAL_STRONGEST_LINE))
def test_real_disturbed_sets_put_the_strongest_line_at_the_level(name):
    """The 0 dB gaussian copy of the first test file puts its strongest line near 0 dB (sweeps: up to +3)."""
    ds = Path(_IAS_DATA_ROOT) / name
    if not (ds / harmonics.HARMONIC_LINES_FILE).exists():
        pytest.skip(f"{ds} was not prepared with harmonic-referenced disturbances")
    lines = harmonics.load_dataset_lines(ds)
    order, channel = _REAL_STRONGEST_LINE[name]
    assert lines.strongest(1) == pytest.approx([order])
    clean_file = min((ds / "test").glob("*.hdf5"))
    with (
        h5py.File(clean_file) as clean,
        h5py.File(ds / "test_disturbed_0dB_gaussian" / clean_file.name) as disturbed,
    ):
        fs = float(clean.attrs["fs"])
        snr = harmonics.line_snr(clean[channel][:], disturbed[channel][:], clean["IAS"][:], fs, lines.orders, lines.settings)
    assert -1.0 < snr.long_term_snr_db()[snr.strongest_line()] < 3.0


# recordings whose reference had 80-140 dB holes before occupancy was decided from exact counts
_REAL_HOLE_CASES = {
    "ball_bearing": ("H-C-3", "Acc_x"),
    "parallel_gearbox": ("teeth_crack_H_torque_circulation_1000rpm_20Nm", "gearbox_vibration_x"),
}


@pytest.mark.skipif(_IAS_DATA_ROOT is None, reason="set IDENTIBENCH_IAS_DATA_ROOT to the prepared IAS datasets")
@pytest.mark.parametrize("name", list(_REAL_HOLE_CASES))
def test_real_references_have_no_holes(name):
    ds = Path(_IAS_DATA_ROOT) / name
    if not (ds / harmonics.HARMONIC_LINES_FILE).exists():
        pytest.skip(f"{ds} was not prepared with harmonic-referenced disturbances")
    lines = harmonics.load_dataset_lines(ds)
    stem, channel = _REAL_HOLE_CASES[name]
    with h5py.File(ds / "test" / f"{stem}.hdf5") as f:
        grid = harmonics.AngleGrid.from_ias(f["IAS"][:], float(f.attrs["fs"]), lines.settings)
        ref = harmonics.HarmonicReference.from_signal(f[channel][:].astype(np.float64), grid, lines.orders)
    assert ref.density.min() > 1e-6 * np.median(ref.density)


# ───────────────────────── HDF5 writing ─────────────────────────


def test_save_signals_hdf5_attrs_roundtrip(tmp_path):
    path = tmp_path / "rec.hdf5"
    _common.save_signals_hdf5(
        {"IAS": np.ones(10), "Acc_x": np.zeros(10)},
        path,
        fs=12800,
        gear_ratio=[95 / 29, 95, 1],
        ias_bandwidth_hz=4.03,
    )

    with h5py.File(path) as f:
        attrs = dict(f.attrs)
        assert f["IAS"].dtype == np.float32
    assert attrs["fs"] == 12800
    np.testing.assert_allclose(np.asarray(attrs["gear_ratio"]), [95 / 29, 95, 1])
    assert attrs["ias_bandwidth_hz"] == pytest.approx(4.03)


# ───────────────────────── test-set patterns + end-to-end ─────────────────────────


def test_ias_test_sets_patterns():
    sets = _common.ias_test_sets(ball_bearing_dataset)
    expected = ["basic", "ood"] + [
        f"disturbed_{level}dB_{variant}"
        for level in _common.DISTURBANCE_LEVELS
        for variant in _common.DISTURBANCE_VARIANTS
    ]
    assert list(sets) == expected
    assert sets["basic"] == [(ball_bearing_dataset, "test/*.hdf5")]
    assert sets["ood"] == [(ball_bearing_dataset, "test_ood/*.hdf5")]
    assert sets["disturbed_7.5dB_combined"] == [(ball_bearing_dataset, "test_disturbed_7.5dB_combined/*.hdf5")]
    assert sets["disturbed_0dB_harmonic"] == [(ball_bearing_dataset, "test_disturbed_0dB_harmonic/*.hdf5")]
    # A dataset without an ood condition simply does not declare the set.
    assert "ood" not in _common.ias_test_sets(ball_bearing_dataset, ood=False)


_SYNTH_SAMPLES = 4000  # 4 s at fs=1000: 20 revolutions at 5 Hz, enough frame pairs for the disturbance writer


def _build_synthetic_ias_dataset(ds, with_ood=True):
    """A small ball-bearing-shaped dataset with known IAS targets, marked prepared."""
    speed = np.full(_SYNTH_SAMPLES, 5.0)
    for split, n_files in {"train": 1, "valid": 1, "test": 2}.items():
        for i in range(n_files):
            _write_ias_file(ds / split / f"{split}_{i}.hdf5", speed, {"Acc_x": _machine(speed, 1000.0, seed=i)})
    if with_ood:
        _write_ias_file(ds / "test_ood" / "ood_0.hdf5", np.full(_SYNTH_SAMPLES, 9.0), {"Acc_x": _machine(speed, 1000.0)})
    _common.write_disturbed_test_sets(ds, vib_keys=["Acc_x"], settings=HarmonicSettings(segment_revs=4), noise_levels=[15])
    # Adopt the synthetic data as a prepared cache. This must carry the dataset's *current*
    # version, or `ensure_exists` treats the directory as stale and downloads the real 200 kHz
    # archive instead of using the fixture.
    (ds / ".prepared").write_text(ball_bearing_dataset.version)


def _narrowed_test_sets():
    """The synthetic dataset only materializes one disturbance level."""
    return {
        name: patterns
        for name, patterns in _common.ias_test_sets(ball_bearing_dataset).items()
        if name in ("basic", "ood", "disturbed_15dB_combined")
    }


def test_run_benchmark_on_synthetic_ias_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    _build_synthetic_ias_dataset(tmp_path / "ball_bearing")
    # 4000-sample files at fs=1000 -> a 0.05 s window is 50 samples (80 windows/file).
    spec = dataclasses.replace(
        BenchmarkBallBearing_Estimation,
        task=WindowedEstimation(window_sec=0.05),
        test_sets=_narrowed_test_sets(),
    )

    def build_model(context):
        def model(u, y_init, attrs):
            assert y_init.shape == (0, 1)  # estimation -> empty warm-up, one window of u
            return np.zeros((len(u), 1))

        return model

    result = idb.run_benchmark(spec, build_model, seed=0)

    # Headline = basic pooled MAE; the zero-model's per-window error is the constant IAS.
    assert result["metric_name"] == "mae"
    assert result["metric_score"] == pytest.approx(5.0, abs=1e-5)
    assert set(result["test_sets"]) == {"basic", "ood", "disturbed_15dB_combined"}
    assert result["test_sets"]["ood"]["mae"] == pytest.approx(9.0, abs=1e-5)
    assert result["test_sets"]["disturbed_15dB_combined"]["mae"] == pytest.approx(5.0, abs=1e-5)
    # The pooled statistics ride alongside the headline MAE on every set.
    basic = result["test_sets"]["basic"]
    assert set(basic) == {"mae", "medae", "std", "max"}
    assert basic["std"] == pytest.approx(0.0, abs=1e-5)  # constant IAS -> identical errors
    assert basic["max"] == pytest.approx(5.0, abs=1e-5)


def test_run_simulation_benchmark_on_synthetic_ias_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    _build_synthetic_ias_dataset(tmp_path / "ball_bearing")  # 4000-sample files at fs=1000, basic IAS = 5.0 Hz
    spec = dataclasses.replace(BenchmarkBallBearing_Simulation, test_sets=_narrowed_test_sets())

    def build_model(context):
        def model(u, y_init, attrs):
            # Free-run: the model gets the WHOLE recording at once with an empty
            # warm-up and returns one IAS estimate per sample (a real model would slide a
            # window over u; here a trivial zero output).
            assert y_init.shape == (0, 1)
            assert len(u) == _SYNTH_SAMPLES
            return np.zeros((len(u), 1))

        return model

    result = idb.run_benchmark(spec, build_model, seed=0)

    # Per-sample MAE in Hz over the full sequence; only the headline `mae` is reported
    # (no pooled medae/std/max — those are WindowedEstimation-specific).
    assert result["benchmark_type"] == "Simulation"
    assert result["metric_name"] == "mae"
    assert result["metric_score"] == pytest.approx(5.0, abs=1e-5)  # |0 - 5| per sample
    assert set(result["test_sets"]) == {"basic", "ood", "disturbed_15dB_combined"}
    assert set(result["test_sets"]["basic"]) == {"mae"}
    assert result["test_sets"]["ood"]["mae"] == pytest.approx(9.0, abs=1e-5)


def test_run_gridwise_benchmark_on_synthetic_ias_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    _build_synthetic_ias_dataset(tmp_path / "ball_bearing")
    # 4000-sample files at fs=1000 -> a 0.05 s window / 0.01 s step gives 396 query points/file.
    spec = dataclasses.replace(
        BenchmarkBallBearing_Estimation,
        task=GridwiseEstimation(window_sec=0.05, step_sec=0.01, min_targ=1.0),
        test_sets=_narrowed_test_sets(),
    )

    def build_model(context):
        def model(u, y_init, attrs):
            assert y_init.shape == (0, 1)
            t_query = attrs["t_query"]
            # Returns a plain Python list, not an ndarray -- exercises the list-coercion fix.
            return [5.02] * len(t_query)

        return model

    result = idb.run_benchmark(spec, build_model, seed=0)

    # Headline = basic acceptable-point percentage: 5.02 vs 5 Hz is 0.004 orders off (< 0.01),
    # vs the 9 Hz ood set 0.44 orders off.
    assert result["metric_name"] == "rel_acceptable_perc"
    assert result["metric_score"] == pytest.approx(100.0)
    assert set(result["test_sets"]) == {"basic", "ood", "disturbed_15dB_combined"}
    assert result["test_sets"]["ood"]["rel_acceptable_perc"] == pytest.approx(0.0)
    assert result["test_sets"]["ood"]["mae"] == pytest.approx(3.98, abs=1e-5)
    basic = result["test_sets"]["basic"]
    assert set(basic) == {"mae", "medae", "std", "max", "rel_acceptable_perc"}
    assert basic["mae"] == pytest.approx(0.02, abs=1e-5)
    assert basic["std"] == pytest.approx(0.0, abs=1e-5)  # constant IAS -> identical errors


def test_gridwise_estimation_rejects_sets_without_scored_targets(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    _build_synthetic_ias_dataset(tmp_path / "ball_bearing")
    # A 6 Hz floor leaves nothing of the 5 Hz basic set to score: fail loudly, not NaN.
    spec = dataclasses.replace(
        BenchmarkBallBearing_Estimation,
        task=GridwiseEstimation(window_sec=0.05, step_sec=0.01, min_targ=6.0),
        test_sets=_narrowed_test_sets(),
    )

    def build_model(context):
        return lambda u, y_init, attrs: np.full(len(attrs["t_query"]), 5.0)

    with pytest.raises(ValueError, match="no target above min_targ"):
        idb.run_benchmark(spec, build_model, seed=0)


def test_gridwise_estimation_rejects_wrong_shaped_predictions(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    _build_synthetic_ias_dataset(tmp_path / "ball_bearing")
    spec = dataclasses.replace(
        BenchmarkBallBearing_Estimation,
        task=GridwiseEstimation(window_sec=0.05, step_sec=0.01),
        test_sets=_narrowed_test_sets(),
    )

    def build_model(context):
        # Wrong length: one prediction regardless of how many query points were asked for.
        return lambda u, y_init, attrs: np.zeros(1)

    with pytest.raises(ValueError, match="expected"):
        idb.run_benchmark(spec, build_model, seed=0)


def test_gridwise_estimation_rejects_multi_output_specs(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    _build_synthetic_ias_dataset(tmp_path / "ball_bearing")
    spec = dataclasses.replace(
        BenchmarkBallBearing_Estimation,
        task=GridwiseEstimation(window_sec=0.05, step_sec=0.01),
        test_sets=_narrowed_test_sets(),
        y_cols=["IAS", "Extra"],
    )

    def build_model(context):
        return lambda u, y_init, attrs: np.zeros(len(attrs["t_query"]))

    with pytest.raises(ValueError, match="one y_col"):
        idb.run_benchmark(spec, build_model, seed=0)


def test_missing_ood_dir_fails_loudly(tmp_path, monkeypatch):
    """Declaring `ood` against a dataset without the condition must raise, not skip."""
    monkeypatch.setenv("IDENTIBENCH_DATA_ROOT", str(tmp_path))
    _build_synthetic_ias_dataset(tmp_path / "ball_bearing", with_ood=False)
    spec = dataclasses.replace(
        BenchmarkBallBearing_Estimation,
        task=WindowedEstimation(window_sec=0.05),
        test_sets=_narrowed_test_sets(),  # declares ood
    )

    def build_model(context):
        return lambda u, y_init, attrs: np.zeros((len(u), 1))

    with pytest.raises(FileNotFoundError, match="test_ood"):
        idb.run_benchmark(spec, build_model, seed=0)


# ───────────────────────── registration ─────────────────────────


def test_registration():
    for key in (
        "BallBearing_Estimation",
        "ParallelGearbox_Estimation",
        "PlanetaryGearbox_Estimation",
        "GasFoilBearing_Estimation",
    ):
        assert key in idb.simulation_benchmarks
        assert key in idb.ias_benchmarks
        spec = idb.ias_benchmarks[key]
        assert isinstance(spec.task, WindowedEstimation)
        assert spec.task.window_sec > 0
        assert next(iter(spec.test_sets)) == "basic"  # built-in tasks headline the first set
        assert spec.y_cols == ["IAS"]
    # Each dataset also has a dense free-run Simulation sibling sharing the same data.
    for key in (
        "BallBearing_Simulation",
        "ParallelGearbox_Simulation",
        "PlanetaryGearbox_Simulation",
        "GasFoilBearing_Simulation",
    ):
        assert key in idb.simulation_benchmarks
        assert key in idb.ias_benchmarks
        sim = idb.ias_benchmarks[key]
        est = idb.ias_benchmarks[key.replace("_Simulation", "_Estimation")]
        assert isinstance(sim.task, idb.Simulation)
        assert sim.task.init_window == 0  # estimation: no output history fed as warm-up
        assert sim.task.metric.__name__ == "mae"
        assert next(iter(sim.test_sets)) == "basic"
        # The two variants differ only in the task; data binding is identical.
        assert sim.datasets == est.datasets
        assert sim.test_sets == est.test_sets
        assert sim.u_cols == est.u_cols
        assert sim.y_cols == est.y_cols == ["IAS"]
    # Each dataset also has a GridwiseEstimation sibling sharing the same data. Its step_sec is
    # per-dataset and derived from the label bandwidth, so pin the values (see ias/__init__).
    for key, expected_step in (
        ("BallBearing_GridwiseEstimation", 0.003),
        ("ParallelGearbox_GridwiseEstimation", 0.1),
        ("PlanetaryGearbox_GridwiseEstimation", 0.003),
        ("GasFoilBearing_GridwiseEstimation", 0.003),
    ):
        assert key in idb.simulation_benchmarks
        assert key in idb.ias_benchmarks
        spec = idb.ias_benchmarks[key]
        assert isinstance(spec.task, GridwiseEstimation)
        assert spec.task.window_sec == 3.0  # uniform context guarantee, unrelated to resolution
        assert spec.task.step_sec == expected_step
        if key.startswith("GasFoilBearing"):
            assert spec.task.min_targ == 10.0  # label uncertainty above the 1 % tolerance below 10 Hz
        else:
            assert spec.task.min_targ == 1.0  # 60 rpm: only standstill drops out of the headline
        assert next(iter(spec.test_sets)) == "basic"  # built-in tasks headline the first set
        assert spec.y_cols == ["IAS"]


def test_gridwise_step_matches_declared_label_bandwidth():
    """Each grid is Nyquist for its dataset's retained band -- except the documented planetary cap."""
    from identibench.datasets.ias import ball_bearing, gas_foil_bearing, parallel_gearbox, planetary_gearbox

    for module, key in (
        (ball_bearing, "BallBearing_GridwiseEstimation"),
        (parallel_gearbox, "ParallelGearbox_GridwiseEstimation"),
        (gas_foil_bearing, "GasFoilBearing_GridwiseEstimation"),
    ):
        step = idb.ias_benchmarks[key].task.step_sec
        nyquist = 1 / (2 * module._IAS_BANDWIDTH_HZ)
        assert step <= nyquist, f"{key}: step {step} exceeds Nyquist {nyquist}"

    # The planetary is knowingly coarser; assert the cap so it cannot drift unnoticed in either
    # direction -- tightening it silently would multiply evaluation cost by over 100x.
    step = idb.ias_benchmarks["PlanetaryGearbox_GridwiseEstimation"].task.step_sec
    nyquist = 1 / (2 * planetary_gearbox._IAS_BANDWIDTH_HZ)
    assert nyquist == pytest.approx(0.00131, abs=1e-5)  # 381.3 Hz, i.e. _CUTOFF_ORDER = 13
    assert step == 0.003 > nyquist
    for dataset_id in ("ball_bearing", "parallel_gearbox", "planetary_gearbox", "gas_foil_bearing"):
        assert dataset_id in idb.datasets.all_datasets
