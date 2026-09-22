"""Planetary gearbox IAS estimation dataset (figshare 28992879).

The IAS label is reconstructed from the **zebra tape on the sun shaft** (``Channel_5_Data``),
with the 1-pulse-per-revolution pickup on the planet carrier (``Channel_6_Data``) as the absolute
angle reference. Four properties of the raw tacho data shape the pipeline:

* The tape's stripes are irregular and not every stripe registers, so there is no fixed
  pulses-per-revolution. A bootstrap stripe template is estimated from the 1PR-referenced phase of
  every edge, pooled across all recordings; it fixes which physical stripe is which in every file.
* Interpolating the carrier reference between its pulses (5.77 sun revolutions apart) is only good
  to ~0.2 of a stripe spacing, and far worse through speed transients, so it cannot label single
  pulses. Stripes are therefore *counted* edge to edge from the template spacing and the tracked
  speed, and the 1PR is used only at its own pulse times, where it is exact, to correct the count.
* Where the sensor switches on each stripe edge moves whenever the rig is reassembled (a stripe's
  apparent width changes by up to ~0.04 of a spacing), which a single pooled template cannot follow
  and which shows up as a comb of integer-order lines in the label. The stripe positions used for
  the reconstruction are therefore self-calibrated from each recording's own edge timing (pooled
  only where that demonstrably helps, ``_TEMPLATE_GROUPS`` / ``_TEMPLATE_BORROW``), and taken at
  stripe centres (midpoint of both edges), which cancels those width changes.
* The 1PR channel lags the zebra channel by ~0.45 ms -- a speed-proportional phase error of up to a
  whole stripe at the highest speeds -- which is compensated before the reference is used.

Self-calibration assumes uniform rotation over each revolution, so it also removes any genuinely
sun-synchronous (integer-order) speed variation. Checked against the 1PR at the 13 sun phases it
samples, what that removes is at most ~0.1 % of speed; non-integer content such as the gear mesh
(10.75 sun orders) and the sun's rotation relative to the carrier (0.827 orders) is kept.
"""

__all__ = [
    "planetary_gearbox_dataset",
    "dl_planetary_gearbox",
    "BenchmarkPlanetaryGearbox_Estimation",
    "BenchmarkPlanetaryGearbox_Simulation",
]

import math
import tempfile
from pathlib import Path

import numpy as np
import scipy.io
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from tqdm import tqdm

from ...benchmark import BenchmarkSpec, Simulation, WindowedEstimation, GridwiseEstimation
from ...dataset import Dataset
from ...metrics import mae
from ._common import (
    DatasetInfo,
    download_and_unpack,
    order_domain_lowpass,
    rising_edge_times,
    save_signals_hdf5,
    ias_test_sets,
    write_disturbed_test_sets,
)

_INFO = DatasetInfo(
    name="Planetary_Gearbox",
    zip_url="https://ndownloader.figshare.com/files/28992879",
    download_headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://figshare.com/articles/dataset/Planetary_gearbox_vibration_data/13476525?file=28992879",
    },
)

# Crack severities forming the out-of-distribution wear set; G1_P4/G2_P1 are the
# basic test recordings and G1_P3/G2_P0 the validation recordings (verbatim split).
_TEST_WEAR_TYPES = ["P5", "P6", "P7"]
_TEST_BASIC_TYPES = ["G1_P4", "G2_P1"]
_VALID_TYPES = ["G1_P3", "G2_P0"]

# Bonfiglioli 300-L: 13 sun teeth, 24 planet, 62 ring. With the ring fixed the sun turns
# (13 + 62) / 13 times per carrier revolution, which is what converts the carrier-mounted
# 1PR reference into sun-shaft angle -- and hence what makes the label sun-referenced.
_SUN_TEETH, _RING_TEETH = 13, 62
_SUN_PER_CARRIER_REV = (_SUN_TEETH + _RING_TEETH) / _SUN_TEETH

# Order-domain cutoff for the reconstructed IAS, in orders of the sun shaft. The pooled
# template resolves 76 stripes/rev, so the order-domain Nyquist is 38.
_CUTOFF_ORDER = 11.32

# Highest frequency the label retains: IAS_max * cutoff_order on the sun shaft. 29.33 Hz is the
# peak across all 15 recordings (medians run 4.5-17.6 Hz); the 39.90 Hz peak of version 3 was a
# mislabelled-pulse spike, not a real speed. This is by far the widest band of the four IAS
# datasets -- the zebra tape resolves ~76x more per revolution than the 1PR pickup it replaced --
# which is what drives both the evaluation grid and the model sample-rate floor.
_IAS_BANDWIDTH_HZ = 29.33 * _CUTOFF_ORDER  # 440.0 Hz

# The 1PR channel lags the zebra channel by this much: 0.35-0.50 ms on every recording, the same
# for both flanks of the pickup's dip. Left in, it is a speed-proportional sun-phase error of up
# to a whole stripe spacing (at the 29 Hz peak), which shifts with every speed step of a recording.
_REF_DELAY_S = 0.45e-3

# Stripe-template estimation: phase histogram resolution, and how many times the per-file
# histograms are re-aligned against the running pooled reference before the peaks are fitted.
_N_PHASE_BINS = 4000
_N_POOL_ITER = 2

# `count_zebra_stripes`: an edge further than `_COUNT_REJECT` of a spacing from the stripe the
# tracked speed predicts is rejected as spurious, and the tracked speed is an exponential average of
# accepted steps with weight `_COUNT_ALPHA`.
_COUNT_REJECT = 0.3
_COUNT_ALPHA = 0.3
# A step predicted to span more than this many stripes (missed stripes, rejected edges) takes its
# speed from timed revolutions instead of the tracked speed.
_COUNT_LONG_STEP = 6
# At each 1PR pulse the counted and the referenced stripe position must agree to within
# `_ANCHOR_TOL` of a stripe once the whole-stripe difference is removed, or that carrier revolution
# is dropped as unresolvable. A recording needing that for more than `_ANCHOR_MAX_DROPPED` of its
# carrier revolutions is rejected outright rather than shipped with a doubtful label (the real
# recordings need it for 1-4 of their 750-2500 revolutions during the run-up, and G1_P0 for 2 more
# at its near-stop).
_ANCHOR_TOL = 0.35
_ANCHOR_MAX_DROPPED = 0.1

# A missed or spurious 1PR pulse renumbers every later one, after which the count disagrees with most
# pulses (by a fraction of a stripe that depends on which of the 13 sun phases a pulse falls on),
# whereas a counting slip is corrected at the first pulse and agrees again afterwards. More than
# `_ANCHOR_BURST` disagreements within 13 consecutive pulses, once the count had agreed, therefore
# fail the recording (the real recordings have at most 2, at G1_P0's near-stop).
_ANCHOR_BURST = 4

# `drop_zebra_outliers`: a match whose local instantaneous rate departs from the running
# median over this many neighbours by more than this factor is discarded. The window must be odd.
_OUTLIER_WINDOW = 101
_OUTLIER_RATIO = 1.35
_OUTLIER_MAX_ITER = 5

# Pooling of the stripe calibration. Each reassembly of the rig (G1 and G2 recordings are
# interleaved in time) moves where the sensor switches on each stripe, so recordings differ by
# 0.004-0.04 of a spacing, while a single recording's calibration repeats to 0.0015-0.01 between its
# two halves (0.026 for the short, slow G1_P1_slow). Pooling therefore only pays for recordings that
# agree to within ~1.5x that repeatability, and was kept only where calibrating on one half and
# scoring the integer-order power (the signature of stripe-position error) on the other half improved
# for every member: the groups below are calibrated together, G1_P1_slow additionally uses G1_P1's
# edges (one way -- the reverse makes G1_P1 worse), and every other recording is calibrated alone.
_TEMPLATE_GROUPS = (
    ("G1_P3_38400Hz", "G1_P4_38400Hz", "G1_P5_38400Hz"),
    ("G1_P6_38400Hz", "G1_P7_38400Hz"),
    ("G2_P4_38400Hz", "G2_P5_38400Hz"),
)
_TEMPLATE_BORROW = {"G1_P1_slow_38400Hz": ("G1_P1_38400Hz",)}

# Self-calibration uses edge intervals spanning at most this many stripes.
_CAL_MAX_STEP = 12

# A dark bar whose width departs from that stripe's usual width by more than this many stripe
# spacings is not used as a stripe centre. Mostly these are two neighbouring bars read as one (the
# light gap between them did not register, so the falling edge belongs to the earlier bar and the
# midpoint lands between the two stripes: ~1 % of G2_P0's bars, deviating by about a whole spacing);
# a few are bars that registered much narrower than usual. The normal scatter is 0.013-0.02.
_BAR_WIDTH_TOL = 0.1


def _parse_fs(mat_data: dict) -> float:
    """Per-file sampling rate from the .MAT header (verbatim nested-index + decimal-comma parse)."""
    return float(
        np.fromstring(mat_data["File_Header"]["SampleFrequency"][0][0][0].replace(",", "."), sep=";").squeeze()
    )


def reference_events(signal: np.ndarray, fs: float) -> np.ndarray:
    """Times (s) of the carrier 1PR pulses, on the zebra channel's time base.

    The pickup's pulse is a dip ~36 deg of carrier rotation wide at every speed, with slow,
    clipped flanks. Each pulse is timed at the midpoint of its falling and rising flank crossings
    and moved earlier by ``_REF_DELAY_S``. Crossings of one threshold alternate, so every rising
    flank pairs with the falling flank before it, except a first one cut off by the start of the
    recording, which is skipped. A dip missed altogether, or a spurious one, cannot be seen here;
    :func:`count_zebra_stripes` catches it, as the count disagreeing with the pulses from then on.
    """
    signal = np.asarray(signal, dtype=float)
    rise = rising_edge_times(signal, fs)
    fall = rising_edge_times(-signal, fs)
    j = np.searchsorted(fall, rise) - 1
    return (fall[j[j >= 0]] + rise[j >= 0]) / 2 - _REF_DELAY_S


# ───────────────────────── bootstrap stripe template ─────────────────────────


def _zebra_phase(t_ref_pulses: np.ndarray, t_zebra_pulses: np.ndarray) -> tuple[np.ndarray, float]:
    """Sun-shaft phase (fraction of a revolution) of each zebra edge, plus the revolutions spanned.

    Cumulative 1PR pulse count is a sample of carrier angle at each reference edge; monotone
    PCHIP through it gives carrier angle at arbitrary times, and scaling by
    ``_SUN_PER_CARRIER_REV`` converts that to sun angle. The first and last ten reference
    pulses are excluded so the interpolant is never extrapolated.
    """
    theta_ref = PchipInterpolator(t_ref_pulses, np.arange(len(t_ref_pulses)))
    mask = (t_zebra_pulses > t_ref_pulses[10]) & (t_zebra_pulses < t_ref_pulses[-10])
    theta_zebra = _SUN_PER_CARRIER_REV * theta_ref(t_zebra_pulses[mask])
    return theta_zebra % 1.0, float(theta_zebra[-1] - theta_zebra[0])


def _phase_histogram(phase: np.ndarray, n_bins: int = _N_PHASE_BINS) -> np.ndarray:
    return np.histogram(phase, bins=n_bins, range=(0, 1))[0].astype(float)


def _circular_offset(reference_hist: np.ndarray, hist: np.ndarray) -> int:
    """Bin shift of ``hist`` that best aligns it with ``reference_hist``, circularly."""
    a = reference_hist - reference_hist.mean()
    b = hist - hist.mean()
    return int(np.argmax(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))).real))


def _fit_template_from_phase(
    phase: np.ndarray, n_revs: float, n_bins: int = _N_PHASE_BINS
) -> tuple[np.ndarray, np.ndarray]:
    """Stripe angles (in revolutions) and their detection counts, from pooled phases.

    Peaks are found on a three-fold tiling of the histogram so a stripe sitting near phase 0
    is not split by the wrap, then each peak is refined to the median phase of the detections
    around it rather than the bin centre.

    Note the resolution limit: peaks are required to be at least half the *mean* stripe
    spacing apart, so two stripes printed closer together than that merge into one. On the
    real tape this does not bite (the pooled fit resolves all 76), but it caps how uneven a
    tape this can characterize.
    """
    hist = _phase_histogram(phase, n_bins)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    n_guess = int(round(len(phase) / n_revs))
    min_distance = max(int(n_bins / n_guess * 0.5), 1)

    tiled = np.concatenate([hist, hist, hist])
    peak_idx_tiled, _ = find_peaks(tiled, distance=min_distance, prominence=hist.mean() * 0.3)
    peak_idx = sorted({(p - n_bins) % n_bins for p in peak_idx_tiled if n_bins <= p < 2 * n_bins})

    template, counts = [], []
    half_width = 0.5 / n_guess * 0.6
    for pk in peak_idx:
        offsets = ((phase - bin_centers[pk] + 0.5) % 1.0) - 0.5
        nearby = offsets[np.abs(offsets) < half_width]
        if len(nearby) == 0:
            continue  # a peak with no detections inside the half-width would give a NaN angle
        # Take the median in the peak's local frame so detections on either side of
        # phase zero remain neighbours; wrap back only after estimating the centre.
        template.append((bin_centers[pk] + np.median(nearby)) % 1.0)
        counts.append(len(nearby))
    order = np.argsort(template)
    return np.array(template)[order], np.array(counts)[order]


def pool_zebra_templates(
    file_phases: list[np.ndarray], file_revs: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool phase-aligned zebra detections across files into one stripe template.

    Files are circularly cross-correlated against a running reference histogram and aligned
    before pooling: small rotational offsets between recording sessions (from
    disassembly/reassembly between test phases) would otherwise blur or split real stripes
    and lose stripe count as more files are added. Pooling every recording roughly doubles
    the evidence behind the template versus a single file and measurably improves the weakest
    stripes; single-file templates from the lower-quality recordings are incomplete on their own.

    The result is the *bootstrap* template: accurate to a few hundredths of a spacing, which is
    ample for telling stripes apart and counting them, but not for the reconstruction itself --
    that uses the per-group calibration of :func:`calibrate_zebra`.

    Returns:
        ``(template, counts, pooled_reference_hist)`` -- stripe angles in revolutions, the
        detections behind each, and the pooled histogram later files are aligned against.
    """
    hists = [_phase_histogram(p) for p in file_phases]
    reference = hists[int(np.argmax(file_revs))]
    pooled = None
    for _ in range(_N_POOL_ITER):
        offsets = [_circular_offset(reference, h) for h in hists]
        aligned = [(p + o / _N_PHASE_BINS) % 1.0 for p, o in zip(file_phases, offsets)]
        pooled = np.concatenate(aligned)
        reference = _phase_histogram(pooled)
    template, counts = _fit_template_from_phase(pooled, sum(file_revs))
    return template, counts, reference


def zebra_phase_offset(phase: np.ndarray, pooled_reference_hist: np.ndarray) -> float:
    """This file's rotational offset (in revolutions) against the pooled template."""
    return _circular_offset(pooled_reference_hist, _phase_histogram(phase)) / _N_PHASE_BINS


# ───────────────────────── counting and calibration ─────────────────────────


def _theta(idx: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Cumulative shaft angle, in revolutions, of global stripe index ``idx``."""
    n = len(template)
    return (idx // n) + template[idx % n]


def count_zebra_stripes(
    t_zebra: np.ndarray, t_ref: np.ndarray, template: np.ndarray, phase_offset: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Assign zebra edges global stripe indices by counting, anchored at the 1PR pulses.

    Each edge is placed relative to the previous accepted one: the tracked speed predicts the angle
    it should sit at, it takes the nearest stripe ahead (skipping any missed in between), and it is
    rejected as spurious if no stripe lies within ``_COUNT_REJECT`` of a spacing of the prediction.
    A step expected to span more than ``_COUNT_LONG_STEP`` stripes takes its speed from the last full
    sun revolution (or, until one is timed, the current carrier revolution) instead, since a small
    error in the tracked speed builds up over a long step. Counting can still slip through a
    transient, so at every 1PR pulse -- where the reference, unlike its interpolation between pulses,
    is exact -- the count is compared with the stripe position the reference puts there (sun angle
    ``_SUN_PER_CARRIER_REV * pulse_number + phase_offset``). If they disagree, the carrier revolution's edges are dropped, since the slip
    could lie anywhere in it; a whole-stripe disagreement is corrected, and the speed re-seeded from
    the reference.

    Labelling each pulse independently from the interpolated reference instead (the earlier
    method) mislabels a fraction of a percent of pulses even on clean tape -- the interpolation is
    good to only ~0.2 of a spacing, far worse through speed steps -- and those mislabels became a
    broadband hump over the first ~20 orders of the label's spectrum.

    Returns:
        ``(t, idx, n_dropped)`` -- the accepted edge times, their strictly increasing global stripe
        indices (angle ``_theta(idx, template)``), and how many carrier revolutions were dropped.

    Raises:
        RuntimeError: If more than ``_ANCHOR_MAX_DROPPED`` of the carrier revolutions had to be
            dropped, or the count keeps disagreeing with the reference (``_ANCHOR_BURST``) -- the mark
            of a missed or spurious 1PR pulse, which renumbers every later one.
    """
    n = len(template)
    tpl = template.tolist()
    ref = t_ref.tolist()

    def theta(k):
        return k // n + tpl[k % n]

    def frac_index(a, k):
        """Fractional stripe index of angle ``a``, searched outward from index ``k``."""
        while theta(k) > a:
            k -= 1
        while theta(k + 1) <= a:
            k += 1
        return k + (a - theta(k)) / (theta(k + 1) - theta(k))

    edges = t_zebra[(t_zebra > ref[0]) & (t_zebra < ref[-1])].tolist()
    rate = _SUN_PER_CARRIER_REV / (ref[1] - ref[0])  # sun rev/s
    start = phase_offset + rate * (edges[0] - ref[0])
    k_prev = round(frac_index(start, int(np.floor(start * n))))
    t_prev, a_prev = edges[0], theta(k_prev)
    times, indices = [t_prev], [k_prev]
    # The latest accepted edge on each stripe, which times the full revolution ending at an edge.
    seen_k, seen_t = [None] * n, [0.0] * n
    seen_k[k_prev % n], seen_t[k_prev % n] = k_prev, t_prev
    rev_rate = None  # sun rev/s over the revolution ending at the previous accepted edge, if timed
    dropped = set()
    faults = []  # pulses at which a count that had already agreed with the reference disagreed
    in_sync = False
    e = 1
    for t in edges[1:] + [math.inf]:  # the sentinel makes the loop check the final pulse as well
        while e < len(ref) and ref[e] < t:  # passing 1PR pulse e
            a_ref = _SUN_PER_CARRIER_REV * e + phase_offset
            dk = frac_index(a_prev + rate * (ref[e] - t_prev), k_prev) - frac_index(a_ref, k_prev)
            shift = round(dk)
            failed = shift != 0 or abs(dk - shift) >= _ANCHOR_TOL
            if failed:
                dropped.add(e)
                if in_sync:
                    faults.append(e)
                if abs(dk - shift) < _ANCHOR_TOL:  # a whole-stripe slip: renumber, keep the phase
                    k_prev -= shift
                    a_prev = theta(k_prev)
                # A wrong speed can be self-consistent (half the true one, reading every second edge as
                # the next stripe), so the speed is taken from the reference as well.
                nxt = min(e + 1, len(ref) - 1)
                rate = _SUN_PER_CARRIER_REV * (nxt - e + 1) / (ref[nxt] - ref[e - 1])
                seen_k, rev_rate = [None] * n, None  # stored indices may be in the old numbering
            in_sync |= not failed
            e += 1
        if t == math.inf:
            break
        dt = t - t_prev
        # Over a long step a small error in the tracked speed -- which is measured on the bootstrap
        # template -- builds up past the rejection tolerance, so long steps take their speed from how
        # long the last sun revolution took or, until one is timed, the current carrier revolution.
        if rate * dt * n > _COUNT_LONG_STEP:
            pred = a_prev + (rev_rate or _SUN_PER_CARRIER_REV / (ref[e] - ref[e - 1])) * dt
        else:
            pred = a_prev + rate * dt
        k = max(round(frac_index(pred, k_prev)), k_prev + 1)
        a = theta(k)
        if abs(a - pred) > _COUNT_REJECT * (a - theta(k - 1)):
            continue
        rate += _COUNT_ALPHA * ((a - a_prev) / dt - rate)
        slot = k % n
        rev_rate = 1 / (t - seen_t[slot]) if seen_k[slot] == k - n else None
        seen_k[slot], seen_t[slot] = k, t
        t_prev, k_prev, a_prev = t, k, a
        times.append(t)
        indices.append(k)

    if len(dropped) > _ANCHOR_MAX_DROPPED * (len(ref) - 1):
        raise RuntimeError(
            f"stripe count disagreed with the 1PR reference in {len(dropped)} of {len(ref) - 1} carrier revolutions"
        )
    burst = [a for a, b in zip(faults, faults[_ANCHOR_BURST:]) if b - a < _SUN_TEETH]
    if burst:
        raise RuntimeError(
            f"stripe count disagreed with the 1PR reference at {_ANCHOR_BURST + 1} of {_SUN_TEETH} consecutive pulses "
            f"from pulse {burst[0]} (t={ref[burst[0]]:.3f} s) -- missed or spurious 1PR pulse?"
        )
    times, indices = np.array(times), np.array(indices, dtype=np.int64)
    keep = ~np.isin(np.searchsorted(t_ref, times), list(dropped))
    times, indices = times[keep], indices[keep]
    # A correction can move the count back below indices already kept from an earlier revolution
    # only if it exceeds a whole revolution's worth of stripes; guard against it all the same.
    keep = np.ones(len(indices), dtype=bool)
    keep[1:] = indices[1:] > np.maximum.accumulate(indices)[:-1]
    return times[keep], indices[keep], len(dropped)


def drop_zebra_outliers(
    t_matched: np.ndarray, global_index: np.ndarray, template: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove labelled edges whose local instantaneous rate is inconsistent with their neighbours.

    A safety net behind :func:`count_zebra_stripes`, which on the real recordings leaves (almost)
    nothing for it to remove: a labelling error that got past the 1PR check would show up here as
    a local rate jump, and is dropped and left as an ordinary gap for the reconstruction to
    interpolate over, exactly like a plain missed detection.
    """
    t, idx = t_matched, global_index
    for _ in range(_OUTLIER_MAX_ITER):
        if len(t) < _OUTLIER_WINDOW + 1:
            break
        inst_rate = np.diff(_theta(idx, template)) / np.diff(t)
        ratio = inst_rate / median_filter(inst_rate, size=_OUTLIER_WINDOW)
        bad = (ratio > _OUTLIER_RATIO) | (ratio < 1 / _OUTLIER_RATIO)
        if not bad.any():
            break
        keep = np.ones(len(t), dtype=bool)
        keep[1:][bad] = False
        t, idx = t[keep], idx[keep]
    return t, idx


def calibrate_zebra(
    recordings: dict[str, dict[str, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Label every recording's stripes, and calibrate the stripe-centre positions per group.

    Builds the bootstrap template from all recordings, counts each recording's stripes
    (:func:`count_zebra_stripes`), and times each counted stripe at its centre -- the midpoint of
    its rising edge and the falling edge of the same dark bar. Stripe-centre positions are then
    self-calibrated: every interval between counted centres, divided by the revolution period
    around it, is the sum of the stripe spacings it spans, and the spacings are the least-squares
    solution over all intervals of the recording -- together with those of the other members of its
    ``_TEMPLATE_GROUPS`` group and of the recordings it ``_TEMPLATE_BORROW``-s from, where present.
    Stripe identity comes from the shared bootstrap template, so the same index is the same
    physical stripe in every recording.

    Args:
        recordings: Per recording stem, ``t_ref`` (from :func:`reference_events`) and ``t_rise`` /
            ``t_fall`` (rising and falling zebra edge times, in s).

    Returns:
        Per stem ``(t, idx, template)`` -- stripe-centre times, their global stripe indices, and
        the calibrated centre positions (revolutions, ascending from 0) of that stem's group.
    """
    phases, revs = {}, {}
    for stem, r in recordings.items():
        phases[stem], revs[stem] = _zebra_phase(r["t_ref"], r["t_rise"])
    template, counts, pooled_hist = pool_zebra_templates(list(phases.values()), list(revs.values()))
    n = len(template)
    total_revs = sum(revs.values())
    print(
        f"Bootstrap stripe template across {len(recordings)} recordings: {n} stripes; per-stripe detection "
        f"rate min={counts.min() / total_revs:.1%} median={np.median(counts) / total_revs:.1%} "
        f"max={counts.max() / total_revs:.1%}"
    )
    if n <= 2 * _CUTOFF_ORDER:
        raise RuntimeError(
            f"template resolved only {n} stripes/rev, giving an order-domain Nyquist of {n / 2} -- "
            f"too low for the {_CUTOFF_ORDER}-order cutoff"
        )

    centres, normal = {}, {}
    for stem, r in recordings.items():
        t, idx, n_dropped = count_zebra_stripes(
            r["t_rise"], r["t_ref"], template, zebra_phase_offset(phases[stem], pooled_hist)
        )
        t_clean, idx_clean = drop_zebra_outliers(t, idx, template)

        # Stripe centres. Rising and falling edges alternate strictly, so a dark bar's falling edge
        # is the last one before its rising edge, provided it comes after the previous rising edge.
        rise, fall = r["t_rise"], r["t_fall"]
        i = np.searchsorted(rise, t_clean)
        j = np.searchsorted(fall, t_clean) - 1
        paired = (j >= 0) & ((i == 0) | (fall[np.maximum(j, 0)] > rise[np.maximum(i - 1, 0)]))
        # Abnormal bar widths: a stripe's usual width is taken where the stripe before it was seen too,
        # which a merge -- it swallows that stripe -- never is, so it holds even for a stripe merged in
        # most revolutions.
        width = (t_clean - fall[j]) / np.gradient(t_clean, idx_clean)
        stripe = idx_clean % n
        after_seen = np.r_[False, np.diff(idx_clean) == 1]
        usual = np.full(n, np.nan)
        for k in range(n):
            sel = paired & (stripe == k)
            if sel.any():
                usual[k] = np.median(width[sel & after_seen] if (sel & after_seen).any() else width[sel])
        abnormal = paired & (np.abs(width - usual[stripe]) >= _BAR_WIDTH_TOL)
        keep = paired & ~abnormal
        t_c, idx = (fall[j[keep]] + t_clean[keep]) / 2, idx_clean[keep]
        centres[stem] = (t_c, idx)
        print(
            f"  {stem}: {len(t_c)} of {len(rise)} stripes labelled; {n_dropped} carrier revolutions dropped by "
            f"the 1PR check, {len(t) - len(t_clean)} edges by the outlier check, {abnormal.sum()} bars of abnormal width"
        )

        # Self-calibration normal equations. Each interval k0 -> k1 is normalised by the period of
        # one revolution centred on it (so a steady speed change cancels to first order), and spans
        # the spacings ending at stripes k0+1 .. k1. That revolution starts at the last stripe seen at
        # or before (k0 + k1) / 2 - n / 2 and ends at the same stripe one turn later: timing a gap by
        # interpolation would presuppose the very spacings being estimated.
        k0, k1, dt = idx[:-1], idx[1:], np.diff(t_c)
        g0 = idx[0]
        t_at = np.full(idx[-1] - g0 + n + 1, np.nan)
        t_at[idx - g0] = t_c
        start = idx[np.maximum(np.searchsorted(idx, (k0 + k1) // 2 - n // 2, side="right") - 1, 0)]
        period = t_at[start + n - g0] - t_at[start - g0]
        use = (k1 - k0 <= _CAL_MAX_STEP) & (start <= k0) & (k1 <= start + n) & np.isfinite(period)
        k0, k1, x = k0[use], k1[use], dt[use] / period[use]
        AtA, Atb = np.zeros((n, n)), np.zeros(n)
        for step in np.unique(k1 - k0):
            sel = k1 - k0 == step
            first = (k0[sel] + 1) % n
            cnt = np.bincount(first, minlength=n)
            tot = np.bincount(first, weights=x[sel], minlength=n)
            for s0 in np.flatnonzero(cnt):
                cols = (s0 + np.arange(step)) % n
                AtA[np.ix_(cols, cols)] += cnt[s0]
                Atb[cols] += tot[s0]
        normal[stem] = (AtA, Atb)

    labels = {}
    for stem in recordings:
        pool = next((g for g in _TEMPLATE_GROUPS if stem in g), (stem,)) + _TEMPLATE_BORROW.get(stem, ())
        group = [s for s in pool if s in normal]
        # Minimum-norm solution: a stripe that is never seen on its own leaves only the sum of its
        # two neighbouring spacings determined; how that sum is split does not matter, since no
        # edge is ever placed on that stripe.
        spacing = np.linalg.lstsq(
            sum(normal[s][0] for s in group), sum(normal[s][1] for s in group), rcond=None
        )[0]
        if (spacing <= 0).any():
            raise RuntimeError(f"self-calibration of {group} gave a non-positive stripe spacing")
        spacing /= spacing.sum()
        # spacing[k] runs from stripe k-1 to stripe k, so stripe 0 sits at 0 and the rest follow.
        labels[stem] = (*centres[stem], np.concatenate([[0.0], np.cumsum(spacing[1:])]))
    return labels


# ───────────────────────── reconstruction ─────────────────────────


def angle_domain_ias(t: np.ndarray, idx: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unfiltered IAS (Hz), uniformly sampled in angle at ``len(template)`` points per revolution.

    The labelled stripes give angle at irregular times. Monotone PCHIP through them, evaluated at
    a uniform *angle* grid, inverts that into the time of each angle step; differencing gives a
    rate that is genuinely uniform in angle, which is what makes an order-domain filter
    well-posed. Timing jitter passes through this with the first-difference ``sin^2(pi k / ppr)``
    shape of a plain encoder.

    Returns:
        ``(ias, t_mid)`` -- the rate over each angle step, and the time at the middle of each step.
    """
    angle = _theta(idx, template)
    n_per_rev = len(template)
    theta_grid = np.arange(angle[0], angle[-1], 1 / n_per_rev)
    t_of_theta = PchipInterpolator(angle, t)(theta_grid)
    return (1 / n_per_rev) / np.diff(t_of_theta), (t_of_theta[:-1] + t_of_theta[1:]) / 2


def reconstruct_ias(
    t: np.ndarray, idx: np.ndarray, template: np.ndarray, fs: float, signal_len: int
) -> tuple[np.ndarray, slice]:
    """Continuous sun-shaft IAS (Hz) from labelled stripes, low-passed in the order domain.

    :func:`angle_domain_ias`, low-passed at ``_CUTOFF_ORDER``, then mapped back onto the
    recording's sample grid. Because the result is truncated to the labelled span rather than
    extrapolated beyond it, no constant-rate hold is needed outside that span -- which is what the
    earlier ~860,000 Hz blowups came from.

    Returns:
        ``(ias, sl)`` -- IAS in Hz on the recording's sample grid over the labelled span, and the
        ``slice`` of that grid, to be applied to the vibration channels as well.
    """
    ias_angle_domain, t_mid = angle_domain_ias(t, idx, template)
    ias_filt = order_domain_lowpass(ias_angle_domain, _CUTOFF_ORDER, len(template))

    t_grid = np.arange(signal_len) / fs
    sl = slice(
        int(np.searchsorted(t_grid, t_mid[0], side="left")), int(np.searchsorted(t_grid, t_mid[-1], side="right"))
    )
    return np.clip(np.interp(t_grid[sl], t_mid, ias_filt), 0, None), sl


# ───────────────────────── dataset preparation ─────────────────────────


def dl_planetary_gearbox(
    save_path: Path,  # directory the files are written to, created if it does not exist
    force_download: bool = False,  # unused; the framework only calls this when the dataset is missing or forced
) -> None:
    """Download, preprocess (zebra tape → sun-shaft IAS), split, and add disturbed test sets.

    Runs in two passes over the archive because both the bootstrap template and the stripe
    calibration are pooled across recordings. The first pass reads only the two tacho channels
    and keeps just their edge times; the second reads only the two vibration channels. Each
    ~2.5 GB .MAT is therefore read once per pass and never held in full alongside the others.
    """
    save_path = Path(save_path)
    for split in ("train", "valid", "test", "test_wear"):
        (save_path / split).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        download_and_unpack(_INFO, temp_dir)

        # The discovery glob is exactly `*_crack/*.MAT` -- other .MAT files in the archive are
        # not used. Sorted, because `pool_zebra_templates` seeds its reference from this order.
        mat_files = sorted(temp_dir.rglob("*_crack/*.MAT"), key=lambda p: p.stem)

        recordings, meta = {}, {}
        for mat_file in tqdm(mat_files, desc="Reading tacho channels", unit="file"):
            mat_data = scipy.io.loadmat(mat_file, variable_names=["Channel_5_Data", "Channel_6_Data", "File_Header"])
            fs = _parse_fs(mat_data)
            zebra = mat_data["Channel_5_Data"].squeeze().astype(float)
            recordings[mat_file.stem] = dict(
                t_ref=reference_events(mat_data["Channel_6_Data"].squeeze(), fs),
                t_rise=rising_edge_times(zebra, fs),
                t_fall=rising_edge_times(-zebra, fs),
            )
            meta[mat_file.stem] = dict(fs=fs, signal_len=len(zebra))
            del mat_data, zebra

        labels = calibrate_zebra(recordings)

        for mat_file in tqdm(mat_files, desc="Reconstructing IAS", unit="file"):
            m = meta[mat_file.stem]
            ias, sl = reconstruct_ias(*labels[mat_file.stem], m["fs"], m["signal_len"])

            mat_data = scipy.io.loadmat(mat_file, variable_names=["Channel_2_Data", "Channel_3_Data"])
            signals = {
                "IAS": ias,
                "Acc_Carrier": mat_data["Channel_2_Data"].squeeze()[sl] / 9 * 9.81,
                "Acc_Sun": mat_data["Channel_3_Data"].squeeze()[sl] / 9 * 9.81,
            }
            del mat_data
            stem = mat_file.stem
            if any(t in stem for t in _TEST_WEAR_TYPES):
                target_subdir = "test_wear"
            elif any(t in stem for t in _TEST_BASIC_TYPES):
                target_subdir = "test"
            elif any(t in stem for t in _VALID_TYPES):
                target_subdir = "valid"
            else:
                target_subdir = "train"
            # gear ratio: (planet carrier, sun, mesh) relative to the SUN, which is the shaft the
            # zebra tape -- and hence the IAS -- refers to. The mesh entry is unchanged from the
            # carrier-referenced list: 62 * 13 / 75 = 13 * (1 - 13/75) was already the mesh order
            # relative to the sun (relative to the carrier it would be the ring tooth count, 62).
            save_signals_hdf5(
                signals,
                save_path / target_subdir / f"{mat_file.stem}.hdf5",
                fs=m["fs"],
                gear_ratio=[
                    1 / _SUN_PER_CARRIER_REV,
                    1,
                    _RING_TEETH * _SUN_TEETH / (_SUN_TEETH + _RING_TEETH),
                ],
                ias_bandwidth_hz=_IAS_BANDWIDTH_HZ,
            )

    write_disturbed_test_sets(save_path, vib_keys=["Acc_Carrier", "Acc_Sun"])


# version 2: IAS reconstructed from the sun-shaft zebra tape (was the 1PR carrier pickup with a
# savgol + fixed 12.5 Hz time-domain low-pass). The label is now sun-referenced, so it is
# ~5.77x the previously shipped values.
# version 3: corrected circular stripe-template median & fixed order domain filter transfer function
# version 4: stripes counted and anchored at delay-compensated 1PR pulses (was: each pulse labelled
# from the interpolated reference), reconstructed from stripe centres with positions
# self-calibrated per reassembly group (was: one bootstrap template for all recordings).
planetary_gearbox_dataset = Dataset("planetary_gearbox", prepare=dl_planetary_gearbox, version="4")

_planetary_gearbox = dict(
    u_cols=["Acc_Carrier", "Acc_Sun"],
    y_cols=["IAS"],
    train=[(planetary_gearbox_dataset, "train/*.hdf5")],
    valid=[(planetary_gearbox_dataset, "valid/*.hdf5")],
    test_sets=ias_test_sets(planetary_gearbox_dataset),
)

BenchmarkPlanetaryGearbox_Estimation = BenchmarkSpec(
    name="BenchmarkPlanetaryGearbox_Estimation",
    # window_sec = largest window any upstream method needed (Ref-FFT-LSTM 2.70 s),
    # rounded to 2.7 s; the per-file fs (this dataset varies it) sizes the window in samples. See ias/__init__.
    task=WindowedEstimation(window_sec=2.7),
    **_planetary_gearbox,
)

BenchmarkPlanetaryGearbox_GridwiseEstimation = BenchmarkSpec(
    name="BenchmarkPlanetaryGearbox_GridwiseEstimation",
    # window_sec=3.0: the largest single window across every upstream method's search space
    # over all four IAS datasets (unlike the per-dataset WindowedEstimation windows above,
    # this one is kept uniform — it's only a context guarantee, not a tuned averaging window).
    # step_sec: the one dataset NOT sampled at Nyquist. _IAS_BANDWIDTH_HZ = 440.0 Hz would need
    # 1.14 ms, i.e. up to ~0.81M query points per file and ~480 MB of diagnostics per run. Capped at
    # 3 ms instead, which fully resolves 15 orders while the sun is below 11.1 Hz -- 60.4% of
    # recorded time. That is a deliberate trade: pooled MAE is an unbiased estimate of the mean
    # absolute error at ANY grid spacing, so a model that fails to track fast content is still
    # penalised at every query point; and 3 ms is already finer than the finest hop any
    # benchmarked method can emit (5 ms for MOPA/ViBES at their smallest window and largest
    # overlap), so the grid never limits what a method can demonstrate. What it does cost is
    # aliasing-free spectral analysis of the stored diagnostics -- hence step_sec is recorded
    # alongside the results so that limitation is visible rather than silent.
    task=GridwiseEstimation(window_sec=3.0, step_sec=0.003),
    **_planetary_gearbox,
)

# Dense free-run sibling (framework Simulation task): the model predicts one IAS estimate
# per sample over the full recording — the window lives in the model (e.g. a sliding window),
# not the benchmark — scored per-sample MAE in Hz. Same data and test sets as the windowed task.
BenchmarkPlanetaryGearbox_Simulation = BenchmarkSpec(
    name="BenchmarkPlanetaryGearbox_Simulation",
    task=Simulation(metric=mae),
    **_planetary_gearbox,
)
