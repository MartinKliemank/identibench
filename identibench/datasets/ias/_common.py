"""Shared download/prep helpers for the IAS (instantaneous angular speed) datasets."""

import multiprocessing
import shutil
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.signal
from scipy import stats
from scipy.signal import butter, buttord, sosfiltfilt
from tqdm import tqdm

from ...dataset import Dataset
from ...utils import download_file, extract_archive, hdf_files_from_path, preparation_workers
from .harmonics import HARMONIC_LINES_FILE, AngleGrid, HarmonicReference, HarmonicSettings, dataset_lines

# Synthetic-disturbance levels of the disturbed test-set copies (dB). A level S is the SNR of the recording's
# speed-locked harmonics: a line carries S dB more power than the added disturbance inside its own one-order
# band, i.e. over one revolution of data (see `harmonics.HarmonicReference`). Exact at constant speed; under
# speed sweeps the strongest line sits within ~0.5-2 dB of S, while weaker lines that share frequencies with
# stronger ones (at other times, or within one order) get more noise than their own power warrants. Against the total
# power, the same nominal level put the dominant line 1.5 (ball bearing) to 11 dB (parallel gearbox) above
# the noise, and in-band SNRs at the HPO_6 models' sampling rates spanned -18 to +29 dB.
DISTURBANCE_LEVELS = [15, 7.5, 0, -7.5]

# Speeds at or below this are not scored by the GridwiseEstimation headline (relative error), as at standstill
# the relative error is undefined. 1 Hz (60 rpm) is below every dataset's slowest normal running (parallel gearbox:
# 1.16 Hz), so only standstill drops out. The gas foil bearing overrides it with 10 Hz (its converter-based label
# is not accurate enough below that; see gas_foil_bearing._MIN_SCORED_IAS_HZ).
MIN_SCORED_IAS_HZ = 1.0

# Disturbed-test-set variants: "combined" sums the harmonic/gaussian/impulsive components
# (noise power split evenly across them); the other three isolate one component at 100% of
# the noise power. Order here is also the per-level order used by `ias_test_sets`
# (combined, the per-level headline, first).
DISTURBANCE_VARIANTS: tuple[str, ...] = ("combined", "harmonic", "gaussian", "impulsive")

# What `disturbance_noise(disturbance_types=...)` receives for each variant above.
# "combined" is never passed to `disturbance_noise` itself -- it is purely an
# orchestration-layer name for the mixture.
_VARIANT_DISTURBANCE_TYPES: dict[str, tuple[str, ...]] = {
    "combined": ("harmonic", "gaussian", "impulsive"),
    "harmonic": ("harmonic",),
    "gaussian": ("gaussian",),
    "impulsive": ("impulsive",),
}

# Harmonic disturbance: a nearby machine, i.e. a comb of lines at k * HARMONIC_IAS_RATIO * IAS_other(t). The
# speed profile IAS_other is the IAS label of a *different* recording (`interferer_ias`), never the file's
# own: a tone tied to the file's own mean speed lets an estimator that picks the strongest low-frequency peak
# score better than on the clean signal. Each line carries 10**(-S/10) times the power an own harmonic would
# have at its frequency (the reference density times one own order): a line-to-line signal-to-interference
# ratio of S; where the comb is denser than one line per own order, its power per own order is that share. The comb runs up to the highest frequency the own lines reach, but at most HARMONIC_MAX_LINES
# lines (each costs one sine over the recording), so on the gearboxes it stops below their mesh orders.
HARMONIC_IAS_RATIO = 3 / np.pi
HARMONIC_MAX_LINES = 128
# The other recording must also run at a different speed: recordings made at the same setpoint (the
# parallel gearbox's 10/20 Nm pairs) would put the comb's first line back at ~0.95x the own speed.
# `write_disturbed_test_sets` takes the next test file whose mean speed differs by at least this
# fraction, else the one that differs most.
INTERFERER_MIN_SPEED_GAP = 0.2

# Impulsive disturbance: heavy-tailed Levy-stable draws, clipped, then shaped like the gaussian
# component. With alpha < 2 the variance is infinite, so without clipping a single sample can carry
# most of the noise energy and the mean-square SNR normalisation becomes meaningless. The clip is set
# by that behaviour rather than by a tail quantile: at 600 the largest sample holds <1% of the noise
# energy on the prepared recordings. Shaping spreads each impulse over the reference's band, which lowers
# the kurtosis, most where the reference is narrowest.
# Note that `scipy.stats.levy_stable.ppf` saturates far in the tail (q >= 0.99999), so use the
# asymptotic P(X > x) ~ (1+beta)/2 * C_a * x**-alpha for tail quantiles instead.
IMPULSIVE_ALPHA = 1.4
IMPULSIVE_BETA = -0.1
IMPULSIVE_CLIP = 600.0


@dataclass
class DatasetInfo:
    """Source archive of one IAS dataset."""

    name: str
    zip_url: str
    download_headers: dict[str, str] | None = None


def _require_sklearn():
    """The stratified splits need scikit-learn; import lazily with an install hint."""
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise ImportError(
            'scikit-learn is required for the IAS datasets. Install it with: pip install "identibench[ias]"'
        ) from e
    return train_test_split


def download_and_unpack(dataset_info: DatasetInfo, output_dir: Path, nested_zip: bool = False) -> Path:
    """Download the dataset zip and extract it into ``output_dir``.

    Args:
        dataset_info: Source archive description.
        output_dir: Extraction target; created if it does not exist.
        nested_zip: Set for Parallel_Gearbox, whose zip contains another zip
            with the actual data that must be extracted as well. macOS resource
            forks (``__MACOSX``, ``._*``) are ignored when locating inner zips.
    """
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "dataset.zip"
        download_file(
            dataset_info.zip_url, archive_path, headers=dataset_info.download_headers or {"User-Agent": "Mozilla/5.0"}
        )
        print("Download completed, extracting .zip ...", end="")
        extract_archive(archive_path, output_dir)
        if nested_zip:
            inner_zips = sorted(
                p for p in output_dir.rglob("*.zip") if "__MACOSX" not in p.parts and not p.name.startswith("._")
            )
            if not inner_zips:
                raise FileNotFoundError(f"nested_zip=True but no inner .zip found under {output_dir}")
            for inner_zip in inner_zips:
                extract_archive(inner_zip, output_dir)
        print("complete.")
    return output_dir


def rising_edge_times(signal: np.ndarray, fs: float) -> np.ndarray:
    """Sub-sample times (s) of the rising edges of an analog pulse train.

    The signal is normalized to [-1, 1] and each upward zero crossing is located to
    sub-sample resolution by linear interpolation between the two bracketing samples.

    Args:
        signal: Raw analog tacho/encoder channel.
        fs: Sampling rate of ``signal`` in Hz.

    Returns:
        Edge times in seconds, ascending.
    """
    signal = np.asarray(signal, dtype=float)
    signal = ((signal - np.min(signal)) / (np.max(signal) - np.min(signal))) * 2 - 1
    # `signal >= 0` rather than `np.sign(signal)`: the sign form steps 0 -> +1 and -1 -> 0
    # separately, so a sample landing exactly on zero registers as two rising edges and the
    # pulse is counted twice.
    peaks = np.argwhere(np.diff((signal >= 0).astype(np.int8)) > 0).ravel()
    peaks = peaks - signal[peaks] / (signal[peaks + 1] - signal[peaks])
    return peaks / fs


def order_domain_lowpass(x: np.ndarray, cutoff_order: float, pulses_per_revolution: float) -> np.ndarray:
    """Zero-phase Butterworth low-pass of an angle-domain sequence, cut off in shaft orders.

    ``x`` must be sampled uniformly in *angle* at ``pulses_per_revolution`` samples per
    revolution. That value is therefore the filter's sampling rate, which lets the cutoff be
    stated directly in orders (events per revolution) and makes the smoothing independent of
    how fast the shaft happens to be turning.

    The stopband is ``min(4 * cutoff_order, 0.25 * ppr)``, specified to ``buttord`` as 1.5 dB
    passband / 30 dB stopband per pass -- so after ``sosfiltfilt``'s two passes the cutoff order
    is the -3 dB power point and the stopband is down 60 dB. Deriving it from the cutoff rather
    than from the Nyquist is what keeps the datasets comparable: a fixed ``0.25 * ppr`` stopband
    would hand the ball bearing a 2nd-order filter (its Nyquist sits 54x above its cutoff) and the
    parallel gearbox a 6th-order one, for no physical reason.

    Args:
        x: Angle-domain sequence, uniformly sampled.
        cutoff_order: Cutoff in orders; must be below the order-domain Nyquist.
        pulses_per_revolution: Samples of ``x`` per shaft revolution.

    Returns:
        The filtered sequence, same shape as ``x``.

    Raises:
        ValueError: If the cutoff is outside ``(0, ppr / 2)``, or the stopband derived from
            it collapses onto the passband.
    """
    nyquist = 0.5 * pulses_per_revolution
    if not 0 < cutoff_order < nyquist:
        raise ValueError(
            f"cutoff_order={cutoff_order} must lie strictly between 0 and the order-domain "
            f"Nyquist {nyquist} (pulses_per_revolution={pulses_per_revolution})"
        )
    stopband = min(4 * cutoff_order, 0.25 * pulses_per_revolution)
    if stopband <= cutoff_order:
        raise ValueError(
            f"stopband {stopband} collapsed onto cutoff_order={cutoff_order}; "
            f"pulses_per_revolution={pulses_per_revolution} is too small for this cutoff"
        )
    butter_order, wn = buttord(cutoff_order, stopband, 1.5, 30, fs=pulses_per_revolution)
    sos = butter(butter_order, wn, btype="low", output="sos", fs=pulses_per_revolution)
    return sosfiltfilt(sos, x, padtype="even", padlen=int(0.01 * len(x)))


def encoder_pulse_to_ias(
    signal: np.ndarray, fs: float, pulses_per_revolution: int, cutoff_order: float
) -> tuple[np.ndarray, slice]:
    """Instantaneous angular speed (Hz) from an analog encoder pulse train.

    Rising edges give per-pulse periods. Each period is
    centred between its two edges, i.e. at angle ``(k - 0.5) / ppr``, so the rate sequence
    is uniform in angle and :func:`order_domain_lowpass` applies to it directly,
    with no angle-domain resampling needed. The periods' reciprocal is then the IAS. Which is interpolated onto the recording's own
    sample grid.

    Args:
        signal: Raw analog tacho/encoder channel.
        fs: Sampling rate of ``signal`` in Hz.
        pulses_per_revolution: Encoder pulses per revolution of the measured shaft.
        cutoff_order: Order-domain cutoff, in orders of that shaft.

    Returns:
        ``(ias, sl)`` -- the IAS in Hz (rev/s), truncated to the span where it is genuinely
        measured, and the ``slice`` of the original sample grid it was cut to. The caller
        must apply ``sl`` to every other channel of the recording to keep them aligned; the
        IAS is not extrapolated beyond the first and last pulse.
    """
    pulse_times = rising_edge_times(signal, fs)
    periods = np.diff(pulse_times)
    corrected_t = pulse_times[1:] - periods / 2
    ias = 1 / order_domain_lowpass(periods, cutoff_order, pulses_per_revolution) / pulses_per_revolution # filter before reciprocal

    t = np.arange(len(signal)) / fs
    sl = slice(
        int(np.searchsorted(t, corrected_t[0], side="left")),
        int(np.searchsorted(t, corrected_t[-1], side="right")),
    )
    return np.clip(np.interp(t[sl], corrected_t, ias), 0, None), sl


def save_signals_hdf5(
    signals: dict[str, np.ndarray], path: Path, fs: float, gear_ratio, ias_bandwidth_orders: float
) -> None:
    """Write the signal channels as float32 datasets with the dataset's metadata attrs.

    Replaces the upstream ``save_to_hdf5`` (attr name normalized
    ``sampling_rate`` → ``fs``; identibench-style float32 storage).
    ``gear_ratio`` may be a scalar or a list (kinematics pass-through metadata).
    ``ias_bandwidth_order`` is the highest frequency the ``IAS`` label retains,
    ``IAS_max * min(cutoff_order, ppr/2)`` in orders. It is a property of how the label was
    built, not of the recording, so it is the same for every file of a dataset. It is
    recorded here because two separate decisions depend on it and would otherwise each
    hardcode it: the benchmark's evaluation-grid spacing (``GridwiseEstimation.step_sec``)
    and, downstream, the lowest model sample rate worth searching — a model sampled below
    ``2 * ias_bandwidth_hz`` cannot represent the target it is scored against.
    """
    with h5py.File(path, "w") as f:
        for key, value in signals.items():
            f.create_dataset(key, data=np.asarray(value), dtype="f4")
        f.attrs["fs"] = fs
        f.attrs["gear_ratio"] = gear_ratio
        f.attrs["ias_bandwidth"] = float(ias_bandwidth_orders)
        f.attrs["ias_bandwidth_units"] = "orders"


def interferer_ias(
    own_ias: np.ndarray,
    fs: float,
    other_ias: np.ndarray,
    other_fs: float,
    rng: np.random.Generator,
    n_candidates: int = 16,
) -> np.ndarray:
    """Speed trace of the interfering machine: another recording's IAS at a random time offset.

    Of ``n_candidates`` random offsets (wrapping around the other recording) the one least
    correlated with ``own_ias`` is kept, so the interferer cannot follow this file's speed profile
    even when both recordings run the same, e.g. periodic up/down, profile.

    Args:
        own_ias: IAS label of the recording being disturbed (Hz), one value per sample.
        fs: Sampling rate of ``own_ias`` in Hz.
        other_ias: IAS label of a different recording (Hz). The same recording also works, as a
            shifted copy, for a test split with a single file.
        other_fs: Sampling rate of ``other_ias`` in Hz.
        rng: Source of the offsets.
        n_candidates: Number of random offsets to choose from.

    Returns:
        The interferer's speed in Hz, same length as ``own_ias``.
    """
    t = np.arange(len(own_ias)) / fs
    t_other = np.arange(len(other_ias)) / other_fs
    duration = len(other_ias) / other_fs
    probe = np.linspace(0, len(own_ias) - 1, 20000).astype(int)
    best_shift, best_corr = None, np.inf
    for shift in rng.uniform(0, duration, size=n_candidates):
        cand = np.interp((t[probe] + shift) % duration, t_other, other_ias)
        with np.errstate(invalid="ignore", divide="ignore"):  # a constant trace has no correlation
            corr = abs(np.nan_to_num(np.corrcoef(cand, own_ias[probe])[0, 1]))
        if corr < best_corr:
            best_shift, best_corr = shift, corr
    return np.interp((t + best_shift) % duration, t_other, other_ias)


def disturbance_noise(
    references: dict[str, HarmonicReference],
    own_ias_hz: np.ndarray,
    rng: np.random.Generator,
    interferer_ias_hz: np.ndarray | None = None,
    disturbance_types: tuple[str, ...] = ("harmonic", "gaussian", "impulsive"),
) -> dict[str, np.ndarray]:
    """Disturbances at 0 dB for the channels of one recording; multiply by ``10**(-S/20)`` for level S.

    The power is split evenly across ``disturbance_types``, each component referenced to the channel's
    `HarmonicReference`:

    * gaussian / impulsive: white (or clipped Levy-stable) draws shaped by ``ref.fir`` and scaled to their
      share of ``ref.total_power``, so their density is that share of the reference at every frequency.
      Every channel gets its own draws.
    * harmonic: a comb of an interfering machine, lines at ``k * HARMONIC_IAS_RATIO * interferer_ias_hz(t)``;
      each line carries its share of the power an own harmonic would have at its current frequency (the
      reference density times one own order, ``own_ias_hz``), or of the reference over the comb's line spacing
      where that is narrower (a slow or stopped interferer). The lines' phases are shared by the channels
      (one machine), their levels follow each channel's reference. A line that would alias is dropped.

    Args:
        references: `HarmonicReference` per channel, all built on the same recording.
        own_ias_hz: IAS label of that recording (Hz), one value per sample.
        rng: Source of every random draw (line phases, gaussian and Levy samples).
        interferer_ias_hz: Speed of the interfering machine in Hz, same length; from `interferer_ias`,
            never the recording's own label. Only needed for "harmonic".
        disturbance_types: Components to add.

    Returns:
        The noise per channel, float64.
    """
    share = 1 / len(disturbance_types)
    own_ias_hz = np.asarray(own_ias_hz, dtype=np.float64)
    fs = next(iter(references.values())).grid.fs
    noise = {channel: np.zeros(len(own_ias_hz)) for channel in references}
    if "harmonic" in disturbance_types:
        if interferer_ias_hz is None:
            raise ValueError('a "harmonic" disturbance needs interferer_ias_hz')
        fundamental = HARMONIC_IAS_RATIO * np.asarray(interferer_ias_hz, dtype=np.float64)
        # float64 throughout: the cumulative sum over long recordings (e.g. 35 M samples) must stay phase-accurate
        phase = np.cumsum(2 * np.pi * fundamental / fs)
        # An own harmonic's power at a frequency is the reference density times one own order (own_ias_hz Hz).
        # Where the comb's lines are closer than one own order (a slow or stopped interferer) the band is
        # their spacing instead, so the comb's power per own order is the share: without it the lines piled
        # up at the reference's low-frequency level and a stopped interferer left a DC offset.
        line_band_hz = np.minimum(np.maximum(own_ias_hz, 0.0), fundamental)
        # enough lines to reach the highest frequency an own line's band covers (at the median fundamental)
        n_lines = {
            channel: int(
                min(HARMONIC_MAX_LINES, np.ceil(ref.freqs[ref.occupied].max() / max(np.median(fundamental), 1e-6)))
            )
            for channel, ref in references.items()
        }
        offsets = rng.uniform(0, 2 * np.pi, size=max(n_lines.values()))  # one phase per line, shared by the channels
        # evaluated in chunks: the per-line temporaries stay in cache instead of ~10 full-length arrays
        chunk = 1 << 16
        for start in range(0, len(own_ias_hz), chunk):
            part = slice(start, start + chunk)
            for k, offset in enumerate(offsets, start=1):
                line_hz = k * fundamental[part]
                wave = np.sin(k * phase[part] + offset)
                for channel, ref in references.items():
                    if k > n_lines[channel]:
                        continue
                    # linear interpolation of the reference on its uniform grid: as np.interp, ~6x faster
                    # without its search
                    pos = np.clip(line_hz / ref.freqs[1], 0, len(ref.freqs) - 1)
                    lower = np.minimum(pos.astype(np.int64), len(ref.freqs) - 2)
                    frac = pos - lower
                    density = ref.density[lower] * (1 - frac) + ref.density[lower + 1] * frac
                    amplitude = np.sqrt(2 * share * density * line_band_hz[part])  # a sine's power is amplitude^2 / 2
                    amplitude[line_hz >= 0.45 * fs] = 0  # would alias: dropped, not folded
                    noise[channel][part] += amplitude * wave
    for kind in ("gaussian", "impulsive"):
        if kind not in disturbance_types:
            continue
        for channel, ref in references.items():
            if kind == "gaussian":
                white = rng.normal(0, 1, size=len(own_ias_hz))
            else:
                # drawn in chunks: scipy's levy_stable.rvs holds ~30 full-size temporaries, i.e.
                # ~8 GB for the 35 M-sample planetary recordings
                chunk = 1 << 21
                white = np.concatenate(
                    [
                        np.clip(
                            stats.levy_stable.rvs(
                                alpha=IMPULSIVE_ALPHA,
                                beta=IMPULSIVE_BETA,
                                loc=0,
                                scale=1,
                                size=min(chunk, len(own_ias_hz) - i),
                                random_state=rng,
                            ),
                            -IMPULSIVE_CLIP,
                            IMPULSIVE_CLIP,
                        )
                        for i in range(0, len(own_ias_hz), chunk)
                    ]
                )
            shaped = scipy.signal.oaconvolve(white, ref.fir, mode="same")  # spectrum shaped like the reference
            noise[channel] += shaped * np.sqrt(share * ref.total_power / np.mean(shaped**2))
    return noise


def add_disturbances(
    sig: np.ndarray,
    target_snr_db: float,
    rng: np.random.Generator,
    ref: HarmonicReference,
    own_ias_hz: np.ndarray,
    interferer_ias_hz: np.ndarray | None = None,
    disturbance_types: tuple[str, ...] = ("harmonic", "gaussian", "impulsive"),
) -> np.ndarray:
    """One channel plus its disturbances at ``target_snr_db`` (see `disturbance_noise`), float64."""
    noise = disturbance_noise({"sig": ref}, own_ias_hz, rng, interferer_ias_hz, disturbance_types)["sig"]
    return sig + noise * 10 ** (-target_snr_db / 20)


def _disturbance_rng(base_seed: int, stem: str, variant: str) -> np.random.Generator:
    """Per-(file, variant) deterministic generator, independent of iteration order.

    The variants of a file do not share noise realizations with each other -- each draws independently
    from its own seed. The levels of a variant scale one realization.
    """
    return np.random.default_rng(base_seed ^ zlib.crc32(f"{stem}|{variant}".encode()))


def _write_disturbed_copies(
    dataset_path: Path,
    file: Path,
    other: Path,
    vib_keys: list[str],
    noise_levels: list[float],
    base_seed: int,
    line_orders: np.ndarray,
    settings: HarmonicSettings,
) -> None:
    """All disturbed copies of one test file: one worker's share of :func:`write_disturbed_test_sets`.

    ``other`` is the test file whose IAS drives the harmonic interferer. Module-level so that a
    spawned worker process can import it.
    """
    with h5py.File(file, "r") as f:
        own_ias, fs = f["IAS"][:].astype(np.float64), float(f.attrs["fs"])
        clean = {k: f[k][:].astype(np.float64) for k in vib_keys}
    with h5py.File(other, "r") as f:
        other_ias, other_fs = f["IAS"][:], float(f.attrs["fs"])
    try:
        grid = AngleGrid.from_ias(own_ias, fs, settings)  # one angle grid for all channels
        refs = {k: HarmonicReference.from_signal(clean[k], grid, line_orders) for k in vib_keys}
    except ValueError as e:
        raise ValueError(f"{file}: {e}") from e
    for variant in DISTURBANCE_VARIANTS:
        rng = _disturbance_rng(base_seed, file.stem, variant)
        interferer = interferer_ias(own_ias, fs, other_ias, other_fs, rng)
        noise = None  # free the previous variant's noise before drawing the next
        noise = disturbance_noise(refs, own_ias, rng, interferer, _VARIANT_DISTURBANCE_TYPES[variant])
        for level in noise_levels:
            dest_dir = dataset_path / f"test_disturbed_{level}dB_{variant}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / file.name
            shutil.copy(str(file), str(dest_path))
            with h5py.File(dest_path, "r+") as f:
                for k in vib_keys:
                    f[k][:] = clean[k] + noise[k] * 10 ** (-level / 20)


def write_disturbed_test_sets(
    dataset_path: Path,
    vib_keys: list[str],
    settings: HarmonicSettings,
    noise_levels: list[float] = DISTURBANCE_LEVELS,
    base_seed: int = 0,
) -> None:
    """Copy the basic test set into disturbed variant dirs with added disturbances.

    First finds the dataset's speed-locked lines over all its prepared splits (`harmonics.dataset_lines`)
    and writes them to ``dataset_path / harmonics.HARMONIC_LINES_FILE``. Then reads ``dataset_path/test``
    and writes, per noise level, one directory per entry in ``DISTURBANCE_VARIANTS`` --
    ``test_disturbed_<level>dB_combined/`` (all three components, power split evenly) plus ``_harmonic/``,
    ``_gaussian/``, ``_impulsive/`` (each isolating one component at the full level) -- corrupting the
    ``vib_keys`` channels in place; the IAS label and the attrs are copied unchanged. ``fs`` is read per file
    from its attrs (the planetary gearbox has per-file rates). Every channel is disturbed relative to its own
    `HarmonicReference`, so the level is the SNR of its harmonics (see ``DISTURBANCE_LEVELS``).

    Each test file is one task, run by ``preparation_workers`` processes (``IDENTIBENCH_WORKERS``
    overrides the count), longest files first. The harmonic interferer runs at the speed profile of
    another test file: the next one in sorted order (cyclic) whose mean speed differs from this file's
    by at least ``INTERFERER_MIN_SPEED_GAP``, else the one that differs most. It is therefore independent
    of the file's own label and not at its speed either. Each (file, variant) pair gets its own
    deterministic seed, so re-runs are byte-identical regardless of order and of the number of workers;
    the variants of a file do not share noise realizations, and the levels of a variant scale one
    realization (so level-to-level differences are not blurred by redrawing the noise).

    Args:
        dataset_path: Prepared dataset directory containing ``test/`` (and the other splits).
        vib_keys: Vibration channels to disturb.
        settings: The dataset's angle-domain analysis settings.
        noise_levels: Levels in dB.
        base_seed: XORed into every per-(file, variant) seed.

    Raises:
        ValueError: If a test channel has no line standing out of its clean background.
    """
    lines = dataset_lines(dataset_path, vib_keys, settings)
    lines.save(dataset_path / HARMONIC_LINES_FILE)
    test_files = hdf_files_from_path(dataset_path / "test")
    mean_speed = {}
    for file in test_files:
        with h5py.File(file, "r") as f:
            mean_speed[file.stem] = float(np.mean(f["IAS"][:]))
    jobs = []
    for i, file in enumerate(test_files):
        # the next file (cyclic) at a clearly different mean speed, else the most different one;
        # the last candidate is the file itself, used only when it is the only test file
        own = mean_speed[file.stem]
        candidates = [test_files[(i + j) % len(test_files)] for j in range(1, len(test_files) + 1)]
        other = next(
            (c for c in candidates[:-1] if abs(mean_speed[c.stem] - own) >= INTERFERER_MIN_SPEED_GAP * abs(own)),
            max(candidates, key=lambda c: abs(mean_speed[c.stem] - own)),
        )
        jobs.append((dataset_path, file, other, vib_keys, noise_levels, base_seed, lines.orders, settings))
    jobs.sort(key=lambda job: job[1].stat().st_size, reverse=True)  # longest first balances the workers

    per_file = len(noise_levels) * len(DISTURBANCE_VARIANTS)
    n_workers = preparation_workers(len(jobs))
    with tqdm(total=per_file * len(jobs), desc="Creating disturbed test sets", unit="file") as pbar:
        if n_workers == 1:
            for job in jobs:
                _write_disturbed_copies(*job)
                pbar.update(per_file)
        else:
            # "spawn": workers start clean instead of inheriting the caller's threads and open handles
            with ProcessPoolExecutor(n_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                for future in as_completed([pool.submit(_write_disturbed_copies, *job) for job in jobs]):
                    future.result()
                    pbar.update(per_file)


def ias_test_sets(dataset: Dataset, ood: bool = True) -> dict[str, list[tuple[Dataset, str]]]:
    """The named test conditions of an IAS spec as patterns, headline (``basic``) first.

    Maps each condition to its directory: ``basic`` → ``test/``, ``ood`` →
    ``test_ood/`` (out-of-distribution; only where the dataset has one), and one
    ``disturbed_<level>dB_<variant>`` → ``test_disturbed_<level>dB_<variant>/``
    per level in ``DISTURBANCE_LEVELS`` × variant in ``DISTURBANCE_VARIANTS``
    (``combined`` first among the variants, as the per-level headline).
    """
    dirs = {"basic": "test"}
    if ood:
        dirs["ood"] = "test_ood"
    dirs |= {
        f"disturbed_{level}dB_{variant}": f"test_disturbed_{level}dB_{variant}"
        for level in DISTURBANCE_LEVELS
        for variant in DISTURBANCE_VARIANTS
    }
    return {name: [(dataset, f"{subdir}/*.hdf5")] for name, subdir in dirs.items()}
