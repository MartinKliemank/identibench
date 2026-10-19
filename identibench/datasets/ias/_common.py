"""Shared download/prep helpers for the IAS (instantaneous angular speed) datasets."""

import multiprocessing
import shutil
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np
import scipy.signal
from scipy import stats
from scipy.signal import butter, buttord, sosfiltfilt
from tqdm import tqdm

from ...dataset import Dataset
from ...utils import download_file, extract_archive, hdf_files_from_path, preparation_workers

# Synthetic-disturbance SNR levels of the disturbed test-set copies (dB), defined per frequency band
# (see BAND_OCTAVES).
DISTURBANCE_LEVELS = [15, 7.5, 0, -7.5]

# Disturbed-test-set variants: "combined" sums the harmonic/gaussian/impulsive components
# (noise power split evenly across them); the other three isolate one component at 100% of
# the noise power. Order here is also the per-level order used by `ias_test_sets`
# (combined, the per-level headline, first).
DISTURBANCE_VARIANTS: tuple[str, ...] = ("combined", "harmonic", "gaussian", "impulsive")

# What `add_disturbances(disturbance_types=...)` receives for each variant above.
# "combined" is never passed to `add_disturbances` itself -- it is purely an
# orchestration-layer name for the mixture.
_VARIANT_DISTURBANCE_TYPES: dict[str, tuple[str, ...]] = {
    "combined": ("harmonic", "gaussian", "impulsive"),
    "harmonic": ("harmonic",),
    "gaussian": ("gaussian",),
    "impulsive": ("impulsive",),
}

# SNR is defined per frequency band, not against the total power: in every BAND_OCTAVES-wide band a
# disturbance carries 10**(-snr/10) times the clean signal's power in that band. Against the total
# power, white noise is worth very different amounts to different datasets and methods -- most of it
# lands outside the band a decimating model keeps, or where the signal has no content -- so the same
# nominal 15 dB came out between -18 and +29 dB inside the bands the HPO_6 models use. The level holds
# on average over the bands: smoothing (and the Welch main lobe) spreads a strong line's power over its
# neighbourhood, so the bands right beside an isolated strong line (planetary 49 Hz line, gas foil
# bearing 1x) get more noise than their own power warrants -- up to ~20 dB in 1/12-octave bands, in
# bands holding a few percent of the signal power. 1/12 rather than the usual 1/3 octave keeps that
# region to about +-3 % of the line.
BAND_OCTAVES = 1 / 12
# Resolution of the long-term clean spectrum the disturbances are scaled to. The Welch bins come out at
# 0.29-0.39 Hz on these datasets (a power of two near fs / REF_RESOLUTION_HZ samples); below ~5-7 Hz a
# 1/12-octave band is narrower than one bin and is widened to it.
REF_RESOLUTION_HZ = 0.5

# Harmonic disturbance: a nearby machine, i.e. a comb of HARMONIC_COUNT lines at
# k * HARMONIC_IAS_RATIO * IAS_other(t). The speed profile IAS_other is the IAS label of a *different*
# recording (`interferer_ias`), never the file's own: a tone tied to the file's own mean speed lets an
# estimator that picks the strongest low-frequency peak score better than on the clean signal. The
# comb ends at HARMONIC_COUNT * f_t (~150 Hz on the parallel gearbox, ~560 Hz on the ball bearing,
# ~14 kHz on the gas foil bearing): that covers the bands the low-rate models use, but there is no
# harmonic noise above it, so the harmonic variant's total SNR is well above the nominal level. Each
# line costs one sine evaluation over the whole recording.
HARMONIC_IAS_RATIO = 3 / np.pi
HARMONIC_COUNT = 32
# The other recording must also run at a different speed: recordings made at the same setpoint (the
# parallel gearbox's 10/20 Nm pairs) would put the comb's first line back at ~0.95x the own speed.
# `write_disturbed_test_sets` takes the next test file whose mean speed differs by at least this
# fraction, else the one that differs most.
INTERFERER_MIN_SPEED_GAP = 0.2

# Impulsive disturbance: heavy-tailed Levy-stable draws, clipped, then shaped like the gaussian
# component. With alpha < 2 the variance is infinite, so without clipping a single sample can carry
# most of the noise energy and the mean-square SNR normalisation becomes meaningless. The clip is set
# by that behaviour rather than by a tail quantile: at 600 the largest sample holds <1% of the noise
# energy on the prepared recordings. Shaping spreads each impulse over the signal's band, which lowers
# the kurtosis, most where the spectrum is narrowest.
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

    Rising edges give per-pulse periods; their reciprocal is the rate. Each period is
    centred between its two edges, i.e. at angle ``(k - 0.5) / ppr``, so the rate sequence
    is *already* uniform in angle and :func:`order_domain_lowpass` applies to it directly,
    with no angle-domain resampling needed. It is then interpolated onto the recording's own
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
    ias = order_domain_lowpass(1 / periods / pulses_per_revolution, cutoff_order, pulses_per_revolution)

    t = np.arange(len(signal)) / fs
    sl = slice(
        int(np.searchsorted(t, corrected_t[0], side="left")),
        int(np.searchsorted(t, corrected_t[-1], side="right")),
    )
    return np.clip(np.interp(t[sl], corrected_t, ias), 0, None), sl


def save_signals_hdf5(
    signals: dict[str, np.ndarray], path: Path, fs: float, gear_ratio, ias_bandwidth_hz: float
) -> None:
    """Write the signal channels as float32 datasets with the dataset's metadata attrs.

    Replaces the upstream ``save_to_hdf5`` (attr name normalized
    ``sampling_rate`` → ``fs``; identibench-style float32 storage).
    ``gear_ratio`` may be a scalar or a list (kinematics pass-through metadata).

    ``ias_bandwidth_hz`` is the highest frequency the ``IAS`` label retains,
    ``IAS_max * min(cutoff_order, ppr/2)`` in Hz. It is a property of how the label was
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
        f.attrs["ias_bandwidth_hz"] = float(ias_bandwidth_hz)


@dataclass
class ReferenceSpectrum:
    """Long-term spectrum of one clean channel: band powers and the noise-shaping filter.

    Built once per (file, channel) with :meth:`from_signal` and shared by every level and variant.

    Attributes:
        fs: Sampling rate of the channel in Hz.
        var: Variance of the channel (its total power with the mean removed).
        f: Welch bin frequencies in Hz, ``0 .. fs/2``.
        cum: Signal power below each ``f`` (trapezoid integral of the one-sided PSD).
        s_smooth: PSD averaged over a ``BAND_OCTAVES``-wide band around each ``f``, never narrower
            than one bin.
    """

    fs: float
    var: float
    f: np.ndarray
    cum: np.ndarray
    s_smooth: np.ndarray

    @classmethod
    def from_signal(cls, sig: np.ndarray, fs: float, resolution_hz: float = REF_RESOLUTION_HZ) -> "ReferenceSpectrum":
        """Welch spectrum of ``sig`` at ``resolution_hz``, coarsened until two segment lengths fit.

        Args:
            sig: Clean 1-D channel.
            fs: Its sampling rate in Hz.
            resolution_hz: Target frequency resolution in Hz.

        Returns:
            The reference spectrum of ``sig``.
        """
        x = np.asarray(sig, dtype=np.float64)
        x = x - x.mean()
        nperseg = 1 << int(np.ceil(np.log2(fs / resolution_hz)))
        while nperseg > max(len(x) // 2, 16):
            nperseg //= 2
        f, s = scipy.signal.welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend=False)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * f[1])])
        # Constant-Q smoothing: geometric band around f, but never narrower than one bin.
        r = 2 ** (BAND_OCTAVES / 2)
        lo = np.clip(np.minimum(f / r, f - f[1] / 2), 0, fs / 2)
        hi = np.clip(np.maximum(f * r, f + f[1] / 2), 0, fs / 2)
        s_smooth = (np.interp(hi, f, cum) - np.interp(lo, f, cum)) / (hi - lo)
        return cls(fs=fs, var=float(x.var()), f=f, cum=cum, s_smooth=s_smooth)

    def band_power(self, f_lo: np.ndarray, f_hi: np.ndarray) -> np.ndarray:
        """Signal power between ``f_lo`` and ``f_hi`` (Hz), elementwise."""
        return np.interp(f_hi, self.f, self.cum) - np.interp(f_lo, self.f, self.cum)

    @cached_property
    def fir(self) -> np.ndarray:
        """Zero-phase FIR whose magnitude response is the square root of the smoothed PSD."""
        n = 2 * (len(self.f) - 1)
        h = np.fft.fftshift(np.fft.irfft(np.sqrt(self.s_smooth), n=n))
        return h * scipy.signal.windows.hann(n, sym=False)


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


def add_disturbances(
    sig: np.ndarray,
    fs: float,
    target_snr_db: float,
    rng: np.random.Generator,
    interferer_ias_hz: np.ndarray,
    ref: ReferenceSpectrum,
    disturbance_types: tuple[str, ...] = ("harmonic", "gaussian", "impulsive"),
) -> np.ndarray:
    """Add disturbances at ``target_snr_db`` per ``BAND_OCTAVES``-wide band, each drawn from ``rng``.

    ``disturbance_types`` selects which of ``"harmonic"`` (a nearby machine), ``"gaussian"`` and
    ``"impulsive"`` (clipped Lévy-stable) to generate. The noise-to-signal power ratio
    ``10**(-target_snr_db/10)`` is split evenly across the requested types and applied band by
    band, relative to the clean signal's power in that band (``ref``):

    * gaussian / impulsive: white draws shaped by ``ref.fir`` (the square root of the smoothed
      spectrum), then scaled to exactly their share of the total power, so their total SNR is the
      target as well. The per-band level holds on average; bands right beside a strong line get
      more noise than their own power warrants (see ``BAND_OCTAVES``).
    * harmonic: a comb of ``HARMONIC_COUNT`` lines at ``k * HARMONIC_IAS_RATIO *
      interferer_ias_hz(t)``. Each line's power is the target fraction of the signal power in the
      band around its *current* frequency. Above ``k ~ 17`` neighbouring lines are less than a band
      apart and split each band's budget, which holds on average over band positions (a band
      centred on line ``k`` sits ``10*log10(k*(r-1/r))`` dB, at most ~2.7 dB, above the target).
      Below ``k ~ 17`` the bands between lines stay empty, and there is no harmonic noise above the
      comb's last line, so the harmonic component's total SNR is well above the target.

    Args:
        sig: Clean 1-D channel.
        fs: Its sampling rate in Hz.
        target_snr_db: Per-band SNR in dB.
        rng: Source of every random draw (line phases, gaussian and Lévy samples).
        interferer_ias_hz: Speed of the interfering machine in Hz, same shape as ``sig``; from
            :func:`interferer_ias`, never the recording's own label.
        ref: Reference spectrum of ``sig`` (:meth:`ReferenceSpectrum.from_signal`).
        disturbance_types: Components to add.

    Returns:
        ``sig`` plus the disturbances, float64.
    """
    # noise-to-signal power ratio in every band, for each requested component
    ratio = 10 ** (-target_snr_db / 10) / len(disturbance_types)
    r = 2 ** (BAND_OCTAVES / 2)
    noise = np.zeros(sig.shape, dtype=np.float64)
    for kind in ("harmonic", "gaussian", "impulsive"):
        if kind not in disturbance_types:
            continue
        if kind == "harmonic":
            f_t = HARMONIC_IAS_RATIO * np.asarray(interferer_ias_hz, dtype=np.float64)
            # float64 throughout: cumsum over long recordings (e.g. 200 kHz for 10s) accumulates
            # too much rounding error in float32 to stay phase-accurate.
            phase = np.cumsum(2 * np.pi * f_t / fs)
            for k in range(1, HARMONIC_COUNT + 1):
                f_k = k * f_t
                # lines of spacing f_t that share the band around f_k split that band's budget
                n_lines = max(1.0, k * (r - 1 / r))
                amp = np.sqrt(2 * ratio * ref.band_power(f_k / r, f_k * r) / n_lines)
                amp[f_k >= 0.45 * fs] = 0  # a line that would alias is dropped, not folded
                noise += amp * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
        else:
            if kind == "gaussian":
                white = rng.normal(0, 1, size=sig.shape)
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
                                size=min(chunk, sig.size - i),
                                random_state=rng,
                            ),
                            -IMPULSIVE_CLIP,
                            IMPULSIVE_CLIP,
                        )
                        for i in range(0, sig.size, chunk)
                    ]
                ).reshape(sig.shape)
            shaped = scipy.signal.oaconvolve(white, ref.fir, mode="same")
            noise += shaped * np.sqrt(ratio * ref.var / np.mean(shaped**2))
    return sig + noise


def _disturbance_rng(base_seed: int, stem: str, level: float, variant: str) -> np.random.Generator:
    """Per-(file, level, variant) deterministic generator, independent of iteration order.

    The variants for a given (file, level) deliberately do not share noise
    realizations with each other -- each draws independently from its own seed.
    """
    return np.random.default_rng(base_seed ^ zlib.crc32(f"{stem}|{level}|{variant}".encode()))


def _write_disturbed_copies(
    dataset_path: Path, file: Path, other: Path, vib_keys: list[str], noise_levels: list[float], base_seed: int
) -> None:
    """All disturbed copies of one test file: one worker's share of :func:`write_disturbed_test_sets`.

    ``other`` is the test file whose IAS drives the harmonic interferer. Module-level so that a
    spawned worker process can import it.
    """
    with h5py.File(file, "r") as f:
        own_ias, fs = f["IAS"][:], float(f.attrs["fs"])
        clean = {k: f[k][:] for k in vib_keys}
    with h5py.File(other, "r") as f:
        other_ias, other_fs = f["IAS"][:], float(f.attrs["fs"])
    refs = {k: ReferenceSpectrum.from_signal(clean[k], fs) for k in vib_keys}
    for level in noise_levels:
        for variant in DISTURBANCE_VARIANTS:
            rng = _disturbance_rng(base_seed, file.stem, level, variant)
            interferer = interferer_ias(own_ias, fs, other_ias, other_fs, rng)
            dest_dir = dataset_path / f"test_disturbed_{level}dB_{variant}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / file.name
            shutil.copy(str(file), str(dest_path))
            with h5py.File(dest_path, "r+") as f:
                for k in vib_keys:
                    f[k][:] = add_disturbances(
                        clean[k],
                        fs,
                        target_snr_db=level,
                        rng=rng,
                        interferer_ias_hz=interferer,
                        ref=refs[k],
                        disturbance_types=_VARIANT_DISTURBANCE_TYPES[variant],
                    )


def write_disturbed_test_sets(
    dataset_path: Path,
    vib_keys: list[str],
    noise_levels: list[float] = DISTURBANCE_LEVELS,
    base_seed: int = 0,
) -> None:
    """Copy the basic test set into disturbed variant dirs with added disturbances.

    Reads ``dataset_path/test`` and writes, per noise level, one directory per entry in
    ``DISTURBANCE_VARIANTS`` -- ``test_disturbed_<level>dB_combined/`` (all three components, power
    split evenly) plus ``_harmonic/``, ``_gaussian/``, ``_impulsive/`` (each isolating one
    component at the full target SNR) -- corrupting the ``vib_keys`` channels in place; the IAS
    label and the attrs are copied unchanged. ``fs`` is read per file from its attrs (the planetary
    gearbox has per-file rates).

    Each test file is one task, run by ``preparation_workers`` processes (``IDENTIBENCH_WORKERS``
    overrides the count), longest files first: a task computes each channel's
    :class:`ReferenceSpectrum` (and its shaping filter) once and writes all of the file's copies.
    The harmonic interferer runs at the speed profile of another test file: the next one in sorted
    order (cyclic) whose mean speed differs from this file's by at least
    ``INTERFERER_MIN_SPEED_GAP``, else the one that differs most. It is therefore independent of
    the file's own label and not at its speed either. Each (file, level, variant) triple gets its
    own deterministic seed, so re-runs are byte-identical regardless of order and of the number of
    workers, and the variants for a given (file, level) do not share noise realizations with each
    other.

    Args:
        dataset_path: Prepared dataset directory containing ``test/``.
        vib_keys: Vibration channels to disturb.
        noise_levels: Per-band SNR levels in dB.
        base_seed: XORed into every per-(file, level, variant) seed.
    """
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
        jobs.append((dataset_path, file, other, vib_keys, noise_levels, base_seed))
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


def ias_test_sets(dataset: Dataset, wear: bool = True) -> dict[str, list[tuple[Dataset, str]]]:
    """The named test conditions of an IAS spec as patterns, headline (``basic``) first.

    Maps each condition to its directory: ``basic`` → ``test/``, ``wear`` →
    ``test_wear/`` (only where the dataset has a wear condition), and one
    ``disturbed_<level>dB_<variant>`` → ``test_disturbed_<level>dB_<variant>/``
    per level in ``DISTURBANCE_LEVELS`` × variant in ``DISTURBANCE_VARIANTS``
    (``combined`` first among the variants, as the per-level headline).
    """
    dirs = {"basic": "test"}
    if wear:
        dirs["wear"] = "test_wear"
    dirs |= {
        f"disturbed_{level}dB_{variant}": f"test_disturbed_{level}dB_{variant}"
        for level in DISTURBANCE_LEVELS
        for variant in DISTURBANCE_VARIANTS
    }
    return {name: [(dataset, f"{subdir}/*.hdf5")] for name, subdir in dirs.items()}
