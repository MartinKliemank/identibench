"""Speed-locked harmonics of IAS recordings: detection, harmonic reference spectrum and per-line SNR.

A recording is resampled to constant shaft angle with its IAS label (`AngleGrid`), cut into frames of
``segment_revs`` revolutions, and every frame is compared with the one a frame length later (angle-domain
discrete/random separation, Antoni & Randall 2004): content locked to the shaft keeps its phase relation
across the gap, random content -- and fixed-Hz content whenever the speed changes -- does not. Per order bin
of the angle-domain spectrum (`angle_spectra`):

* ``S``  total power density, the mean ``|X_b|^2`` over the frame pairs,
* ``D``  locked power density, ``|mean X_b conj(X_{b+2})|``: the part that keeps its phase relation,
* ``g2`` coherence ``|C|^2 / (S_1 S_2)``, 0 (nothing locked) to 1 (everything locked).

Densities are one-sided and per order. Within a constant-speed recording fixed-Hz and locked content look
the same, so the lines of a dataset are the orders locked in at least half of its recordings
(`dataset_lines`, saved next to the data as ``harmonic_lines.json``). `HarmonicReference` turns one
recording's lines into the reference spectrum the disturbed test sets are scaled to, and `line_snr`
measures the realised per-line SNR of any disturbed copy, per frame and in the long term.
"""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np
import scipy.signal as ss

from ...utils import hdf_files_from_path, preparation_workers

# A frame is paired with the one this many hops later; the hop is half a frame, so the pair does not overlap.
PAIR_DELAY_HOPS = 2
# Coherence test: false-alarm probability per order bin, and the least locked share of a bin's power that
# counts (with tens of thousands of frame pairs, statistically significant is not the same as relevant).
FALSE_ALARM_PROB = 1e-4
MIN_COHERENCE = 0.1
# The first order bins hold DC and slow trends, which look coherent across neighbouring frames.
FIRST_USABLE_BIN = 3
# Hann main-lobe half width in bins: a line's power is summed over +-MAIN_LOBE_BINS bins.
MAIN_LOBE_BINS = 2
# Lines are analysed only while their one-order band lies below this fraction of the anti-alias cutoff,
# clear of the cutoff filter's transition band.
USABLE_CUTOFF_FRACTION = 0.8
# A dataset line must be locked in at least this share of the recordings that can observe it.
MIN_LOCKED_SHARE = 0.5
# A line enters a recording's reference if it stands this far above the clean non-locked background
# (one-order band); weaker lines are mostly background and would set the reference to it.
MIN_INHERENT_DB = 3.0
# Frequency bins per one-order band (at the slowest frame) of the reference and its shaping filter. The
# filter's Hann taper blurs every band edge by ~2 bins, so the noise next to a much stronger line spills
# across: at 8 bins per order a weaker neighbour's band got 0.3 dB too much noise, at 32 bins 0.07 dB.
REFERENCE_BINS_PER_ORDER = 32
HARMONIC_LINES_FILE = "harmonic_lines.json"


@dataclass(frozen=True)
class HarmonicSettings:
    """How a dataset's recordings are analysed in the angle domain.

    Attributes:
        segment_revs: Frame length in shaft revolutions; the order resolution is ``1 / segment_revs``.
            Fine enough to separate the closest line families, short enough to leave tens of frame
            pairs in the shortest recordings. Orders below ``FIRST_USABLE_BIN / segment_revs`` are not
            analysed.
        cutoff_hz: Anti-alias cutoff of the angle resampling (None: 0.42 fs). Lines are analysed only
            below ``USABLE_CUTOFF_FRACTION`` of it.
        min_speed_hz: Frames that are slower than this anywhere are dropped (None: 0.9 x the 1st
            percentile of the speed while turning, clear of the label's own ripple).
    """

    segment_revs: int
    cutoff_hz: float | None = None
    min_speed_hz: float | None = None


def _lowpass(x: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase 8th-order Butterworth low-pass of the mean-free channel."""
    return ss.sosfiltfilt(ss.butter(8, cutoff_hz, fs=fs, output="sos"), x - np.mean(x))


@dataclass(frozen=True)
class AngleGrid:
    """A recording's mapping onto constant shaft-angle steps, and its analysis frames.

    Built once per recording from its IAS label and shared by all of its channels.

    Attributes:
        fs: Sampling rate [Hz].
        angle_revs: Shaft angle [revolutions] at every time sample.
        samples_per_rev: Samples per revolution of the angle grid; with ``min_speed_hz`` this keeps
            everything up to ``cutoff_hz`` alias-free in every valid frame.
        segment_revs, cutoff_hz, min_speed_hz: The resolved `HarmonicSettings`.
        frame_speed: Mean speed [Hz] of every frame.
        frame_time: Time [s] at the centre of every frame.
        valid_frames: Frames whose speed never drops below ``min_speed_hz``.
    """

    fs: float
    angle_revs: np.ndarray
    samples_per_rev: int
    segment_revs: int
    cutoff_hz: float
    min_speed_hz: float
    frame_speed: np.ndarray
    frame_time: np.ndarray
    valid_frames: np.ndarray

    @classmethod
    def from_ias(cls, ias: np.ndarray, fs: float, settings: HarmonicSettings) -> "AngleGrid":
        """Angle grid and frames of a recording with IAS label ``ias`` [Hz] sampled at ``fs``."""
        ias = np.maximum(np.asarray(ias, dtype=np.float64), 1e-9)  # strictly increasing angle for np.interp
        min_speed_hz = settings.min_speed_hz
        if min_speed_hz is None:
            # a frame is dropped if its speed dips below the minimum anywhere: stay clear of the label's ripple
            min_speed_hz = 0.9 * float(np.percentile(ias[ias > 0.05 * ias.max()], 1))
        cutoff_hz = 0.42 * fs if settings.cutoff_hz is None else settings.cutoff_hz
        # Read along time, the angle grid is sampled at samples_per_rev x speed. Everything up to the cutoff
        # stays below half of that at every usable speed if samples_per_rev >= 2 cutoff / min_speed (2.2 for
        # margin), rounded up to a power of two.
        samples_per_rev = 1 << int(np.ceil(np.log2(2.2 * cutoff_hz / min_speed_hz)))
        angle_revs = np.cumsum(ias) / fs
        speed_on_grid = np.interp(np.arange(angle_revs[0], angle_revs[-1], 1 / samples_per_rev), angle_revs, ias)
        segment_len = settings.segment_revs * samples_per_rev
        hop_len = segment_len // 2
        n_frames = (len(speed_on_grid) - segment_len) // hop_len + 1
        if n_frames < 1:
            raise ValueError(
                f"recording too short: {angle_revs[-1]:.1f} revolutions, frames of {settings.segment_revs}"
            )
        # A frame is exactly two consecutive hop blocks, so its minimum and mean speed follow from per-block
        # minima and means. The grid's tail beyond the last frame belongs to no block (a last block running to
        # the end of the grid, but divided by hop_len, overestimated the last frame's speed by up to 1.5x).
        blocks = speed_on_grid[: (n_frames + 1) * hop_len].reshape(n_frames + 1, hop_len)
        block_min, block_mean = blocks.min(axis=1), blocks.mean(axis=1)
        frame_centre = angle_revs[0] + (np.arange(n_frames) * hop_len + segment_len / 2) / samples_per_rev
        return cls(
            fs=fs, angle_revs=angle_revs, samples_per_rev=samples_per_rev, segment_revs=settings.segment_revs,
            cutoff_hz=cutoff_hz, min_speed_hz=min_speed_hz,
            frame_speed=0.5 * (block_mean[:-1] + block_mean[1:]),
            frame_time=np.interp(frame_centre, angle_revs, np.arange(len(angle_revs)) / fs),
            valid_frames=np.minimum(block_min[:-1], block_min[1:]) >= min_speed_hz,
        )  # fmt: skip

    @property
    def segment_len(self) -> int:
        """Frame length in samples of the angle grid."""
        return self.segment_revs * self.samples_per_rev

    @property
    def hop_len(self) -> int:
        """Hop between frames in samples of the angle grid (half a frame)."""
        return self.segment_len // 2

    @property
    def bin_width(self) -> float:
        """Order resolution of the frames' spectra."""
        return 1 / self.segment_revs

    def resample(self, x: np.ndarray) -> np.ndarray:
        """Channel ``x`` (aligned with the IAS) low-passed at ``cutoff_hz`` and read at constant angle steps."""
        return self._on_grid(_lowpass(x, self.fs, self.cutoff_hz))

    def _on_grid(self, x_lowpassed: np.ndarray) -> np.ndarray:
        """An already low-passed channel read at constant angle steps."""
        grid = np.arange(self.angle_revs[0], self.angle_revs[-1], 1 / self.samples_per_rev)
        return np.interp(grid, self.angle_revs, x_lowpassed)


def _segment_pair_spectra(x, segment_len, hop_len, valid_frames, sample_rate):
    """(S, D, g2, n_pairs) of ``x`` from its frame pairs, see the module docstring.

    Only pairs whose two frames are both valid count. ``sample_rate`` is the sampling rate of ``x`` along
    its axis (per revolution in the angle domain); the spectra are one-sided densities per unit of it.
    """
    window = np.hanning(segment_len + 1)[:-1]
    n_frames = (len(x) - segment_len) // hop_len + 1
    power_first = power_second = 0.0  # sums of |X|^2 over the first / second frame of every pair
    cross = 0.0  # sum of X_first * conj(X_second)
    n_pairs = 0
    # FFT the frames in chunks of ~2e7 samples to bound memory; consecutive chunks overlap by
    # PAIR_DELAY_HOPS frames, so pairs that straddle a chunk border are formed too
    chunk_frames = max(PAIR_DELAY_HOPS + 1, int(2e7 // segment_len))
    for chunk_start in range(0, n_frames, chunk_frames):
        frames = np.arange(max(chunk_start - PAIR_DELAY_HOPS, 0), min(n_frames, chunk_start + chunk_frames))
        X = np.fft.rfft(np.lib.stride_tricks.sliding_window_view(x, segment_len)[frames * hop_len] * window, axis=1)
        first, second = frames[:-PAIR_DELAY_HOPS], frames[PAIR_DELAY_HOPS:]  # pairs (frame, frame + delay)
        # both frames usable, and the pair not already counted in the previous chunk's overlap
        use = valid_frames[first] & valid_frames[second] & (second >= chunk_start)
        X_first, X_second = X[:-PAIR_DELAY_HOPS][use], X[PAIR_DELAY_HOPS:][use]
        power_first = power_first + (np.abs(X_first) ** 2).sum(axis=0)
        power_second = power_second + (np.abs(X_second) ** 2).sum(axis=0)
        cross = cross + (X_first * np.conj(X_second)).sum(axis=0)
        n_pairs += int(use.sum())
    if n_pairs == 0:
        raise ValueError("no pair of valid frames: recording too short or too slow for the angle analysis")
    # mean over the pairs, as a one-sided density: 2 / (rate * sum(window^2)) per pair
    to_density = 2 / (sample_rate * np.sum(window**2) * n_pairs)
    power_first, power_second, cross = power_first * to_density, power_second * to_density, cross * to_density
    # random content averages out of the cross spectrum, locked content does not
    return 0.5 * (power_first + power_second), np.abs(cross), np.abs(cross) ** 2 / (power_first * power_second), n_pairs


@dataclass(frozen=True)
class AngleSpectra:
    """Angle-domain spectra of one channel, per order bin (see the module docstring).

    Attributes:
        orders: Bin orders, ``0, 1/segment_revs, ...``.
        S, D, g2: Total and locked power density [per order], and their coherence.
        n_pairs: Number of frame pairs averaged.
        variance: Variance of the low-passed channel (time domain), for relative powers.
        max_order: Highest order that stays below ``USABLE_CUTOFF_FRACTION`` x cutoff at the median speed of
            the valid frames; above it the low-passed channel holds no content to judge.
    """

    orders: np.ndarray
    S: np.ndarray
    D: np.ndarray
    g2: np.ndarray
    n_pairs: int
    variance: float
    max_order: float


def _spectra_of(x_angle: np.ndarray, grid: AngleGrid, variance: float) -> AngleSpectra:
    """`AngleSpectra` of an already resampled channel."""
    S, D, g2, n_pairs = _segment_pair_spectra(
        x_angle, grid.segment_len, grid.hop_len, grid.valid_frames, grid.samples_per_rev
    )
    orders = np.fft.rfftfreq(grid.segment_len, 1 / grid.samples_per_rev)
    max_order = USABLE_CUTOFF_FRACTION * grid.cutoff_hz / np.median(grid.frame_speed[grid.valid_frames])
    return AngleSpectra(orders=orders, S=S, D=D, g2=g2, n_pairs=n_pairs, variance=variance, max_order=max_order)


def angle_spectra(x: np.ndarray, grid: AngleGrid) -> AngleSpectra:
    """Total and locked power density and coherence of channel ``x`` per order (see the module docstring)."""
    x_lowpassed = _lowpass(x, grid.fs, grid.cutoff_hz)
    return _spectra_of(grid._on_grid(x_lowpassed), grid, float(np.var(x_lowpassed)))


def coherence_threshold(n_pairs: int) -> float:
    """Coherence an order bin needs to count as locked.

    With nothing locked, the coherence of B independent pairs follows Beta(1, B - 1), i.e.
    ``P(g2 > z) = (1 - z)^(B - 1)``; the threshold is the z at which that equals ``FALSE_ALARM_PROB``.
    Neighbouring pairs overlap by half a frame; counting only half of them as independent is conservative
    (with Hann windows the products of adjacent pairs correlate only weakly), which matters only for
    recordings with few pairs. At least ``MIN_COHERENCE``.
    """
    independent_pairs = max(n_pairs / 2 - 1, 1)
    return max(1 - FALSE_ALARM_PROB ** (1 / independent_pairs), MIN_COHERENCE)


def _peak_bins(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Candidate bins that are the maximum of ``values`` within +-MAIN_LOBE_BINS, strongest claiming first.

    A candidate within one main lobe of an already accepted (stronger) peak is that peak's skirt.
    """
    peaks = []
    for i in candidates[np.argsort(-values[candidates])]:
        lo, hi = max(i - MAIN_LOBE_BINS, 0), min(i + MAIN_LOBE_BINS + 1, len(values))
        if values[i] >= values[lo:hi].max() and not any(abs(i - p) <= MAIN_LOBE_BINS for p in peaks):
            peaks.append(i)
    return np.array(peaks, dtype=int)


def _locked_bins(spectra: AngleSpectra) -> np.ndarray:
    """Bins whose coherence passes `coherence_threshold`, past the DC bins and below ``max_order``."""
    usable = (np.arange(len(spectra.orders)) >= FIRST_USABLE_BIN) & (spectra.orders < spectra.max_order)
    return usable & (spectra.g2 > coherence_threshold(spectra.n_pairs))


def _inherent_snr(
    orders: np.ndarray, S: np.ndarray, D: np.ndarray, line_bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per line bin: the non-locked background density around it and its inherent (clean) SNR [dB].

    The background is the median of ``S - D`` over the bins within +-1 order (at least +-8 bins) that lie in no
    main lobe of ``line_bins``; the SNR is the locked power in the line's main lobe over that background
    integrated over one order. NaN where no background can be measured: every bin around the line belongs to some
    line's main lobe (lines one order apart at 4 bins per order), or the background is zero.
    """
    bin_width = orders[1]
    is_line_bin = np.zeros(len(orders), bool)  # every bin inside some line's main lobe
    for i in line_bins:
        is_line_bin[max(i - MAIN_LOBE_BINS, 0) : i + MAIN_LOBE_BINS + 1] = True
    background, snr_db = np.full(len(line_bins), np.nan), np.full(len(line_bins), np.nan)
    half_ring = max(1.0, 8 * bin_width)
    for k, i in enumerate(line_bins):
        ring = np.arange(*np.searchsorted(orders, [orders[i] - half_ring, orders[i] + half_ring]))
        ring = ring[~is_line_bin[ring]]
        if not len(ring):
            continue
        bg = np.median(np.maximum(S[ring] - D[ring], 0))
        if bg > 0:
            locked = D[i - MAIN_LOBE_BINS : i + MAIN_LOBE_BINS + 1].sum() * bin_width
            background[k] = bg
            with np.errstate(divide="ignore"):  # no locked power at all: -inf dB
                snr_db[k] = 10 * np.log10(locked / bg)
    return background, snr_db


def find_lines(spectra: AngleSpectra) -> np.ndarray:
    """Orders of one recording's locked lines, strongest first.

    Local maxima of ``D`` whose coherence passes `coherence_threshold`. In a constant-speed recording this
    includes fixed-Hz content, and a fixed-Hz component that lingers at one order (where a speed sweep turns)
    looks locked there too; `dataset_lines` removes both by consensus over the dataset's recordings.
    """
    return spectra.orders[_peak_bins(spectra.D, np.flatnonzero(_locked_bins(spectra)))]


@dataclass(frozen=True)
class LineList:
    """Speed-locked lines of a dataset, strongest first.

    Attributes:
        orders: Line orders on the analysis bin grid (multiples of ``1 / segment_revs``), as the disturbance
            generator works with them.
        refined_orders: The same lines to sub-bin precision (parabola through the log median locked power at the
            peak bin and its neighbours), for estimators that track them: the grid is up to half a bin, 0.125
            order on the ball bearing, off a non-integer order such as the parallel gearbox's 95/29.
        locked_share: Share of the recordings that can observe a line in which it is locked.
        median_locked_power: Median over those recordings of the line's peak-bin locked power, relative
            to the channel's variance.
        median_snr_db: Median over those recordings of the line's inherent SNR: the locked power in its main
            lobe over the non-locked background around it, integrated over one order [dB]. What the line offers
            an estimator, where ``median_locked_power`` favours the high orders that carry most of an
            acceleration signal's power. NaN where no recording leaves background to measure it against.
        settings: The `HarmonicSettings` the lines were found with.
        splits: Splits the recordings came from.
        n_recordings: Number of channel recordings analysed.
    """

    orders: np.ndarray
    refined_orders: np.ndarray
    locked_share: np.ndarray
    median_locked_power: np.ndarray
    median_snr_db: np.ndarray
    settings: HarmonicSettings
    splits: tuple[str, ...]
    n_recordings: int

    def strongest(self, n: int) -> np.ndarray:
        """Orders of the ``n`` strongest lines."""
        return self.orders[:n]

    def highest_snr(self, n: int) -> np.ndarray:
        """Indices of the ``n`` lines that stand furthest out of their background; unmeasured ones last."""
        return np.argsort(-np.nan_to_num(self.median_snr_db, nan=-np.inf), kind="stable")[:n]

    def save(self, path: Path) -> None:
        """Write the line list as JSON."""
        out = {
            "settings": asdict(self.settings),
            "splits": list(self.splits),
            "n_recordings": self.n_recordings,
            "lines": [
                {"order": float(o), "refined_order": float(r), "locked_share": float(s), "median_locked_power": float(p),
                 "median_snr_db": float(q)}
                for o, r, s, p, q in zip(
                    self.orders, self.refined_orders, self.locked_share, self.median_locked_power, self.median_snr_db
                )
            ],
        }
        Path(path).write_text(json.dumps(out, indent=1))

    @classmethod
    def load(cls, path: Path) -> "LineList":
        """Read a line list written by `save` (older files: refined orders = orders, NaN SNR)."""
        d = json.loads(Path(path).read_text())
        return cls(
            orders=np.array([line["order"] for line in d["lines"]]),
            refined_orders=np.array([line.get("refined_order", line["order"]) for line in d["lines"]]),
            locked_share=np.array([line["locked_share"] for line in d["lines"]]),
            median_locked_power=np.array([line["median_locked_power"] for line in d["lines"]]),
            median_snr_db=np.array([line.get("median_snr_db", np.nan) for line in d["lines"]]),
            settings=HarmonicSettings(**d["settings"]),
            splits=tuple(d["splits"]),
            n_recordings=d["n_recordings"],
        )


def _recording_votes(
    file: Path, channels: list[str], settings: HarmonicSettings
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """One recording's vote per channel over the bins it can observe: (locked, peak-bin power / variance, S, D).

    S and D let `dataset_lines` measure each line's inherent SNR once the lines are known. Module-level so that a
    spawned worker process can import it. Returns nothing for a recording too short or too slow for the analysis.
    """
    with h5py.File(file, "r") as f:
        fs = float(f.attrs["fs"])
        try:
            grid = AngleGrid.from_ias(f["IAS"][:], fs, settings)
        except ValueError:
            return []
        votes = []
        for channel in channels:
            try:
                spec = angle_spectra(f[channel][:].astype(np.float64), grid)
            except ValueError:
                continue
            # tolerate the line one bin off (label scale error, slip)
            locked = np.convolve(_locked_bins(spec), np.ones(3), mode="same") > 0
            locked &= np.arange(len(locked)) >= FIRST_USABLE_BIN
            n_observable = int(np.sum(spec.orders < spec.max_order))  # the bins this recording votes on
            votes.append((
                locked[:n_observable],
                (spec.D[:n_observable] * grid.bin_width / spec.variance).astype(np.float32),
                spec.S[:n_observable].astype(np.float32),
                spec.D[:n_observable].astype(np.float32),
            ))  # fmt: skip
    return votes


def dataset_lines(
    dataset_path: Path,
    channels: list[str],
    settings: HarmonicSettings,
    splits: tuple[str, ...] = ("train", "valid", "test"),
) -> LineList:
    """Speed-locked lines of a prepared dataset, by consensus over its recordings.

    Every channel of every recording in ``splits`` votes on the order bins it can observe (below
    ``USABLE_CUTOFF_FRACTION`` x cutoff at its median speed): locked if the bin's (or a neighbour's)
    coherence passes `coherence_threshold`. Lines are the local maxima of the median locked power among the
    bins locked in at least ``MIN_LOCKED_SHARE`` of their voters (and voted on by at least 3). A kinematic
    order sits at the same order in every recording, a fixed-Hz component lands at a different one at each
    speed, so constant-speed recordings cannot smuggle fixed-Hz content in. Each line also gets its median
    inherent SNR over the recordings that can observe it (see `LineList.median_snr_db`).

    The recordings are analysed in ``preparation_workers`` processes, which are spawned: called from a
    script, the script needs an ``if __name__ == "__main__":`` guard. For lines handed to an estimation
    method as prior knowledge, use only the splits it may learn from (e.g. ``("train",)``).

    Args:
        dataset_path: Prepared dataset directory with one sub-directory per split.
        channels: Vibration channels to analyse.
        settings: Angle-domain analysis settings of the dataset.
        splits: Splits to take the recordings from; missing ones are skipped.
    """
    files = [
        f for split in splits if (dataset_path / split).is_dir() for f in hdf_files_from_path(dataset_path / split)
    ]
    n_workers = preparation_workers(len(files))
    if n_workers == 1:
        per_file = [_recording_votes(f, channels, settings) for f in files]
    else:
        # "spawn": workers start clean instead of inheriting the caller's threads and open handles
        with ProcessPoolExecutor(n_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            per_file = list(pool.map(_recording_votes, files, [channels] * len(files), [settings] * len(files)))
    votes = [v for file_votes in per_file for v in file_votes]
    if not votes:
        raise ValueError(f"no recording under {dataset_path} {splits} could be analysed")
    n_bins = max(len(locked) for locked, *_ in votes)
    locked_votes, n_voting = np.zeros(n_bins), np.zeros(n_bins)
    power_rows = np.full((len(votes), n_bins), np.nan, dtype=np.float32)  # NaN where a recording does not vote
    for row, (locked, power, _, _) in enumerate(votes):
        locked_votes[: len(locked)] += locked
        n_voting[: len(locked)] += 1
        power_rows[row, : len(power)] = power
    locked_share = np.where(n_voting >= 3, locked_votes / np.maximum(n_voting, 1), 0.0)
    median_power = np.zeros(n_bins)
    voted = n_voting > 0
    median_power[voted] = np.nanmedian(power_rows[:, voted], axis=0)
    lines = _peak_bins(median_power, np.flatnonzero(locked_share >= MIN_LOCKED_SHARE))  # strongest first
    # sub-bin orders: vertex of the parabola through the log median power at the peak bin and its neighbours
    refined = lines.astype(np.float64)
    for k, b in enumerate(lines):
        if b + 1 < n_bins and (median_power[b - 1 : b + 2] > 0).all():
            below, peak, above = np.log(median_power[b - 1 : b + 2])
            if below - 2 * peak + above < 0:
                refined[k] += 0.5 * (below - above) / (below - 2 * peak + above)

    # inherent SNR of every line in every recording that can observe it, with the background ring excluding the
    # main lobes of all lines that recording observes (as `HarmonicReference` does)
    snr_rows = np.full((len(votes), len(lines)), np.nan)
    for row, (_, _, S, D) in enumerate(votes):
        observed = lines < len(S)
        if observed.any():
            orders = np.arange(len(S)) / settings.segment_revs
            snr = _inherent_snr(orders, S, D, lines[observed])[1]
            snr[lines[observed] + MAIN_LOBE_BINS >= len(S)] = np.nan  # main lobe cut off at the observable edge
            snr_rows[row, observed] = snr
    median_snr = np.full(len(lines), np.nan)  # NaN: no recording leaves background to measure it against
    for k in range(len(lines)):
        measured = snr_rows[~np.isnan(snr_rows[:, k]), k]
        if len(measured):
            median_snr[k] = np.median(measured)
    return LineList(
        orders=lines / settings.segment_revs, refined_orders=refined / settings.segment_revs,
        locked_share=locked_share[lines], median_locked_power=median_power[lines],
        median_snr_db=median_snr, settings=settings, splits=tuple(splits), n_recordings=len(votes),
    )  # fmt: skip


def _frame_band_power(x_angle, grid, bins_per_band, frames):
    """(frames, bands): per frame, the one-sided power in the bins of each row of ``bins_per_band``."""
    window = np.hanning(grid.segment_len + 1)[:-1]
    to_power = 2 / (grid.samples_per_rev * np.sum(window**2)) * grid.bin_width  # density x bin width
    out = np.zeros((len(frames), len(bins_per_band)))
    chunk_frames = max(1, int(2e7 // grid.segment_len))  # bound the memory of the FFT
    windows = np.lib.stride_tricks.sliding_window_view(x_angle, grid.segment_len)
    for j0 in range(0, len(frames), chunk_frames):
        chunk = frames[j0 : j0 + chunk_frames]
        X = np.fft.rfft(windows[chunk * grid.hop_len] * window, axis=1)
        out[j0 : j0 + chunk_frames] = (np.abs(X[:, bins_per_band]) ** 2).sum(axis=2) * to_power
    return out


@dataclass(frozen=True)
class LineSNR:
    """Per-line SNR of a disturbed recording, per frame and in the long term.

    The SNR of line k in frame b is its clean power ``line_power[b, k]`` over the noise power inside its
    one-order band ``(o_k +- 0.5) x frame_speed[b]`` (one revolution of data). The *effective* SNR also
    counts the clean recording's own non-locked background in that band.

    The noise power is measured in the frames' spectra, whose Hann main lobe spans +-2 bins: a band next to a
    much stronger neighbour picks up some of the neighbour's noise. Synthetic check, neighbours 6 dB apart:
    ~1 dB too much noise with 4-revolution frames (0.25-order bins), ~0.2 dB with 32.

    Attributes:
        orders: Line orders.
        frame_time, frame_speed, frame_duration: Centre time [s], mean speed [Hz] and duration [s] of the
            hop each valid frame stands for.
        line_power, noise_power: (frames, lines), NaN where a line's band is above the analysed range.
        background: Clean non-locked background power in each line's one-order band (time average).
    """

    orders: np.ndarray
    frame_time: np.ndarray
    frame_speed: np.ndarray
    frame_duration: np.ndarray
    line_power: np.ndarray
    noise_power: np.ndarray
    background: np.ndarray

    def snr_db(self, effective: bool = False) -> np.ndarray:
        """(frames, lines) per-frame SNR [dB]; -inf where a line is below the background in that frame."""
        noise = self.noise_power + (self.background[None, :] if effective else 0.0)
        with np.errstate(divide="ignore"):
            return 10 * np.log10(self.line_power / noise)

    def long_term_snr_db(self, effective: bool = False) -> np.ndarray:
        """(lines,) time-weighted SNR [dB]: sum of duration x line power over sum of duration x noise power."""
        noise = self.noise_power + (self.background[None, :] if effective else 0.0)
        w = np.where(np.isfinite(self.line_power), self.frame_duration[:, None], 0.0)
        with np.errstate(divide="ignore"):  # -inf for a line that is below the background in every frame
            return 10 * np.log10(np.nansum(w * self.line_power, axis=0) / np.nansum(w * noise, axis=0))

    def in_band(self, max_freq_hz: float) -> np.ndarray:
        """(frames, lines) whether a line's one-order band lies below ``max_freq_hz`` in that frame."""
        return ((self.orders[None, :] + 0.5) * self.frame_speed[:, None] < max_freq_hz) & np.isfinite(self.line_power)

    def strongest_line(self, max_freq_hz: float | None = None) -> int:
        """Index of the line with the most clean power (time-weighted) while it lies below ``max_freq_hz``.

        For a method that keeps ``[0, f_max]``, ``strongest_line(f_max)`` is the line whose SNR describes
        what that method sees; mask the frames where it is out of band with `in_band`.
        """
        inside = np.isfinite(self.line_power) if max_freq_hz is None else self.in_band(max_freq_hz)
        weighted = np.where(inside, self.frame_duration[:, None] * self.line_power, 0.0).sum(axis=0)
        if not weighted.any():
            raise ValueError(f"no line lies below {max_freq_hz} Hz")
        return int(np.argmax(weighted))


@dataclass(frozen=True)
class HarmonicReference:
    """Reference spectrum of one channel, built from its speed-locked lines.

    In every frame each line's power is spread over +-0.5 orders at the frame's speed; per frequency the
    reference is the mean over the frames in which some line covers it (a plain time mean would dilute
    lines that sweep through), log-interpolated across frequencies no line ever reaches and flat beyond the
    outermost lines. Noise with this spectrum carries, in every line's one-order band, as much power as the
    lines that typically sit there.

    Attributes:
        grid: The recording's `AngleGrid`.
        spectra: The clean channel's `AngleSpectra`.
        freqs: Uniform frequency grid [Hz], 0 .. fs/2; a one-order band spans >= REFERENCE_BINS_PER_ORDER bins
            at the slowest frame.
        density: The reference [power/Hz] on ``freqs``.
        occupied: Frequencies some line's band covered in at least one frame (the rest is interpolated).
        orders: Orders of the lines in the reference.
        inherent_db: Their clean SNR: locked power over the non-locked background in a one-order band [dB].
        background: That background's power in each line's one-order band.
        line_power: (frames, lines) clean line power per valid frame, NaN where the line's band lies above
            the analysed range.
        frames: Indices of the valid frames in ``grid``.
    """

    grid: AngleGrid
    spectra: AngleSpectra
    freqs: np.ndarray
    density: np.ndarray
    occupied: np.ndarray
    orders: np.ndarray
    inherent_db: np.ndarray
    background: np.ndarray
    line_power: np.ndarray
    frames: np.ndarray

    @classmethod
    def from_signal(
        cls, x: np.ndarray, grid: AngleGrid, line_orders: np.ndarray, min_inherent_db: float = MIN_INHERENT_DB
    ) -> "HarmonicReference":
        """Reference of clean channel ``x`` (aligned with ``grid``'s IAS) from the dataset's ``line_orders``.

        Raises:
            ValueError: If no line stands ``min_inherent_db`` above the clean background, or the recording
                is too short or too slow for the angle analysis.
        """
        fs = grid.fs
        x_lowpassed = _lowpass(x, fs, grid.cutoff_hz)
        x_angle = grid._on_grid(x_lowpassed)
        spectra = _spectra_of(x_angle, grid, float(np.var(x_lowpassed)))
        orders, S, D, bin_width = spectra.orders, spectra.S, spectra.D, grid.bin_width
        frames = np.flatnonzero(grid.valid_frames)
        frame_speed = grid.frame_speed[frames]
        usable_hz = USABLE_CUTOFF_FRACTION * grid.cutoff_hz

        # the dataset's lines on this recording's bin grid, past the DC bins and with their one-order band
        # below the usable cutoff at least in the slowest frame
        line_bins = np.unique(np.round(np.asarray(line_orders) / bin_width).astype(int))
        line_bins = line_bins[
            (line_bins >= FIRST_USABLE_BIN)
            & ((orders[np.minimum(line_bins, len(orders) - 1)] + 0.5) * frame_speed.min() < usable_hz)
        ]
        background_density, inherent_db = _inherent_snr(orders, S, D, line_bins)
        keep = inherent_db >= min_inherent_db  # NaN (no background to measure) is never kept
        if not keep.any():
            raise ValueError("no line stands out of the clean background: nothing to reference the disturbances to")
        kept_bins, background_density, inherent_db = line_bins[keep], background_density[keep], inherent_db[keep]
        kept_orders = orders[kept_bins]

        # line power per frame: the main lobe minus the background inside it. Linear interpolation onto the
        # angle grid attenuates by sinc^2(f/fs) in amplitude, sinc^4 in power, which is undone per frame.
        lobe_bins = kept_bins[:, None] + np.arange(-MAIN_LOBE_BINS, MAIN_LOBE_BINS + 1)[None, :]
        lobe_power = _frame_band_power(x_angle, grid, lobe_bins, frames)
        interp_loss = np.sinc(kept_orders[None, :] * frame_speed[:, None] / fs) ** 4
        line_power = (
            np.maximum(lobe_power - background_density[None, :] * (2 * MAIN_LOBE_BINS + 1) * bin_width, 0) / interp_loss
        )
        line_power[(kept_orders[None, :] + 0.5) * frame_speed[:, None] >= usable_hz] = np.nan  # band out of range
        frame_duration = (grid.hop_len / grid.samples_per_rev) / frame_speed  # one hop of revolutions, in s

        # In a frame of speed w line k covers [(o_k - 0.5) w, (o_k + 0.5) w) Hz with density P_k / w (its power
        # spread over one order). Per frequency bin, over all frames, time-weighted: the summed density of the
        # bands covering it, and the time during which any band covers it. Difference arrays (+value at a
        # band's first bin, -value at its end) turn the band edges into per-bin sums by a cumulative sum.
        n_fft = min(1 << int(np.ceil(np.log2(REFERENCE_BINS_PER_ORDER * fs / frame_speed.min()))), 1 << 20)
        freqs = np.fft.rfftfreq(n_fft, 1 / fs)
        df = freqs[1]
        band_first = np.clip(
            np.ceil((kept_orders[None, :] - 0.5) * frame_speed[:, None] / df).astype(int), 0, len(freqs)
        )
        band_end = np.clip(np.ceil((kept_orders[None, :] + 0.5) * frame_speed[:, None] / df).astype(int), 0, len(freqs))
        line_density = line_power / frame_speed[:, None]
        usable = np.isfinite(line_density)  # (frames, lines)

        def per_bin(first, end, weights=None):
            """Per frequency bin, the summed weight of the bands [first, end) that cover it."""
            steps = np.bincount(first.ravel(), weights, len(freqs) + 1) - np.bincount(
                end.ravel(), weights, len(freqs) + 1
            )
            return np.cumsum(steps)[:-1]

        weighted_density = np.where(usable, frame_duration[:, None] * line_density, 0.0)
        power_sum = per_bin(band_first, band_end, weighted_density.ravel())
        # The time during which *any* band covers a bin needs each frame's union of bands. The orders are sorted,
        # so the part of band k that no lower band covers yet starts at the furthest band end reached so far;
        # these parts are disjoint, and their difference arrays add up to the union.
        band_end = np.where(usable, band_end, 0)
        reached = np.maximum.accumulate(np.pad(band_end[:, :-1], ((0, 0), (1, 0))), axis=1)
        union_first = np.maximum(band_first, reached)
        union_end = np.maximum(band_end, union_first)
        occupied_time = per_bin(union_first, union_end, np.repeat(frame_duration, len(kept_orders)))
        # Occupancy from exact integer counts: where bands end, the float sums cancel to ~1e-17 rather than 0, and
        # a bin covered only by lines at zero power must not enter the reference with such a residue (it punched
        # 80-140 dB holes into the reference of some recordings, and the gap fill spread them).
        positive = weighted_density > 0
        occupied = (per_bin(union_first, union_end) > 0) & (per_bin(band_first[positive], band_end[positive]) > 0)
        occupied &= power_sum > 0  # a vanishing positive density can still lose to the residue
        if not occupied.any():
            raise ValueError("the lines carry no power in any frame: nothing to reference the disturbances to")

        # Mean while occupied, log-log interpolated across the gaps. Beyond the outermost lines the reference
        # continues flat at their typical level, their time-weighted mean density: the extreme occupied bins are
        # often reached by a single frame, so their own value is noisy (2-6 dB off, at worst 35 dB).
        conditional = power_sum[occupied] / occupied_time[occupied]
        mean_line_density = weighted_density.sum(axis=0) / np.where(usable, frame_duration[:, None], 0.0).sum(axis=0)
        with_power = np.flatnonzero(mean_line_density > 0)  # lines sorted by order
        log_freqs = np.log(np.maximum(freqs, df / 2))
        density = np.exp(
            np.interp(
                log_freqs,
                log_freqs[occupied],
                np.log(conditional),
                left=np.log(mean_line_density[with_power[0]]),
                right=np.log(mean_line_density[with_power[-1]]),
            )
        )
        return cls(
            grid=grid, spectra=spectra, freqs=freqs, density=density, occupied=occupied, orders=kept_orders,
            inherent_db=np.array(inherent_db), line_power=line_power, frames=frames,
            # background power in one order, with the interpolation loss undone (on average over the frames)
            background=background_density / np.mean(interp_loss, axis=0),
        )  # fmt: skip

    @cached_property
    def total_power(self) -> float:
        """Integral of the reference over 0 .. fs/2."""
        return float(np.sum(self.density) * self.freqs[1])

    @cached_property
    def fir(self) -> np.ndarray:
        """Zero-phase FIR whose magnitude response is the square root of the reference."""
        n_taps = 2 * (len(self.freqs) - 1)
        # the inverse real FFT of a real, positive magnitude is a symmetric response centred on tap 0;
        # fftshift centres it in the tap array, the Hann window tapers its ends
        taps = np.fft.fftshift(np.fft.irfft(np.sqrt(self.density), n=n_taps))
        return taps * ss.windows.hann(n_taps, sym=False)

    def line_band_power(self, sig: np.ndarray) -> np.ndarray:
        """(frames, lines) power of ``sig`` inside each line's one-order band per valid frame.

        ``sig`` must be aligned with the channel the reference was built from (same length and IAS), e.g.
        the noise added to it. NaN where the line's band lies above the analysed range.
        """
        grid = self.grid
        half_band = round(0.5 / grid.bin_width)  # bins on each side of the line: +-0.5 orders
        band_bins = (
            np.round(self.orders / grid.bin_width).astype(int)[:, None] + np.arange(-half_band, half_band + 1)[None, :]
        )
        band_power = _frame_band_power(grid.resample(sig), grid, band_bins, self.frames)
        # the bins span 2 half_band + 1 bin widths, slightly more than one order: normalise to exactly one
        band_power /= (2 * half_band + 1) * grid.bin_width
        band_power /= np.sinc(self.orders[None, :] * grid.frame_speed[self.frames, None] / grid.fs) ** 4
        band_power[np.isnan(self.line_power)] = np.nan
        return band_power

    def line_snr(self, noise: np.ndarray) -> LineSNR:
        """Per-line SNR of the clean channel against ``noise`` (e.g. disturbed minus clean)."""
        frame_speed = self.grid.frame_speed[self.frames]
        return LineSNR(
            orders=self.orders, frame_time=self.grid.frame_time[self.frames], frame_speed=frame_speed,
            frame_duration=(self.grid.hop_len / self.grid.samples_per_rev) / frame_speed,
            line_power=self.line_power, noise_power=self.line_band_power(noise), background=self.background,
        )  # fmt: skip


def line_snr(
    clean: np.ndarray,
    disturbed: np.ndarray,
    ias: np.ndarray,
    fs: float,
    line_orders: np.ndarray,
    settings: HarmonicSettings,
) -> LineSNR:
    """Per-line SNR of a disturbed copy of a recording, per frame and in the long term.

    Args:
        clean: Clean channel.
        disturbed: The same channel with disturbances (same length).
        ias: IAS label [Hz] of the recording.
        fs: Sampling rate [Hz].
        line_orders: The dataset's line orders (`dataset_lines`, `load_dataset_lines`).
        settings: The dataset's `HarmonicSettings`.
    """
    reference = HarmonicReference.from_signal(clean, AngleGrid.from_ias(ias, fs, settings), line_orders)
    return reference.line_snr(np.asarray(disturbed, dtype=np.float64) - clean)


def load_dataset_lines(dataset_path: Path) -> LineList:
    """The line list written next to a prepared IAS dataset (all splits; see `dataset_lines`)."""
    return LineList.load(Path(dataset_path) / HARMONIC_LINES_FILE)
