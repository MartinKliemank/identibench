"""Gas foil bearing IAS estimation dataset (TU Berlin, pre-converted HDF5)."""

__all__ = [
    "gas_foil_bearing_dataset",
    "dl_gas_foil_bearing",
    "BenchmarkGasFoilBearing_Estimation",
    "BenchmarkGasFoilBearing_Simulation",
]

import re
import shutil
import tempfile
from pathlib import Path

import h5py
from tqdm import tqdm

from ...benchmark import BenchmarkSpec, Simulation, WindowedEstimation, GridwiseEstimation
from ...dataset import Dataset
from ...metrics import mae
from ._common import (
    DatasetInfo,
    _require_sklearn,
    download_and_unpack,
    ias_test_sets,
    write_disturbed_test_sets,
)
from .harmonics import HarmonicSettings

# Angle-domain analysis of the disturbed test sets (`harmonics`): 8-revolution frames (0.125-order
# resolution, enough for 1x-6x and sub-synchronous orders); frames slower than 60 Hz
# (start-up and run-out) are left out, which keeps the angle grid at 512 samples per revolution.
HARMONIC_SETTINGS = HarmonicSettings(segment_revs=8, cutoff_hz=10000.0, min_speed_hz=60.0)

_INFO = DatasetInfo(
    name="Gas_Foil_Bearing",
    zip_url="https://tubcloud.tu-berlin.de/s/G8oFpozTW64cs3X/download",
)
# e.g. Load1_1_UpDown11s_03.hdf5  or  Load2_1_2_Random2_01.hdf5
_FILENAME_PATTERN = re.compile(r"^(Load\d+)_(\d+(?:_\d+)*)_([A-Za-z][A-Za-z0-9]*)_(\d{2})(\.hdf5)$")


# Highest frequency the label retains. Unlike the other three this is not an order cutoff: the
# IAS comes from an analog RPM channel that the upstream TDMS conversion low-pass filters in the
# time domain (zero-phase 4th-order Butterworth, -3 dB at 40 Hz).
_IAS_BANDWIDTH_HZ = 40.0


def _copy_with_idb_attrs(src: Path, dest_dir: Path) -> None:
    """Copy one pre-converted file, renaming ``sampling_rate`` → ``fs``, synthesizing
    ``gear_ratio=1`` (no gearing; kept for cross-dataset consistency) and recording the
    label bandwidth the other IAS datasets carry."""
    dest = dest_dir / src.name
    shutil.copy(str(src), str(dest))
    with h5py.File(dest, "r+") as f:
        if "sampling_rate" in f.attrs:
            f.attrs["fs"] = float(f.attrs["sampling_rate"])
            del f.attrs["sampling_rate"]
        f.attrs["gear_ratio"] = 1
        f.attrs["ias_bandwidth"] = float(_IAS_BANDWIDTH_HZ)
        f.attrs["ias_bandwidth_units"] = "Hz"


def dl_gas_foil_bearing(
    save_path: Path,  # directory the files are written to, created if it does not exist
    force_download: bool = False,  # unused; the framework only calls this when the dataset is missing or forced
) -> None:
    """Download the pre-converted HDF5 archive, split, and add disturbed test sets.

    The archive is already HDF5 (TDMS conversion happened upstream), so no
    signal preprocessing is needed. There is no wear condition for this dataset.
    Upstream, every recording was trimmed to its rotating span plus at most 1.5 s of
    standstill at either end; ``attrs["t_start"]`` is the time of its first kept sample
    in the original recording, in s.
    """
    train_test_split = _require_sklearn()
    save_path = Path(save_path)
    for split in ("train", "valid", "test"):
        (save_path / split).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        download_and_unpack(_INFO, temp_dir)

        parsed_files = []
        for file in tqdm(sorted(temp_dir.rglob("*.hdf5")), desc="Processing Gas_Foil_Bearing files"):
            match = _FILENAME_PATTERN.match(file.name)
            if not match:
                print(f"Warning: Could not parse filename: {file.name}")
                continue
            load_str = match.group(1)  # e.g. "Load1"
            trajectory = match.group(3)  # e.g. "UpDown11s"
            if "anual" in trajectory:
                continue  # skip manual recordings
            parsed_files.append({"full_path": file, "trajectory": trajectory, "load_str": load_str})

        file_list = [f["full_path"] for f in parsed_files]
        strata_labels = []
        for f in parsed_files:
            if "Hold" in f["trajectory"]:
                strata_labels.append(f"Hold_{f['load_str']}")
            elif "Random" in f["trajectory"]:
                strata_labels.append(f["trajectory"])
            else:
                strata_labels.append(f"{f['trajectory']}_{f['load_str']}")

        train_valid_paths, test_basic_paths, train_valid_labels, _ = train_test_split(
            file_list, strata_labels, test_size=0.20, random_state=42, stratify=strata_labels
        )
        train_paths, valid_paths, _, _ = train_test_split(
            train_valid_paths, train_valid_labels, test_size=0.25, random_state=42, stratify=train_valid_labels
        )

        for paths, subdir in [(train_paths, "train"), (valid_paths, "valid"), (test_basic_paths, "test")]:
            for path in tqdm(paths, desc=f"Copying {subdir} files"):
                _copy_with_idb_attrs(path, save_path / subdir)

    write_disturbed_test_sets(save_path, vib_keys=["Acc_x", "Acc_y"], settings=HARMONIC_SETTINGS)


# version 2: the upstream pre-converted archive behind `_INFO.zip_url` was replaced. This
# dataset's own preprocessing is unchanged -- its IAS comes from an analog RPM channel rather
# than a pulse train, so the order-domain reference the other three moved to does not apply.
# versions 3-4: label unchanged; disturbed test sets rebuilt (clipped Levy impulsive component).
# version 5: removed bernoulli noise from disturbed test sets.
# version 8: lower lowpass filter
# version 9: disturbed test sets rebuilt: the harmonic interferer runs at another test file's
# speed profile (was tied to this file's own IAS), and the SNR is defined per 1/12-octave band, with the noise
# shaped like the signal's spectrum (was white noise against the total power).
# version 10: disturbed test sets referenced to the speed-locked harmonics (`harmonics`): every line
# sits S dB above the noise in its one-order band (was: the clean spectrum per 1/12 octave).
# version 12: new upstream archive: label low-pass -3 dB at 40 Hz (was -6 dB at 40 Hz), freeze bridging at
# the converter's gate switches reworked, standstill trimmed to at most 1.5 s at each end (attrs["t_start"]).
gas_foil_bearing_dataset = Dataset("gas_foil_bearing", prepare=dl_gas_foil_bearing, version="12")

_gas_foil_bearing = dict(
    u_cols=["Acc_x", "Acc_y"],
    y_cols=["IAS"],
    train=[(gas_foil_bearing_dataset, "train/*.hdf5")],
    valid=[(gas_foil_bearing_dataset, "valid/*.hdf5")],
    # This dataset has no `wear` condition, so no such set is declared.
    test_sets=ias_test_sets(gas_foil_bearing_dataset, wear=False),
)

BenchmarkGasFoilBearing_Estimation = BenchmarkSpec(
    name="BenchmarkGasFoilBearing_Estimation",
    # window_sec = largest window any upstream method needed (MOPA 2.49 s — the order-tracking
    # methods decimate this fast signal), rounded to 2.5 s; smaller methods crop. See ias/__init__.
    task=WindowedEstimation(window_sec=2.5),
    **_gas_foil_bearing,
)

BenchmarkGasFoilBearing_GridwiseEstimation = BenchmarkSpec(
    name="BenchmarkGasFoilBearing_GridwiseEstimation",
    # window_sec=3.0: the largest single window across every upstream method's search space
    # over all four IAS datasets (unlike the per-dataset WindowedEstimation windows above,
    # this one is kept uniform — it's only a context guarantee, not a tuned averaging window).
    # The first/last window_sec / 2 = 1.5 s of a file are never scored; the upstream files keep 1.5 s of
    # standstill at each end for exactly this, so raising window_sec would drop run-up/run-down samples.
    # step_sec: Nyquist for the label's retained band, _IAS_BANDWIDTH_HZ = 40.0 Hz -> <12.5 ms,
    # taken at 3 ms to match the other fast datasets (~4x margin).
    task=GridwiseEstimation(window_sec=3.0, step_sec=0.003),
    **_gas_foil_bearing,
)

# Dense free-run sibling (framework Simulation task): the model predicts one IAS estimate
# per sample over the full recording — the window lives in the model (e.g. a sliding window),
# not the benchmark — scored per-sample MAE in Hz.
BenchmarkGasFoilBearing_Simulation = BenchmarkSpec(
    name="BenchmarkGasFoilBearing_Simulation",
    task=Simulation(metric=mae),
    **_gas_foil_bearing,
)
