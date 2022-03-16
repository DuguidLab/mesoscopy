"""
Script to preprocess mesoscale imaging data, one at a time.
Preprocessing separates the two channels, applies the haemodynamic correction,
realigns the images to a common coordinate space (ABA) and extracts the delta F signal.
"""
import os
import argparse
import pathlib
import tables
import matplotlib

import numpy as np
import scipy.ndimage as ndi

matplotlib.use("Agg")
from matplotlib import pyplot as plt

matplotlib.rcParams["figure.dpi"] = 150

from timeit import default_timer as timer  # Debugging performance


def main():
    parser = argparse.ArgumentParser(description="Preprocess mesoscope data")
    parser.add_argument("path", type=str, help="Path to recording file.")
    # parser.add_argument(
    #     "--interim",
    #     type=pathlib.Path,
    #     help="Directory for storing interim QA files. Defaults to /tmp.",
    #     default="/tmp/",
    # )
    # parser.add_argument(
    #     "--output",
    #     type=pathlib.Path,
    #     help="Output directory. Defaults to local ./out from wherever the script was called",
    #     default="./out",
    # )

    reg_parser = parser.add_argument_group(
        "registration", "Image registration to the ABA Common Coordinate Framework."
    )
    reg_parser.add_argument(
        "--registration",
        choices={"auto", "manual"},
        default="manual",
        help="Registration method for ABA.",
    )
    reg_parser.add_argument(
        "--x",
        type=int,
        help="x-axis displacement for manual registration. Defaults to 0.",
        default=0,
    )
    reg_parser.add_argument(
        "--y",
        type=int,
        help="y-axis displacement for manual registration. Defaults to 0.",
        default=0,
    )
    reg_parser.add_argument(
        "--rot",
        type=int,
        help="Rotation for manual registration in degrees. Defaults to 0.",
        default=0,
    )
    args = parser.parse_args()

    animal_id = args.path.split("/")[-1].replace(".h5", "")

    print("Working on {}".format(args.path))
    raw = tables.open_file(args.path, "r")

    interim_path = args.path.replace("raw", "interim")
    os.makedirs(interim_path.replace(".h5", ""), exist_ok=True)
    interim_hdf5 = tables.open_file(interim_path, "w", title=animal_id)
    print("Interim HDF5 file at {}".format(interim_path))

    start = timer()
    frame_means = np.mean(raw.root.frames, axis=(1, 2))
    end = timer()
    print("Frame means took {} s".format(end - start))

    plt.clf()
    plt.hist(frame_means)
    outpath = args.path.replace("raw", "interim").replace(
        ".h5", "_qa_100-histogram.png"
    )
    os.makedirs(outpath.replace(outpath.split("/")[-1], ""), exist_ok=True)
    plt.savefig(outpath)
    print("Saved histogram for first 100 frames at {}".format(outpath))

    print("Separating channels...")
    start = timer()
    channel_threshold = frame_means.mean()
    end = timer()
    print("Calculating channel threshold took {} s".format(end - start))

    gcamp_frames = interim_hdf5.create_earray(
        interim_hdf5.root,
        "gcamp",
        tables.UInt8Atom(),
        shape=(0, raw.root.frames.shape[1], raw.root.frames.shape[2]),
        expectedrows=raw.root.frames.shape[0] / 2,
    )
    isosb_frames = interim_hdf5.create_earray(
        interim_hdf5.root,
        "isosb",
        tables.UInt8Atom(),
        shape=(0, raw.root.frames.shape[1], raw.root.frames.shape[2]),
        expectedrows=raw.root.frames.shape[0] / 2,
    )

    start = timer()
    gcamp_timestamps = []
    for idx, frame in enumerate(raw.root.frames.iterrows()):
        if np.mean(frame, axis=(0, 1)) > channel_threshold:
            # if (idx % 2) != 0:
            isosb_frames.append([frame])
        else:
            gcamp_frames.append([frame])
            gcamp_timestamps.append(raw.root.timestamps[idx])
    end = timer()
    print("Separating channels took {} s".format(end - start))

    print("Generating average isosbestic frame...")

    start = timer()
    isosb_mean = np.mean(isosb_frames[:7500], axis=0)
    end = timer()
    print("Calculating mean isosb frame took {} s".format(end - start))

    isosb_frames.remove()
    interim_hdf5.flush()

    outpath = args.path.replace("raw", "interim").replace(".h5", "_qa_isosb-mean.png")
    plt.imsave(outpath, isosb_mean)
    print("Saved mean isosbestic image at {}".format(outpath))

    print("Correcting for haemoglobin effects using isosbestic average...")
    hb_corrected = interim_hdf5.create_earray(
        interim_hdf5.root,
        "hb_corrected",
        tables.Float32Atom(),
        shape=(0, raw.root.frames.shape[1], raw.root.frames.shape[2]),
        expectedrows=raw.root.frames.shape[0] / 2,
    )

    start = timer()
    for frame in gcamp_frames.iterrows():
        hb_corrected.append([frame / isosb_mean])
    end = timer()
    print("frame / isosb_mean took {} s".format(end - start))

    print("Corrected array has shape {}".format(hb_corrected.shape))

    outpath = args.path.replace("raw", "interim").replace(
        ".h5", "_qa_haem-corr-sample-frame.png"
    )
    plt.imsave(outpath, hb_corrected[100])
    print("Saved sample corrected frame at {}".format(outpath))

    print("Filtering with a 3x3 Gaussian...")
    filtered = interim_hdf5.create_earray(
        interim_hdf5.root,
        "filtered",
        tables.Float32Atom(),
        shape=(0, raw.root.frames.shape[1], raw.root.frames.shape[2]),
        expectedrows=raw.root.frames.shape[0] / 2,
    )

    start = timer()
    for frame in hb_corrected.iterrows():
        filtered.append([ndi.gaussian_filter(frame, 3)])
    end = timer()
    print("Filtering took {} s".format(end - start))

    outpath = args.path.replace("raw", "interim").replace(
        ".h5", "_gaussian-filter-sample-frame.png"
    )
    plt.imsave(outpath, filtered[100])
    print("Saved sample filtered frame at {}".format(outpath))

    hb_corrected.remove()
    interim_hdf5.flush()

    print("Realigning to common bregma...")
    x_shift = args.x
    y_shift = args.y
    rot_shift = args.rot

    realigned = interim_hdf5.create_earray(
        interim_hdf5.root,
        "realigned",
        tables.Float32Atom(),
        shape=(0, raw.root.frames.shape[1], raw.root.frames.shape[2]),
        expectedrows=raw.root.frames.shape[0] / 2,
    )

    start = timer()
    for frame in filtered.iterrows():
        realigned.append(
            [
                ndi.shift(
                    ndi.rotate(frame, rot_shift, axes=(0, 1), reshape=False),
                    (y_shift, x_shift),
                )
            ]
        )
    end = timer()
    print("Realigning took {} s".format(end - start))

    outpath = args.path.replace("raw", "interim").replace(
        ".h5", "_realigned-sample-frame.png"
    )
    plt.imsave(outpath, realigned[100])
    print("Saved sample realigned frame at {}".format(outpath))

    filtered.remove()
    interim_hdf5.flush()

    print("Calculating F0...")
    start = timer()
    f0 = np.percentile(realigned[:7500], 5, axis=0)
    end = timer()
    print("Calculating F0 took {} s".format(end - start))

    print("Calculating delta F / F0")
    outpath = args.path.replace("raw", "preprocessed")
    os.makedirs(outpath.replace(outpath.split("/")[-1], ""), exist_ok=True)
    hdf5_file = tables.open_file(outpath, "w", title=animal_id)
    deltaF = hdf5_file.create_earray(
        hdf5_file.root,
        "deltaf",
        tables.Float32Atom(),
        shape=(0, gcamp_frames.shape[1], gcamp_frames.shape[2]),
        expectedrows=gcamp_frames.shape[0],
    )
    timestamps = hdf5_file.create_earray(
        hdf5_file.root,
        "timestamps",
        tables.StringAtom(26),
        shape=(0,),
        expectedrows=gcamp_frames.shape[0],
    )

    start = timer()
    for frame in realigned.iterrows():
        deltaF.append([(frame - f0) / (f0 + 1)])
    timestamps.append(gcamp_timestamps)
    end = timer()
    print("DeltaF took {} s".format(end - start))

    outpath = args.path.replace("raw", "interim").replace(
        ".h5", "_deltaf-sample-frame.png"
    )
    plt.imsave(outpath, deltaF[100])

    print("Saving preprocessed file...")
    interim_hdf5.flush()
    hdf5_file.flush()

    print("Saved preprocessed file to {}".format(outpath))

    print("Cleaning up...")
    os.remove(interim_path)


if __name__ == "__main__":
    main()
