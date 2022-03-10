import os
import argparse
import pathlib
import tables
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

matplotlib.rcParams["figure.dpi"] = 150


def main():
    parser = argparse.ArgumentParser(
        description="Sample mesoscope data to extract sample images for manual registration"
    )
    parser.add_argument("path", type=str, help="Path to recording file.")

    args = parser.parse_args()

    print("Sampling {}".format(args.path))
    raw = tables.open_file(args.path, "r")

    os.makedirs(
        args.path.replace("raw", "interim").replace(args.path.split("/")[-1], ""),
        exist_ok=True,
    )

    plt.clf()
    outpath = args.path.replace("raw", "interim").replace(".h5", "_regsample1.png")
    plt.imsave(outpath, raw.root.frames[100])
    print("Saved sample 1 at {}".format(outpath))

    plt.clf()
    outpath = args.path.replace("raw", "interim").replace(".h5", "_regsample2.png")
    plt.imsave(outpath, raw.root.frames[101])
    print("Saved sample 2 at {}".format(outpath))


if __name__ == "__main__":
    main()
