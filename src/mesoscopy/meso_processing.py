import os
import argparse
import itertools
import tables
import numpy as np
import pandas as pd
import scipy.stats as stat


def main():
    parser = argparse.ArgumentParser(
        description="Process mesoscope data to generate average activity per ABA area."
    )
    parser.add_argument("path", type=str, help="Path to preprocessed file.")
    parser.add_argument("aba", type=str, help="Path to registered ABA file.")
    parser.add_argument("annotations", type=str, help="Path to ABA annotations file.")
    args = parser.parse_args()

    print("Loading ABA mask...")
    aba_annotations = pd.read_csv(args.annotations, delimiter=", ", engine="python")

    aba_exclude = [
        "FRP1",
        "VISpl1",
        "VISpor1",
        "VISli1",
        "TEa1",
        "AUDd1",
        "AUDp1",
        "AUDpo1",
        "AUDv1",
        "ORBm1",
    ]

    aba_annotations = aba_annotations[~aba_annotations.acronym.isin(aba_exclude)]

    aba_mask = np.load(args.aba)
    l_aba = np.zeros(aba_mask.shape, dtype=np.uint16)
    l_aba[:300] = aba_mask[:300]
    r_aba = np.zeros(aba_mask.shape, dtype=np.uint16)
    r_aba[300:] = aba_mask[300:]

    print("Working on {}".format(args.path))
    preprocessed = tables.open_file(args.path, "r")

    print("Calculating mean deltaF per area...")
    total_frames = preprocessed.root.deltaf.shape[0]
    activity = []
    for idx, frame in enumerate(preprocessed.root.deltaf.iterrows()):
        print("Processing frame {} of {}...".format(idx, total_frames), end="\r")
        for _, area in aba_annotations.iterrows():
            l_mask = np.ma.masked_array(frame, np.not_equal(l_aba, area.id))
            r_mask = np.ma.masked_array(frame, np.not_equal(r_aba, area.id))
            activity.append(
                {
                    "frame": idx,
                    "area": "L_" + area.acronym,
                    "mean": np.ma.mean(l_mask),
                    "std": np.ma.std(l_mask),
                    "timestamp": preprocessed.root.timestamps[idx],
                }
            )
            activity.append(
                {
                    "frame": idx,
                    "area": "R_" + area.acronym,
                    "mean": np.ma.mean(r_mask),
                    "std": np.ma.std(l_mask),
                    "timestamp": preprocessed.root.timestamps[idx],
                }
            )

    outpath = args.path.replace("preprocessed", "processed").replace(
        ".h5", "_area-activity.csv"
    )
    os.makedirs(outpath.replace(outpath.split("/")[-1], ""), exist_ok=True)
    print("Saving to {}".format(outpath))
    df = pd.DataFrame(activity)
    df.to_csv(outpath)

    print("Calculating activity correlations for ABA pairs...")
    l_annotations = ["L_" + acr for acr in aba_annotations.acronym.tolist()]
    r_annotations = ["R_" + acr for acr in aba_annotations.acronym.tolist()]

    areas_personsr = []
    for pair in itertools.combinations(l_annotations + r_annotations, 2):
        if pair[0] == pair[1]:
            continue
        stim = df[df.area == pair[0]]["mean"].to_numpy()
        resp = df[df.area == pair[1]]["mean"].to_numpy()
        corr = stat.pearsonr(stim, resp)
        areas_personsr.append(
            {
                "stim": pair[0],
                "resp": pair[1],
                "r": corr[0],
                "p": corr[1],
            }
        )

    outpath = args.path.replace("preprocessed", "processed").replace(
        ".h5", "_corr-pearsons.csv"
    )
    os.makedirs(outpath.replace(outpath.split("/")[-1], ""), exist_ok=True)
    print("Saving to {}".format(outpath))
    df_pearsons = pd.DataFrame(areas_personsr)
    df_pearsons.to_csv(outpath)


if __name__ == "__main__":
    main()
