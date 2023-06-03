import os
import logging
from collections import defaultdict
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import research.common.dataset_config as dc


def get_volume_paths(dataset_path: str) -> Dict[int, str]:
    volume_dict = {}
    exclude = ["rotated", "flipped", "noisy"]

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if any(ex in file for ex in exclude):
                continue

            if file.endswith(".nii"):
                path = os.path.join(root, file)
                image_id = int(file[file.rindex("I") + 1 : -4])

                volume_dict[image_id] = path

    return volume_dict


def get_df(cfg: dc.DatasetConfig) -> Tuple[Dict[int, str], pd.DataFrame]:
    required = ["PTID", "IMAGEUID", "DX", "DX_bl", "Month"]
    subs = list(set(required) | set(cfg.ni_vars))

    # get ADNIMERGE dataframe, remove any irrelevant data, remove any data we dont have downloaded
    volume_dict = get_volume_paths(cfg.scan_paths)
    df = pd.read_csv("ADNIMERGE.csv", low_memory=False).dropna(subset=subs)[subs]
    df = df[df["IMAGEUID"].isin(volume_dict.keys())]

    # remove any data that we've downloaded but doesn't meet criteria for current task
    new_dict = {}
    for image_id in df["IMAGEUID"].values:
        new_dict[int(image_id)] = volume_dict[int(image_id)]

    return new_dict, df


def get_num_seq_rows(
    df: pd.DataFrame, freq: int, seq_length: int
) -> List[Tuple[List[int], List[int], List[int]]]:
    if len(df) < seq_length:
        return []

    seqs = []
    for i in range(len(df) - seq_length + 1):
        seq_df = df.iloc[i : i + seq_length]
        dxs, months, ids = (
            seq_df["DX"].values,
            seq_df["Month"].values,
            seq_df["IMAGEUID"].values,
        )

        # make sure each scan is `freq` months apart
        gap_arr = np.unique(np.array(months[:-1]) - np.array(months[1:]))
        if len(gap_arr) == 1 and np.unique(gap_arr)[0] == freq * -1:
            seqs.append((dxs, months, ids))

    return seqs


def create_class_dict(
    cfg: dc.DatasetConfig,
) -> Tuple[Dict[int, str], pd.DataFrame, Dict[str, Tuple[dc.PatientCohort, List[int]]]]:
    paths, df = get_df(cfg)
    unique_pt = df["PTID"].unique()

    cohort_dict = {}
    for pt in unique_pt:
        # get patient dataframe, and all instances of sequential visits
        pt_df = df[df["PTID"] == pt].sort_values(by=["Month"])
        seqs = get_num_seq_rows(pt_df, cfg.seq_visit_delta, cfg.num_seq_visits)
        num_dx = len(pt_df["DX"].unique())

        dx = pt_df["DX"].unique()[0]
        res = (dc.PatientCohort.dx_to_cohort(dx), None)

        # if there were `cfg.num_seq_visits` visits of `cfg.seq_visit_delta` months apart
        if len(seqs) != 0:
            for dxs, months, ids in seqs:
                # get df for months `months[0] + cfg.progression_window +/- cfg.tolerance`
                mdf = pt_df["Month"] - (months[0] + cfg.progression_window)
                final_df = pt_df[(mdf >= -cfg.tolerance) & (mdf <= cfg.tolerance)]

                if len(final_df) == 0:
                    continue

                final_dx = final_df["DX"].values
                dementia_after_window = "Dementia" in final_dx

                is_converter = dxs[-1] == "MCI" and dementia_after_window
                is_nonconverter = dxs[-1] == "MCI" and not dementia_after_window

                if is_converter:
                    res = (dc.PatientCohort.pMCI, ids)
                    break
                elif is_nonconverter:
                    res = (dc.PatientCohort.sMCI, ids)
                    break

        cohort_dict[pt] = res
    return paths, df, cohort_dict


def create_dataset(cfg: dc.DatasetConfig) -> List:
    paths, df, cohort_dict = create_class_dict(cfg)
    seqlen = cfg.num_seq_visits

    data_ids = defaultdict(lambda: ([], []))
    for ptid in cohort_dict:
        cohort, ids = cohort_dict[ptid]
        pt_df = df[df["PTID"] == ptid]

        selected_ids, selected_data = data_ids[cohort.get_task_type()]
        if cohort.is_classification():
            candidates = [pt_df[pt_df["DX"] == dx] for dx in cfg.cohorts]

            for cand in candidates:
                if len(cand) > 0:
                    dx, imid = cand[["DX", "IMAGEUID"]].values[0]
                    selected_ids.append((dc.PatientCohort.dx_to_cohort(dx), int(imid)))
                    if len(cfg.ni_vars) > 0:
                        selected_data.append(cand[cfg.ni_vars].values[0])

                    break

        elif cohort.is_prediction():
            rows = pt_df[pt_df["IMAGEUID"].isin(ids)].sort_values(by=["Month"])
            assert len(rows) == seqlen

            selected_ids.append((rows["IMAGEUID"].values, cohort))

            if len(cfg.ni_vars) > 0:
                for v in rows[cfg.ni_vars].values:
                    selected_data.append(v if len(v) > 0 else -1)

    scan_ids, ni_data = data_ids[cfg.task]
    if len(scan_ids) == 0 or len(ni_data) == 0:
        logging.error(
            f"Invalid number of scan_ids or ni_ids: {len(scan_ids)} {len(ni_data)}"
        )
        exit(1)

    ni_data = np.array(ni_data)
    mean, std = np.mean(ni_data, axis=0), np.std(ni_data, axis=0)
    ni_data = np.apply_along_axis(lambda row: (row - mean) / std, 1, ni_data)

    dataset = []
    for idx, (dx, imids) in enumerate(scan_ids):
        if cfg.task == dc.DatasetTask.CLASSIFICATION:
            if len(cfg.cohorts) > 0 and dx.name not in cfg.cohorts:
                continue
            elif len(ni_data) == 0:
                dataset.append((paths[int(imids)], None, dx))
            else:
                dataset.append((paths[int(imids)], ni_data[idx], dx))
        elif cfg.task == dc.DatasetTask.PREDICTION:
            volume_paths = [paths[int(image_id)] for image_id in imids]
            if len(ni_data) == 0:
                dataset.append((volume_paths, None, dx))
            else:
                dataset.append(
                    (volume_paths, ni_data[seqlen * idx : seqlen * idx + seqlen], dx)
                )

    logging.info(f"Found {len(dataset)} patients in this dataset")
    return dataset
