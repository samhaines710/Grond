#!/usr/bin/env python3
import argparse
import logging

import numpy as np
import pandas as pd

from ml_classifier import MLClassifier


def population_stability_index(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """
    Compute the Population Stability Index (PSI) between two distributions.
    """
    # Create quantile bins on the expected (training) distribution
    expected_bins = pd.qcut(expected, buckets, duplicates="drop")
    actual_bins = pd.qcut(actual, buckets, duplicates="drop")

    exp_counts = expected_bins.value_counts(normalize=True).sort_index()
    act_counts = actual_bins.value_counts(normalize=True).sort_index()

    # Align indices
    idx = exp_counts.index.union(act_counts.index)
    exp = exp_counts.reindex(idx, fill_value=0)
    act = act_counts.reindex(idx, fill_value=0)

    # PSI formula
    psi = ((exp - act) * np.log(exp / act)).sum()
    return psi


def main():
    parser = argparse.ArgumentParser(
        description="Compute PSI for recent vs. reference data to detect model drift."
    )
    parser.add_argument(
        "--ref-data",
        required=True,
        help="Path to CSV of reference (training) features."
    )
    parser.add_argument(
        "--recent-data",
        required=True,
        help="Path to CSV of recent production features."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="PSI threshold above which an alert is logged."
    )
    parser.add_argument(
        "--log-file",
        help="Optional path to write logs (defaults to stdout)."
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    logger.info("Loading reference dataset from %s", args.ref_data)
    ref_df = pd.read_csv(args.ref_data)

    logger.info("Loading recent dataset from %s", args.recent_data)
    recent_df = pd.read_csv(args.recent_data)

    logger.info("Initializing MLClassifier for feature extraction")
    clf = MLClassifier()
    pipeline = clf.pipeline
    feature_names = clf.feature_names

    # Assume first step in pipeline is the scaler
    scaler = pipeline.steps[0][1]

    logger.info("Transforming reference and recent features")
    ref_vals = scaler.transform(ref_df[feature_names])
    recent_vals = scaler.transform(recent_df[feature_names])

    # Compute PSI per feature
    psis = {}
    for idx, feat in enumerate(feature_names):
        psi = population_stability_index(
            pd.Series(ref_vals[:, idx]),
            pd.Series(recent_vals[:, idx])
        )
        psis[feat] = psi
        logger.debug("PSI for %s: %.4f", feat, psi)

    overall_psi = np.mean(list(psis.values()))
    logger.info("Overall PSI across %d features: %.4f", len(feature_names), overall_psi)

    if overall_psi > args.threshold:
        logger.warning(
            "Model drift detected: PSI %.4f exceeds threshold %.4f",
            overall_psi,
            args.threshold
        )
    else:
        logger.info("No significant drift: PSI %.4f within threshold %.4f",
                    overall_psi, args.threshold)


if __name__ == "__main__":
    main()
