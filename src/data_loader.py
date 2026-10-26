import pandas as pd
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

def find_file(directory: Path, keywords: list):
    """Find a file in the directory that matches all keywords."""
    if not directory.exists():
        return None
    for file in os.listdir(directory):
        if file.endswith(".tsv"):
            if all(k.lower() in file.lower() for k in keywords):
                return directory / file
    return None

def load_dataset(directory: Path, is_train: bool = True):
    """Loads source1, source2, source3 and ground_truth (if train)."""
    s1_path = find_file(directory, ["source1"])
    s2_path = find_file(directory, ["source2"])
    s3_path = find_file(directory, ["source3"])
    
    if is_train:
        gt_path = find_file(directory, ["train", "ground_truth"])
        if not gt_path:
            gt_path = find_file(directory, ["ground_truth"]) # fallback
    else:
        gt_path = None

    if not all([s1_path, s2_path, s3_path]):
        raise FileNotFoundError(f"Could not find all source files in {directory}")
    
    if is_train and not gt_path:
        raise FileNotFoundError(f"Could not find ground_truth in {directory}")

    s1 = pd.read_csv(s1_path, sep="\t", dtype=str).dropna(how="all")
    s2 = pd.read_csv(s2_path, sep="\t", dtype=str).dropna(how="all")
    s3 = pd.read_csv(s3_path, sep="\t", dtype=str).dropna(how="all")
    
    s2['source'] = 'S2'
    s3['source'] = 'S3'
    
    s23 = pd.concat([s2, s3], ignore_index=True)
    
    gt = pd.read_csv(gt_path, sep="\t", dtype=str) if gt_path else None
    
    logger.info(f"Loaded {len(s1)} S1 records, {len(s2)} S2 records, {len(s3)} S3 records.")
    
    return s1, s23, gt

def generate_eda(s1, s23, gt, log_dir):
    """Generates simple EDA report."""
    report = []
    report.append(f"S1 shape: {s1.shape}")
    report.append(f"S2+S3 shape: {s23.shape}")
    report.append(f"S1 unique countries: {s1['country'].nunique()}")
    report.append(f"S2+S3 unique countries: {s23.shape}")
    if gt is not None:
        report.append(f"Ground Truth shape: {gt.shape}")
        
    with open(Path(log_dir) / "eda_report.txt", "w") as f:
        f.write("\n".join(report))
