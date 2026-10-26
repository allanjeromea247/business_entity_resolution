import pandas as pd
from pathlib import Path
import os

def organize_data():
    base_dir = Path("C:/Users/dhris/OneDrive/Documents/AIMLHACK/business_entity_resolution/data")
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    
    test_dir.mkdir(parents=True, exist_ok=True)

    print("Loading original sources from data/train...")
    s1 = pd.read_csv(train_dir / "source1.tsv", sep="\t", dtype=str)
    s2 = pd.read_csv(train_dir / "source2.tsv", sep="\t", dtype=str)
    s3 = pd.read_csv(train_dir / "source3.tsv", sep="\t", dtype=str)

    train_gt = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", dtype=str)
    test_gt = pd.read_csv(train_dir / "test_ground_truth.tsv", sep="\t", dtype=str)

    print("Identifying train and test entity IDs...")
    # Get all S1 IDs in train and test
    train_s1_ids = set(train_gt['s1_id'].dropna())
    test_s1_ids = set(test_gt['s1_id'].dropna())

    # We will split S1 based on ground truth presence
    train_s1 = s1[s1['entity_id'].isin(train_s1_ids)]
    test_s1 = s1[s1['entity_id'].isin(test_s1_ids)]

    # We can just copy S2 and S3 entirely to both train and test since candidate blocking 
    # handles pulling relevant pairs, but to be clean, let's just pass the full S2/S3 to both 
    # or save them as train_source2 / test_source2.
    # It's safest to provide the full S2/S3 to both train and test so all potential matches are available.
    train_s2 = s2
    train_s3 = s3
    test_s2 = s2
    test_s3 = s3

    print("Saving organized files...")
    train_s1.to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)
    train_s2.to_csv(train_dir / "train_source2.tsv", sep="\t", index=False)
    train_s3.to_csv(train_dir / "train_source3.tsv", sep="\t", index=False)
    
    test_s1.to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)
    test_s2.to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)
    test_s3.to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)
    
    # We leave train_ground_truth.tsv where it is
    print("Data successfully organized into train/ and test/ folders!")

if __name__ == "__main__":
    organize_data()
