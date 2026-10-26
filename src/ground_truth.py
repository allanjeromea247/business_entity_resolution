import pandas as pd

def parse_ground_truth(gt):
    """Parses ground truth into positive pairs."""
    true_pairs = set()
    for _, row in gt.iterrows():
        # Support old format (entity_id, match_ids)
        if 'entity_id' in gt.columns and 'match_ids' in gt.columns:
            s1_id = row['entity_id']
            matches = str(row['match_ids']).split(',') if pd.notna(row['match_ids']) and row['match_ids'] else []
            for m in matches:
                if m:
                    true_pairs.add((s1_id, m))
        # Support new format (s1_id, s2_id, s3_id)
        else:
            s1_id = row['s1_id']
            if 's2_id' in gt.columns and pd.notna(row['s2_id']) and row['s2_id']:
                true_pairs.add((s1_id, row['s2_id']))
            if 's3_id' in gt.columns and pd.notna(row['s3_id']) and row['s3_id']:
                true_pairs.add((s1_id, row['s3_id']))
    return true_pairs

def generate_training_labels(candidate_pairs, true_pairs):
    """Labels candidate pairs as 1 (match) or 0 (non-match)."""
    labels = []
    for pair in candidate_pairs:
        labels.append(1 if pair in true_pairs else 0)
    return labels
