import pandas as pd
from src.config import OUTPUT_DIR, TEST_DIR
import sys

def validate():
    print("Validating submission...")
    
    # Check if files exist
    results_path = OUTPUT_DIR / 'matching_results.tsv'
    cands_path = OUTPUT_DIR / 'candidate_pairs.tsv'
    
    if not results_path.exists():
        print("FAIL: matching_results.tsv not found.")
        return False
    if not cands_path.exists():
        print("FAIL: candidate_pairs.tsv not found.")
        return False
        
    res = pd.read_csv(results_path, sep='\t', dtype=str)
    cands = pd.read_csv(cands_path, sep='\t', dtype=str)
    
    # Load test S1 to check if all IDs are present
    from src.data_loader import load_dataset
    s1, _, _ = load_dataset(TEST_DIR, is_train=False)
    
    s1_test_ids = set(s1['entity_id'])
    s1_res_ids = set(res['entity_id'])
    
    if s1_test_ids != s1_res_ids:
        print("FAIL: Not all S1 IDs are in results or unexpected S1 IDs found.")
        return False
        
    if len(res['entity_id']) != len(s1_res_ids):
        print("FAIL: Duplicate S1 IDs found in results.")
        return False
        
    cand_pairs = set(zip(cands['s1_id'], cands['s23_id']))
    
    for _, row in res.iterrows():
        s1_id = row['entity_id']
        matches = str(row['match_ids']).split(',') if pd.notna(row['match_ids']) and row['match_ids'] else []
        
        seen_matches = set()
        for m in matches:
            if m in seen_matches:
                print(f"FAIL: Duplicate match ID {m} for S1 {s1_id}")
                return False
            seen_matches.add(m)
            
            if (s1_id, m) not in cand_pairs:
                print(f"FAIL: Predicted match {(s1_id, m)} not in candidate_pairs.tsv")
                return False

    print("PASS: Submission is valid.")
    return True

if __name__ == "__main__":
    if not validate():
        sys.exit(1)
