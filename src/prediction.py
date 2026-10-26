import pandas as pd

def generate_predictions(candidates_list, y_prob, threshold, s1_ids):
    y_pred_dict = {}
    
    for (s1_id, s23_id), prob in zip(candidates_list, y_prob):
        if prob >= threshold:
            y_pred_dict.setdefault(s1_id, set()).add(s23_id)
            
    # Format output
    output_rows = []
    for s1_id in s1_ids:
        matches = y_pred_dict.get(s1_id, set())
        sorted_matches = sorted(list(matches))
        match_str = ",".join(sorted_matches)
        output_rows.append([s1_id, match_str])
        
    return pd.DataFrame(output_rows, columns=['entity_id', 'match_ids'])

def save_predictions(df, path):
    df.to_csv(path, sep='\t', index=False)

def save_candidates(candidates_list, path):
    df = pd.DataFrame(candidates_list, columns=['s1_id', 's23_id'])
    df.to_csv(path, sep='\t', index=False)
