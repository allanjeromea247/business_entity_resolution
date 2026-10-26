import pandas as pd
import numpy as np
from rapidfuzz import fuzz

def compute_features_for_pairs(candidates_list, s1_df, s23_df):
    s1_df = s1_df.set_index('entity_id')
    s23_df = s23_df.set_index('entity_id')
    
    features = []
    for s1_id, s23_id in candidates_list:
        try:
            row_s1 = s1_df.loc[s1_id]
            row_s23 = s23_df.loc[s23_id]
        except KeyError:
            continue
            
        f = {}
        
        n1 = str(row_s1['name_norm'])
        n2 = str(row_s23['name_norm'])
        a1 = str(row_s1['address_norm'])
        a2 = str(row_s23['address_norm'])
        c1 = str(row_s1['country_norm'])
        c2 = str(row_s23['country_norm'])

        t_n1 = row_s1['name_tokens']
        t_n2 = row_s23['name_tokens']
        t_a1 = row_s1['address_tokens']
        t_a2 = row_s23['address_tokens']
        
        # Name features
        f['name_exact'] = int(n1 == n2 and len(n1) > 0)
        f['name_levenshtein'] = fuzz.ratio(n1, n2)
        f['name_jaccard'] = len(t_n1.intersection(t_n2)) / max(1, len(t_n1.union(t_n2)))
        f['name_token_overlap'] = len(t_n1.intersection(t_n2))
        f['name_length_diff'] = abs(len(n1) - len(n2))
        
        # Address features
        f['address_exact'] = int(a1 == a2 and len(a1) > 0)
        f['address_levenshtein'] = fuzz.ratio(a1, a2)
        f['address_jaccard'] = len(t_a1.intersection(t_a2)) / max(1, len(t_a1.union(t_a2)))
        f['address_token_overlap'] = len(t_a1.intersection(t_a2))
        f['address_length_diff'] = abs(len(a1) - len(a2))
        
        # Country features
        f['country_exact'] = int(c1 == c2 and len(c1) > 0)
        f['country_missing'] = int(len(c1) == 0 or len(c2) == 0)

        # Token count diff
        f['name_token_count_diff'] = abs(len(t_n1) - len(t_n2))
        f['address_token_count_diff'] = abs(len(t_a1) - len(t_a2))
        
        features.append(f)
        
    return pd.DataFrame(features)
