import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)

def generate_tfidf_candidates(s1, s23, col_name, top_k=30):
    candidates = set()
    s1_texts = s1[col_name].fillna('')
    s23_texts = s23[col_name].fillna('')
    
    if len(s1_texts) == 0 or len(s23_texts) == 0:
        return candidates

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    try:
        s23_tfidf = vectorizer.fit_transform(s23_texts)
        s1_tfidf = vectorizer.transform(s1_texts)
    except ValueError:
        # Fallback if vocabulary is empty
        return candidates

    nbrs = NearestNeighbors(n_neighbors=min(top_k, len(s23_texts)), metric='cosine').fit(s23_tfidf)
    distances, indices = nbrs.kneighbors(s1_tfidf)

    s1_ids = s1['entity_id'].values
    s23_ids = s23['entity_id'].values

    for i, s1_idx in enumerate(s1_ids):
        for j, s23_idx_pos in enumerate(indices[i]):
            candidates.add((s1_idx, s23_ids[s23_idx_pos]))
            
    return candidates

def generate_candidates(s1, s23, top_k=30):
    candidates = set()
    
    s1_ids = s1['entity_id'].values
    s23_ids = s23['entity_id'].values
    
    # 1. Exact Name Blocking
    name_map = {}
    for idx, row in s23.iterrows():
        name = row['name_norm']
        if name:
            name_map.setdefault(name, []).append(row['entity_id'])
            
    for idx, row in s1.iterrows():
        name = row['name_norm']
        if name in name_map:
            for c_id in name_map[name]:
                candidates.add((row['entity_id'], c_id))

    # 2. Token blocking
    s23_token_map = {}
    for idx, row in s23.iterrows():
        for token in row['name_tokens']:
            if len(token) > 2: # Ignore very short tokens
                s23_token_map.setdefault(token, []).append(row['entity_id'])
                
    for idx, row in s1.iterrows():
        for token in row['name_tokens']:
            if len(token) > 2 and token in s23_token_map:
                # To prevent explosion, limit matches per token if it's too common
                matched_ids = s23_token_map[token]
                if len(matched_ids) < 100: 
                    for c_id in matched_ids:
                        candidates.add((row['entity_id'], c_id))

    # 3. TF-IDF retrieval on Name
    logger.info("Generating Name TF-IDF candidates...")
    candidates.update(generate_tfidf_candidates(s1, s23, 'name_norm', top_k=top_k))
    
    # 4. TF-IDF retrieval on Address
    logger.info("Generating Address TF-IDF candidates...")
    candidates.update(generate_tfidf_candidates(s1, s23, 'address_norm', top_k=top_k))

    logger.info(f"Total candidate pairs generated: {len(candidates)}")
    return candidates

def evaluate_blocking(candidates, gt):
    if gt is None:
        return
        
    true_pairs = set()
    for _, row in gt.iterrows():
        if 'entity_id' in gt.columns and 'match_ids' in gt.columns:
            s1_id = row['entity_id']
            matches = str(row['match_ids']).split(',') if pd.notna(row['match_ids']) and row['match_ids'] else []
            for m in matches:
                if m:
                    true_pairs.add((s1_id, m))
        else:
            s1_id = row['s1_id']
            if 's2_id' in gt.columns and pd.notna(row['s2_id']) and row['s2_id']:
                true_pairs.add((s1_id, row['s2_id']))
            if 's3_id' in gt.columns and pd.notna(row['s3_id']) and row['s3_id']:
                true_pairs.add((s1_id, row['s3_id']))
                
    if len(true_pairs) == 0:
        return
        
    found = len(true_pairs.intersection(candidates))
    recall = found / len(true_pairs)
    
    logger.info(f"Blocking Recall: {recall:.4f} ({found}/{len(true_pairs)})")
    
    s1_count = len(set([p[0] for p in candidates]))
    avg_cand = len(candidates) / max(1, s1_count)
    logger.info(f"Average candidates per S1: {avg_cand:.2f}")

    return recall
