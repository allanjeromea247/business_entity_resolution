import numpy as np

def calculate_macro_f05(y_true_dict, y_pred_dict, s1_ids):
    """
    Calculates macro entity-level F0.5 score.
    y_true_dict: {s1_id: set(s23_ids)}
    y_pred_dict: {s1_id: set(s23_ids)}
    """
    entity_scores = []
    
    for s1_id in s1_ids:
        true_set = y_true_dict.get(s1_id, set())
        pred_set = y_pred_dict.get(s1_id, set())
        
        if len(true_set) == 0 and len(pred_set) == 0:
            entity_scores.append(1.0)
        elif len(true_set) == 0 and len(pred_set) > 0:
            entity_scores.append(0.0)
        elif len(true_set) > 0 and len(pred_set) == 0:
            entity_scores.append(0.0)
        else:
            tp = len(true_set.intersection(pred_set))
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            if precision == 0 and recall == 0:
                entity_scores.append(0.0)
            else:
                beta = 0.5
                f05 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
                entity_scores.append(f05)
                
    return np.mean(entity_scores)

def optimize_threshold(y_prob, candidates_list, gt_dict, s1_ids):
    best_threshold = 0.5
    best_score = 0.0
    
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95]
    
    for th in thresholds:
        y_pred_dict = {}
        for (s1_id, s23_id), prob in zip(candidates_list, y_prob):
            if prob >= th:
                y_pred_dict.setdefault(s1_id, set()).add(s23_id)
                
        score = calculate_macro_f05(gt_dict, y_pred_dict, s1_ids)
        if score > best_score:
            best_score = score
            best_threshold = th
            
    return best_threshold, best_score
