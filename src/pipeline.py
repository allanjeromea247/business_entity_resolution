from src.data_loader import load_dataset, generate_eda
from src.preprocessing import preprocess_dataframe
from src.blocking import generate_candidates, evaluate_blocking
from src.ground_truth import parse_ground_truth, generate_training_labels
from src.features import compute_features_for_pairs
from src.model import train_model, save_model, load_model
from src.evaluation import optimize_threshold, calculate_macro_f05
from src.prediction import generate_predictions, save_predictions, save_candidates
import src.config as cfg
import json
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def run_training_pipeline():
    logger.info("[1/10] Loading data...")
    s1, s23, gt = load_dataset(cfg.TRAIN_DIR, is_train=True)
    generate_eda(s1, s23, gt, cfg.LOGS_DIR)
    
    logger.info("[2/10] Normalizing...")
    s1 = preprocess_dataframe(s1)
    s23 = preprocess_dataframe(s23)
    
    # Train/Validation split by S1 entity
    s1_train, s1_val = train_test_split(s1, test_size=0.2, random_state=42)
    logger.info(f"Train S1: {len(s1_train)}, Val S1: {len(s1_val)}")
    
    logger.info("[3/10] Building blocking indexes... (Train)")
    train_candidates = generate_candidates(s1_train, s23, top_k=cfg.TOP_K_TFIDF)
    
    logger.info("Evaluating Train Blocking...")
    evaluate_blocking(train_candidates, gt)
    
    train_true_pairs = parse_ground_truth(gt)
    
    logger.info("[4/10] Generating candidates... (Train/Val pairs)")
    train_candidates_list = list(train_candidates)
    train_labels = generate_training_labels(train_candidates_list, train_true_pairs)
    
    logger.info("[5/10] Calculating features... (Train)")
    X_train = compute_features_for_pairs(train_candidates_list, s1_train, s23)
    
    logger.info("[6/10] Training model...")
    model = train_model(X_train, train_labels)
    save_model(model, cfg.MODEL_PATH)
    
    logger.info("[7/10] Evaluating... (Validation)")
    val_candidates = generate_candidates(s1_val, s23, top_k=cfg.TOP_K_TFIDF)
    val_candidates_list = list(val_candidates)
    
    X_val = compute_features_for_pairs(val_candidates_list, s1_val, s23)
    y_prob_val = model.predict_proba(X_val)[:, 1] if len(X_val) > 0 else []
    
    # Validation GT dict
    val_s1_ids = set(s1_val['entity_id'])
    val_gt_dict = {}
    for p in train_true_pairs:
        if p[0] in val_s1_ids:
            val_gt_dict.setdefault(p[0], set()).add(p[1])
            
    logger.info("[8/10] Optimizing threshold...")
    if len(val_candidates_list) > 0:
        best_th, best_f05 = optimize_threshold(y_prob_val, val_candidates_list, val_gt_dict, val_s1_ids)
    else:
        best_th, best_f05 = cfg.THRESHOLD_DEFAULT, 0.0
        
    logger.info(f"Best Validation F0.5: {best_f05:.4f} at Threshold: {best_th}")
    
    with open(cfg.CONFIG_PATH, 'w') as f:
        json.dump({'threshold': best_th, 'f05': best_f05}, f)
        
    return best_th, best_f05

def run_prediction_pipeline():
    logger.info("[1/10] Loading data...")
    s1, s23, _ = load_dataset(cfg.TEST_DIR, is_train=False)
    
    logger.info("[2/10] Normalizing...")
    s1 = preprocess_dataframe(s1)
    s23 = preprocess_dataframe(s23)
    
    logger.info("[3/10] Loading model...")
    model = load_model(cfg.MODEL_PATH)
    with open(cfg.CONFIG_PATH, 'r') as f:
        th_config = json.load(f)
    threshold = th_config['threshold']
    
    logger.info("[4/10] Generating candidates... (Test)")
    test_candidates = generate_candidates(s1, s23, top_k=cfg.TOP_K_TFIDF)
    test_candidates_list = list(test_candidates)
    
    logger.info("[5/10] Calculating features... (Test)")
    if len(test_candidates_list) > 0:
        X_test = compute_features_for_pairs(test_candidates_list, s1, s23)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Country Override Rule: If both countries exist but do not match, force prob to 0.
        mismatch_mask = (X_test['country_exact'] == 0) & (X_test['country_missing'] == 0)
        y_prob[mismatch_mask] = 0.0
    else:
        y_prob = []
        
    logger.info("[9/10] Generating test predictions...")
    s1_ids = list(s1['entity_id'])
    preds_df = generate_predictions(test_candidates_list, y_prob, threshold, s1_ids)
    
    out_preds_path = cfg.OUTPUT_DIR / "matching_results.tsv"
    out_cands_path = cfg.OUTPUT_DIR / "candidate_pairs.tsv"
    save_predictions(preds_df, out_preds_path)
    save_candidates(test_candidates_list, out_cands_path)
    
    logger.info(f"Saved matching results to {out_preds_path}")
    logger.info(f"Saved candidate pairs to {out_cands_path}")
