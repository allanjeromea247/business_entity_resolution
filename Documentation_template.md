# Documentation

## Approach
The pipeline consists of preprocessing text, generating high-recall candidate pairs using blocking methods, extracting character and token level similarity features, and classifying matches using a Random Forest model. Threshold is tuned via validation set.

## Data preprocessing
Names and addresses are lowercased, unaccented, and common business/street abbreviations are normalized (e.g., Pvt -> pvt, Street -> street).

## Blocking
We use exact name matching, shared informative name tokens, and nearest neighbor search over character n-gram TF-IDF representations of names and addresses.

## Feature engineering
Features include exact match boolean, Levenshtein distance, Jaccard similarity of tokens, token overlap count, and length differences for both names and addresses.

## Model
RandomForestClassifier with class weights balanced.

## Training
Positive pairs from ground truth. Negative pairs sampled from blocking candidates.

## Validation
Macro entity-level F0.5 score averaged over all Source 1 entities in the validation fold.

## Threshold
Grid search over [0.50, 0.95] to maximize validation F0.5.

## Error analysis
Common issues include false positives on companies with the same generic names but different addresses, or missing completely different trading names.

## Generalization
Countries are not hardcoded. The system applies identical similarity metrics dynamically to any country text.

## Reproducibility
Run `python run.py`.
