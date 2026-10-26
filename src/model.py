from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X, y):
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X, y)
    return clf

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
