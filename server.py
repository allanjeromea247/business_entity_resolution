import json
import pandas as pd
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
from src.normalization import normalize_name, normalize_address
from src.features import compute_features_for_pairs
from src.model import load_model
from src.config import MODEL_PATH, CONFIG_PATH
import os

# Load ML components
print("Loading model...")
model = load_model(MODEL_PATH)
with open(CONFIG_PATH, 'r') as f:
    th_config = json.load(f)
THRESHOLD = th_config['threshold']

class SimpleAPI(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('index.html', 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Create a mock dataframe
            s1 = pd.DataFrame([{
                'entity_id': 'S1_DEMO',
                'business_name': data['s1_name'],
                'business_address': data['s1_address'],
                'country': data['s1_country']
            }])
            
            s23 = pd.DataFrame([{
                'entity_id': 'S2_DEMO',
                'business_name': data['s2_name'],
                'business_address': data['s2_address'],
                'country': data['s2_country']
            }])
            
            # Normalization
            for df in [s1, s23]:
                df['name_norm'] = df['business_name'].apply(normalize_name)
                df['address_norm'] = df['business_address'].apply(normalize_address)
                df['country_norm'] = df['country'].fillna('').str.lower().str.strip()
                df['name_tokens'] = df['name_norm'].apply(lambda x: set(x.split()))
                df['address_tokens'] = df['address_norm'].apply(lambda x: set(x.split()))

            # Extract Features
            candidates_list = [('S1_DEMO', 'S2_DEMO')]
            X_test = compute_features_for_pairs(candidates_list, s1, s23)
            
            # Predict
            prob = float(model.predict_proba(X_test)[0, 1])
            
            # Country Override Rule
            if X_test.iloc[0]['country_exact'] == 0 and X_test.iloc[0]['country_missing'] == 0:
                prob = 0.0
                
            is_match = prob >= THRESHOLD
            
            # Respond
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = json.dumps({
                'probability': prob,
                'is_match': bool(is_match),
                'threshold_used': THRESHOLD
            })
            self.wfile.write(response.encode('utf-8'))

def run_server():
    server_address = ('', 8050)
    httpd = HTTPServer(server_address, SimpleAPI)
    print("========================================")
    print("Frontend running at http://localhost:8050")
    print("========================================")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
