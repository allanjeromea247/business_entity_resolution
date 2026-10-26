from src.normalization import normalize_name, normalize_address

def preprocess_dataframe(df):
    df = df.copy()
    
    df['name_norm'] = df['business_name'].apply(normalize_name)
    df['address_norm'] = df['business_address'].apply(normalize_address)
    df['country_norm'] = df['country'].fillna('').str.lower().str.strip()
    
    # Tokenization
    df['name_tokens'] = df['name_norm'].apply(lambda x: set(x.split()))
    df['address_tokens'] = df['address_norm'].apply(lambda x: set(x.split()))
    
    return df
