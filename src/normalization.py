import re
import unicodedata

def normalize_text_basic(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    # Remove periods entirely first so "M.G." becomes "mg" instead of "m g"
    text = text.replace('.', '')
    # Replace other punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_name(name):
    text = normalize_text_basic(name)
    
    # Common abbreviations
    text = re.sub(r'\b(pvt|private)\b', 'pvt', text)
    text = re.sub(r'\b(ltd|limited)\b', 'ltd', text)
    text = re.sub(r'\b(corp|corporation)\b', 'corp', text)
    text = re.sub(r'\b(inc|incorporated)\b', 'inc', text)
    text = re.sub(r'\b(co|company)\b', 'co', text)
    text = re.sub(r'\b(llc)\b', 'llc', text)
    text = re.sub(r'\band\b', ' ', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_address(address):
    text = normalize_text_basic(address)
    
    # Common address terms
    text = re.sub(r'\b(rd|road)\b', 'road', text)
    text = re.sub(r'\b(st|street)\b', 'street', text)
    text = re.sub(r'\b(ave|avenue)\b', 'avenue', text)
    text = re.sub(r'\b(apt|apartment)\b', 'apartment', text)
    text = re.sub(r'\b(bldg|building)\b', 'building', text)
    text = re.sub(r'\b(fl|floor)\b', 'floor', text)
    
    # Common US States to Abbreviations
    text = re.sub(r'\b(california)\b', 'ca', text)
    text = re.sub(r'\b(new york)\b', 'ny', text)
    text = re.sub(r'\b(texas)\b', 'tx', text)
    text = re.sub(r'\b(florida)\b', 'fl', text)
    text = re.sub(r'\b(washington)\b', 'wa', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text
