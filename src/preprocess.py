import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

def preprocess_text(text):
    """
    Preprocess text by lowercasing, removing punctuation and stopwords, and stemming.
    """
    if not isinstance(text, str):
        return ""
        
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and Stemming
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    cleaned_tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    
    return " ".join(cleaned_tokens)

if __name__ == "__main__":
    sample_text = "Hello! This is a sample message... to test preprocessing."
    print(f"Original: {sample_text}")
    print(f"Processed: {preprocess_text(sample_text)}")
