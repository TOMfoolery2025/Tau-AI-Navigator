import re
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK data is present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextNormalizer:
    """
    Adapted from TVAStringCleansing and TVALabelTools.
    Handles 'Blocklists' and 'Stemming' to clean noise from OSM/GTFS data.
    """
    def __init__(self, language="english"):
        self.stemmer = SnowballStemmer(language)
        # Transport-specific blocklist to remove noise
        self.blocklist = {
            "stop", "station", "platform", "zone", "public", "transport", 
            "entrance", "exit", "way", "lane", "street", "road", "undefined",
            "yes", "no", "true", "false", "available"
        }

    def clean_and_stem(self, text):
        """
        Splits text, removes stop words (Blocklist), and stems relevant concepts.
        """
        if not text or not isinstance(text, str):
            return []
        
        # 1. Aggressive Cleaning (Regex from TVA)
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.lower().split()
        
        valid_concepts = []
        for w in words:
            # 2. Blocklist Filter
            if w not in self.blocklist and len(w) > 2:
                # 3. Stemming
                stemmed = self.stemmer.stem(w)
                valid_concepts.append(stemmed)
                
        return list(set(valid_concepts))

class VectorSearchEngine:
    """
    Adapted from TVASemanticSearchTools.
    Provides Vector Embedding and Cosine Similarity search.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Using a lighter model than TVA for Hackathon speed
        self.model = SentenceTransformer(model_name)
        self.cached_embeddings = None
        self.cached_metadata = []

    def encode_text(self, text_list):
        """Generates vector embeddings for a list of strings."""
        if not text_list:
            return None
        return self.model.encode(text_list, convert_to_tensor=True)

    def fit_index(self, data_objects, text_key='description'):
            self.cached_metadata = data_objects
            # FIX: Force conversion to string and handle None explicitly
            corpus = []
            for obj in data_objects:
                val = obj.get(text_key)
                corpus.append(str(val) if val is not None else "")
                
            self.cached_embeddings = self.model.encode(corpus, convert_to_tensor=True)
            print(f"✅ Indexed {len(corpus)} items semantically.")

    def search(self, query, top_k=5):
        """
        Performs Cosine Similarity search (TVA Logic).
        """
        if self.cached_embeddings is None:
            return []
        
        # Encode query
        query_vec = self.model.encode(query, convert_to_tensor=True)
        
        # Calculate Cosine Similarity
        # Move to CPU for numpy operations if using Torch
        query_vec = query_vec.cpu().numpy().reshape(1, -1)
        corpus_vecs = self.cached_embeddings.cpu().numpy()
        
        scores = cosine_similarity(query_vec, corpus_vecs)[0]
        
        # Sort results
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0.25: # Threshold to reduce noise
                item = self.cached_metadata[idx]
                item['similarity_score'] = float(score)
                results.append(item)
                
        return results