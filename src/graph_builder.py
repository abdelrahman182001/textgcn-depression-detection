import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

class TextGCNGraph:
    def __init__(self, df: pd.DataFrame):
        print("\n--- Initializing Graph Builder ---")
        self.df = df
        self.num_docs = len(df)
        
        # CRITICAL ENGINEERING FIX: 
        # We override the default token_pattern. By using a simple split, 
        # we guarantee that emojis and emoticons like :( are treated as valid vocabulary words.
        def custom_tokenizer(text):
            return text.split()
            
        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, lowercase=False)
        
    def build_tfidf_edges(self):
        """
        Calculates TF-IDF to create edges between Word Nodes and Document Nodes.
        """
        print("Calculating TF-IDF (Word-Document edges)...")
        
        # Fit the vectorizer to our cleaned text. 
        # This returns a sparse matrix of shape (num_docs, num_vocab)
        tfidf_matrix = self.vectorizer.fit_transform(self.df['cleaned_text'])
        
        # Extract the total vocabulary size
        self.vocab = self.vectorizer.get_feature_names_out()
        self.num_vocab = len(self.vocab)
        self.total_nodes = self.num_docs + self.num_vocab
        
        print(f"Graph Dimensions Locked:")
        print(f"-> Document Nodes: {self.num_docs}")
        print(f"-> Word Nodes:     {self.num_vocab}")
        print(f"-> Total Nodes:    {self.total_nodes}")
        
        return tfidf_matrix

    def get_node_id_maps(self):
        """
        Maps every document and every word to a specific integer index in our upcoming N x N matrix.
        Documents are 0 to (D-1). Words are D to (D+M-1).
        """
        doc_ids = {f"Doc_{i}": i for i in range(self.num_docs)}
        word_ids = {word: i + self.num_docs for i, word in enumerate(self.vocab)}
        
        return doc_ids, word_ids