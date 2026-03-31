import pandas as pd
from preprocessing import load_and_clean_data
from embedder import EmotionEmbedder
from graph_builder import TextGCNGraph

def main():
    print("=== TextGCN Pipeline Executing ===")
    
    # --- PHASE 1: Preprocessing ---
    print("\n[PHASE 1] Data Preprocessing")
    data_path = "../data/dummy_dataset.csv"
    cleaned_df = load_and_clean_data(data_path)
    
    # --- PHASE 2: Emotion Representation ---
    print("\n[PHASE 2] Emotion Representation Extraction")
    embedder = EmotionEmbedder(model_name="mental/mental-bert-base-uncased")
    embedded_df = embedder.process_dataset(cleaned_df)
    
    # --- PHASE 3: Graph Construction (Part 1: TF-IDF) ---
    print("\n[PHASE 3] Graph Construction: TF-IDF")
    graph_builder = TextGCNGraph(embedded_df)
    tfidf_matrix = graph_builder.build_tfidf_edges()
    
    # Verification Check
    doc_ids, word_ids = graph_builder.get_node_id_maps()
    print("\n--- Verification Check: Graph Nodes ---")
    print("Sample of extracted Word Nodes (checking for emojis/emoticons):")
    
    # Print a few words from our vocabulary to prove the emojis survived
    vocab_list = list(word_ids.keys())
    for word in vocab_list:
        if any(char in word for char in [':', '/', '💪']):
            print(f"SUCCESS -> Preserved Affective Token: {word}")

if __name__ == "__main__":
    main()