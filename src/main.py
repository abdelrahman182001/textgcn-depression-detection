import pandas as pd
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from preprocessing import load_and_clean_data
from embedder import EmotionEmbedder
from graph_builder import TextGCNGraph
from model import TextGCNModel

# --- Custom Graph Metrics ---
def masked_loss(y_true, y_pred, mask):
    """Calculates Cross-Entropy Loss only on the unmasked (Document) nodes."""
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)  # Normalize
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(y_true, y_pred, mask):
    """Calculates Accuracy only on the unmasked (Document) nodes."""
    correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy_all = tf.cast(correct_predictions, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def main():
    print("=== TextGCN Pipeline Executing ===")
    
    # --- PHASE 1: Preprocessing ---
    data_path = "../data/dummy_dataset.csv"
    cleaned_df = load_and_clean_data(data_path)
    
    # --- PHASE 2: Emotion Representation ---
    embedder = EmotionEmbedder(model_name="mental/mental-bert-base-uncased")
    embedded_df = embedder.process_dataset(cleaned_df)
    
    # --- PHASE 3: Graph Construction ---
    graph_builder = TextGCNGraph(embedded_df)
    tfidf_matrix = graph_builder.build_tfidf_edges()
    pmi_edges = graph_builder.build_pmi_edges(window_size=20)
    A_matrix = graph_builder.build_adjacency_matrix(pmi_edges)

    # --- PHASE 4: Deep Learning Architecture ---
    num_docs = graph_builder.num_docs
    num_words = graph_builder.num_vocab
    total_nodes = graph_builder.total_nodes
    
    doc_features = np.vstack(embedded_df['doc_embedding'].values)
    word_features = np.zeros((num_words, 768))
    X_matrix = np.vstack([doc_features, word_features])
    
    X_tf = tf.convert_to_tensor(X_matrix, dtype=tf.float32)
    
    A_coo = sp.csr_matrix(A_matrix).tocoo()
    indices = np.column_stack((A_coo.row, A_coo.col))
    A_tf = tf.sparse.SparseTensor(
        indices=indices,
        values=A_coo.data.astype(np.float32),
        dense_shape=A_coo.shape
    )
    A_tf = tf.sparse.reorder(A_tf)
    
    model = TextGCNModel(num_classes=2, hidden_dim=200, dropout_rate=0.5)

    # --- PHASE 5: THE TRAINING LOOP ---
    print("\n[PHASE 5] Initializing Training Loop")
    
    # 1. Define the Ground Truth Labels for our 5 dummy tweets
    # 1 = Depressed, 0 = Non-Depressed
    raw_labels = [1, 0, 1, 0, 1] 
    
    # Convert to One-Hot Encoding (e.g., 1 -> [0, 1], 0 -> [1, 0])
    doc_labels = tf.one_hot(raw_labels, depth=2).numpy()
    
    # Pad labels with zeros for the word nodes (they aren't graded)
    word_labels = np.zeros((num_words, 2))
    Y_matrix = np.vstack([doc_labels, word_labels])
    Y_tf = tf.convert_to_tensor(Y_matrix, dtype=tf.float32)
    
    # 2. Create the Training Mask (True for docs, False for words)
    train_mask = np.zeros(total_nodes, dtype=bool)
    train_mask[:num_docs] = True
    mask_tf = tf.convert_to_tensor(train_mask)
    
    # 3. Setup the Optimizer (Adam is the industry standard)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
    
    print("Commencing Training (200 Epochs)...")
    epochs = 200
    
    for epoch in range(epochs):
        # Start recording the math
        with tf.GradientTape() as tape:
            # 1. Forward Pass (Make a guess)
            predictions = model([X_tf, A_tf], training=True)
            
            # 2. Calculate the Error (Loss) using the mask
            loss = masked_loss(Y_tf, predictions, mask_tf)
        
        # 3. Calculate the gradients (How much to change the weights)
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # 4. Apply the updates
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # 5. Track Accuracy
        acc = masked_accuracy(Y_tf, predictions, mask_tf)
        
        # Print progress every 20 epochs
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

    print("\n[SUCCESS] Model Training Complete!")
    
    # --- FINAL TEST ---
    print("\n--- Final Predictions on Training Data ---")
    final_preds = model([X_tf, A_tf], training=False)
    for i in range(num_docs):
        pred_class = np.argmax(final_preds[i].numpy())
        confidence = np.max(final_preds[i].numpy()) * 100
        real_class = raw_labels[i]
        print(f"Document {i+1} -> Predicted: Class {pred_class} ({confidence:.1f}% confidence) | Actual: Class {real_class}")

if __name__ == "__main__":
    main()