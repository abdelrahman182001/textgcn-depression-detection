import tensorflow as tf
from spektral.layers import GCNConv

class TextGCNModel(tf.keras.Model):
    def __init__(self, num_classes=2, hidden_dim=200, dropout_rate=0.5):
        super().__init__()
        print(f"--- Initializing TextGCN Architecture ---")
        print(f"-> Hidden Dimension: {hidden_dim}")
        print(f"-> Output Classes: {num_classes}")
        
        # Layer 1: The Hidden Feature Extractor
        self.gcn1 = GCNConv(hidden_dim, activation='relu')
        
        # Dropout: Randomly turns off neurons during training to prevent overfitting
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Layer 2: The Classifier (Depressed vs. Non-Depressed)
        self.gcn2 = GCNConv(num_classes, activation='softmax')

    def call(self, inputs):
        """
        The forward pass of the neural network. 
        Spektral expects a list containing [Node Features (X), Adjacency Matrix (A)]
        """
        x, a = inputs
        
        # Pass data through Layer 1
        x = self.gcn1([x, a])
        x = self.dropout(x)
        
        # Pass data through Layer 2
        x = self.gcn2([x, a])
        
        return x