import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, max_seq_len, dropout):
        super(GPTModel, self).__init__()
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final output layer
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Store max sequence length for positional encoding
        self.max_seq_len = max_seq_len
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()

        # Create positional indices
        positions = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, seq_len).to(input_ids.device)

        # Embedding lookup: token + positional embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(positions)
        x = token_embeds + position_embeds
        x = self.dropout(x)

        # Pass through each transformer layer
        for layer in self.layers:
            x = layer(x)
        
        # Final linear layer to project to vocab size
        logits = self.fc_out(x)
        
        return logits

    # Weight initialization function
    def init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
