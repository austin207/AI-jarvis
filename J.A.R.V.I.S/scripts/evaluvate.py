import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import os
import sys
import json

# Load config file
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Set project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.model import GPTModel
from dataset import TextDataset

# Load parameters from config.json
batch_size = config['training']['batch_size']
max_seq_len = config['model']['max_seq_len']
vocab_size = config['model']['vocab_size']
embedding_dim = config['model']['embedding_dim']
num_heads = config['model']['num_heads']
num_layers = config['model']['num_layers']
dropout = config['model']['dropout']
model_checkpoint_path = config['model']['checkpoint']
tokenized_data_path = config['data']['processed']['tokenized_data']

# Load trained model using hyperparameters from config
model = GPTModel(
    vocab_size=vocab_size, 
    embedding_dim=embedding_dim, 
    num_heads=num_heads, 
    num_layers=num_layers, 
    max_seq_len=max_seq_len, 
    dropout=dropout
)
model.load_state_dict(torch.load(model_checkpoint_path))
model.eval()

# Load dataset
test_dataset = TextDataset(file_path=tokenized_data_path, max_seq_len=max_seq_len)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_tokens = 0

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[:, :-1].to(device)  # All tokens except the last
            labels = batch[:, 1:].to(device)  # All tokens except the first

            # Get model outputs
            outputs = model(inputs)

            # Reshape outputs and labels for the loss function
            # outputs.shape: [batch_size, seq_len-1, vocab_size]
            # labels.shape: [batch_size, seq_len-1]
            outputs = outputs.reshape(-1, vocab_size)  # Reshape to [batch_size * seq_len-1, vocab_size]
            labels = labels.reshape(-1)  # Reshape to [batch_size * seq_len-1]

            # Calculate loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * inputs.size(0)  # Multiply by batch size
            total_tokens += inputs.size(0) * inputs.size(1)  # Total number of tokens processed
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    print(f"Average Loss: {avg_loss}, Perplexity: {perplexity}")

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run evaluation
evaluate_model(model, test_loader, device)
