import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import sys
import os
import json

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Load config file
config_path = os.path.join(project_root, 'config', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Import your model and dataset
from models.model import GPTModel
from dataset import TextDataset

# Hyperparameters loaded from config.json
learning_rate = config['training']['learning_rate']
num_epochs = config['training']['num_epochs']
batch_size = config['training']['batch_size']
max_seq_len = config['data']['max_sequence_length']
vocab_size = config['model']['vocab_size']  # Should match your tokenizer vocab size

# Initialize model
model = GPTModel(
    vocab_size=vocab_size, 
    embedding_dim=config['model']['embedding_dim'], 
    num_heads=config['model']['num_heads'], 
    num_layers=config['model']['num_layers'], 
    max_seq_len=max_seq_len, 
    dropout=config['model']['dropout']
)

# Initialize optimizer and loss function
optimizer = Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Load data
train_dataset = TextDataset(file_path=config['data']['processed']['tokenized_data'], max_seq_len=max_seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Learning rate scheduler (optional but useful)
total_steps = num_epochs * len(train_loader)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['training'].get('num_warmup_steps', 0), num_training_steps=total_steps)

# Define log file path from config.json
# Define log file path from config.json
log_dir = os.path.dirname(config['training']['training_logs'])  # This gives you the directory path
os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists
log_file_path = config['training']['training_logs']  # Full path to the log file
log_file_path = os.path.join(log_dir, 'training_log.txt')

# Define checkpoint directory and file paths
checkpoint_dir = config['model']['checkpoints']
os.makedirs(checkpoint_dir, exist_ok=True)

# Function to save checkpoints
def save_checkpoint(epoch, model, optimizer, loss, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# Function to load checkpoint
def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    return start_epoch

# Check if a checkpoint exists and load it
resume_from_checkpoint = config['training'].get('resume_from_checkpoint', False)
start_epoch = 0
if resume_from_checkpoint:
    checkpoint_path = config['training'].get('checkpoint_path', '')
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer)

print('Training initiated...')

# Training function
def train_model(model, dataloader, optimizer, loss_fn, scheduler, device, log_file_path, start_epoch=0):
    model.train()
    model.to(device)
    
    with open(log_file_path, 'w' if start_epoch == 0 else 'a') as log_file:  # Append to log file if resuming
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            
            for batch in dataloader:
                inputs = batch[:, :-1].to(device)
                labels = batch[:, 1:].to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)

                # Reshape output and compute loss
                loss = loss_fn(outputs.reshape(-1, vocab_size), labels.reshape(-1))  # Updated reshape

                # Backpropagation and optimization
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()

            average_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}, Average Loss: {average_epoch_loss}")
            
            # Write to log file
            log_file.write(f"{epoch + 1} {epoch_loss:.4f}\n")
            
            # Save checkpoint at regular intervals (e.g., every 2 epochs)
            if (epoch + 1) % 2 == 0:
                save_checkpoint(epoch, model, optimizer, epoch_loss, checkpoint_dir)
    
    print('Session Finished')

# Device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Start training
train_model(model, train_loader, optimizer, loss_fn, scheduler, device, log_file_path, start_epoch)

# Save the final trained model
model_save_path = os.path.join(config['model']['checkpoints'], 'gpt_model_final.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
