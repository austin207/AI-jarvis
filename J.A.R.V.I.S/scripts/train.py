import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import your model and dataset
from models.model import GPTModel
from dataset import TextDataset

# Hyperparameters
learning_rate = 5e-5
num_epochs = 5
batch_size = 16
max_seq_len = 1024
vocab_size = 50257  # Should match your tokenizer vocab size

# Initialize model
model = GPTModel(vocab_size=vocab_size, embedding_dim=768, num_heads=12, num_layers=12, max_seq_len=max_seq_len, dropout=0.1)

# Initialize optimizer and loss function
optimizer = Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Load data
train_dataset = TextDataset(file_path=r"(Path of tokenised data from processed dataset)", max_seq_len=max_seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Learning rate scheduler (optional but useful)
total_steps = num_epochs * len(train_loader)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)

# Define log file path
log_dir = r'(Path to save logs in logs directory  for plotting learning rate graph)'
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'training_log.txt')

# Training function
def train_model(model, dataloader, optimizer, loss_fn, scheduler, device, log_file_path):
    model.train()
    model.to(device)
    
    with open(log_file_path, 'w') as log_file:  # Open log file for writing
        for epoch in range(num_epochs):
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
            print(f"Epoch [{epoch+1}/{num_epochs}],Loss: {epoch_loss}, Average Loss: {average_epoch_loss}")
            
            # Write to log file
            log_file.write(f"{epoch + 1} {epoch_loss:.4f}\n")


# Device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Start training
train_model(model, train_loader, optimizer, loss_fn, scheduler, device, log_file_path)

# Save the trained model
model_save_path = r'(Path to save trained model for testing in checkpoints)'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

