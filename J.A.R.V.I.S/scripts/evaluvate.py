import torch
from torch.utils.data import DataLoader
from models.model import GPTModel
from dataset import TextDataset
import math

# Hyperparameters
batch_size = 8
max_seq_len = 1024
vocab_size = 50257

# Load trained model
model = GPTModel(vocab_size=vocab_size, embedding_dim=768, num_heads=12, num_layers=12, max_seq_len=max_seq_len, dropout=0.1)
model.load_state_dict(torch.load('(Path of final model)'))
model.eval()

# Load dataset
test_dataset = TextDataset(file_path="(Path of tokenised data)", max_seq_len=max_seq_len)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[:, :-1].to(device)
            labels = batch[:, 1:].to(device)
            
            outputs = model(inputs)
            loss_fn = nn.CrossEntropyLoss()
            
            loss = loss_fn(outputs.view(-1, vocab_size), labels.view(-1))
            total_loss += loss.item() * inputs.size(0)
            total_tokens += inputs.size(0) * inputs.size(1)
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    print(f"Average Loss: {avg_loss}, Perplexity: {perplexity}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run evaluation
evaluate_model(model, test_loader, device)
