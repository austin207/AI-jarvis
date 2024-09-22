import torch
import os
import sys
import json

# Load config file
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.model import GPTModel
from tokenizers import ByteLevelBPETokenizer

# Load tokenizer from paths defined in config.json
vocab_path = config['data']['tokenizer_files']['vocab']
merges_path = config['data']['tokenizer_files']['merges']

tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)

# Load model configuration from config.json
vocab_size = tokenizer.get_vocab_size()
max_seq_len = config['model']['max_seq_len']
embedding_dim = config['model']['embedding_dim']
num_heads = config['model']['num_heads']
num_layers = config['model']['num_layers']
dropout = config['model']['dropout']

# Initialize the model using config parameters
model = GPTModel(
    vocab_size=vocab_size, 
    embedding_dim=embedding_dim, 
    num_heads=num_heads, 
    num_layers=num_layers, 
    max_seq_len=max_seq_len, 
    dropout=dropout
)

# Load pre-trained model weights from path defined in config.json
model_path = config['model']['checkpoint']
model.load_state_dict(torch.load(model_path))
model.eval()

# Sample next token using temperature and top-k sampling
def sample_next_token(logits, temperature=1.0, top_k=10):
    # Apply temperature scaling
    logits = logits / temperature

    # Ensure top_k does not exceed the number of logits (vocabulary size)
    top_k = min(top_k, logits.size(-1))

    # Get the top-k logits and their corresponding token indices
    top_k_logits, top_k_indices = torch.topk(logits, top_k)

    # Convert logits to probabilities
    probabilities = torch.softmax(top_k_logits, dim=-1)

    # Sample a token from the top-k probabilities
    next_token_index = torch.multinomial(probabilities, num_samples=1)

    # Return the actual token id
    return top_k_indices[0, next_token_index.item()].item()

# Text generation function with temperature and top-k sampling
def generate_text(model, prompt, max_length=50, temperature=1.0, top_k=10):
    model.eval()

    # Tokenize the input prompt
    tokens = tokenizer.encode(prompt).ids
    tokens = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension [1, seq_len]

    # Generate tokens iteratively
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(tokens)  # Get the model outputs for current tokens
            next_token_logits = outputs[:, -1, :]  # Get logits for the last token

            # Sample next token using temperature and top-k
            next_token = sample_next_token(next_token_logits, temperature=temperature, top_k=top_k)

        # Add the new token to the sequence
        tokens = torch.cat([tokens, torch.tensor([[next_token]])], dim=1)

        # Break if EOS token is generated
        if next_token == tokenizer.token_to_id("</s>"):  # EOS token may vary, use the correct one
            break

    # Decode the token ids back to text
    generated_ids = tokens.squeeze().tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)  # Skip special tokens like <eos>, etc.

    return generated_text

# Interactive prompt and generate response
def prompt_and_generate():
    while True:
        prompt = input("Enter a prompt (type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break

        # Call the generate_text function with default temperature and top_k
        generated_text = generate_text(model, prompt, max_length=100, temperature=0.8, top_k=10)
        print(f"Model response: {generated_text}\n")

if __name__ == "__main__":
    prompt_and_generate()
