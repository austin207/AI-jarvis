import torch
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.model import GPTModel
from tokenizers import ByteLevelBPETokenizer

# Load tokenizer
tokenizer = ByteLevelBPETokenizer(
    r"(Path of vocab.json in processed dataset)", 
    r"(Path of merges.txt in processed dataset)"
)

# Load model
vocab_size = tokenizer.get_vocab_size()
max_seq_len = 1024

model = GPTModel(vocab_size=vocab_size, embedding_dim=768, num_heads=12, num_layers=12, max_seq_len=max_seq_len, dropout=0.1)
model.load_state_dict(torch.load(r"(Path of trained model from checkpoints for testing)"))
model.eval()

# Text generation function
def generate_text(model, prompt, max_length=50):
    model.eval()
    
    # Tokenize input prompt
    tokens = tokenizer.encode(prompt).ids
    tokens = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension [1, seq_len]

    # Generate tokens iteratively
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(tokens)  # Get the model outputs for current tokens
            next_token_logits = outputs[:, -1, :]  # Get logits for the last token
            next_token = torch.argmax(next_token_logits, dim=-1).item()  # Get the token with max probability

        # Add the new token to the sequence
        tokens = torch.cat([tokens, torch.tensor([[next_token]])], dim=1)

        # Break if EOS token is generated
        if next_token == tokenizer.token_to_id("</s>"):  # EOS token may vary, use the correct one
            break

    # Decode the token ids back to text
    generated_ids = tokens.squeeze().tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)  # Skip special tokens like <eos>, etc.

    return generated_text

# Generate text with a prompt
prompt = "Once upon a time"
generated_text = generate_text(model, prompt, max_length=100)
print("Generated text: ", generated_text)
