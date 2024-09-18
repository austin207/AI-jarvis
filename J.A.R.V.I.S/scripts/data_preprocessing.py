import os
from transformers import GPT2Tokenizer

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Input and output file paths
input_file = r"C:\Users\mrult\ai_project\data\raw\dataset.txt"
output_dir = r"C:\Users\mrult\ai_project\data\processed"
output_file = os.path.join(output_dir, "tokenized_data.txt")
tokenizer_dir = os.path.join(output_dir, "tokenizer")  # Directory to save tokenizer files

# Ensure the output and tokenizer directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(tokenizer_dir, exist_ok=True)

def preprocess_data(input_file, output_file, tokenizer_dir):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    tokenized_data = []
    for line in lines:
        # Tokenize the line
        tokens = tokenizer.encode(line.strip())
        tokenized_data.append(tokens)

    # Save tokenized data to file
    with open(output_file, 'w') as out_f:
        for tokens in tokenized_data:
            out_f.write(" ".join(map(str, tokens)) + '\n')

    # Save the tokenizer vocab.json and merges.txt
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"Tokenizer files (vocab.json, merges.txt) saved to {tokenizer_dir}")

if __name__ == "__main__":
    preprocess_data(input_file, output_file, tokenizer_dir)
    print(f"Tokenized data saved to {output_file}")
