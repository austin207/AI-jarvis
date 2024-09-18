import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, file_path, max_seq_len):
        self.max_seq_len = max_seq_len
        
        # Load the tokenized data from the file
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Convert the text lines into lists of integers (token IDs)
        self.tokenized_data = [list(map(int, line.strip().split())) for line in lines]
        
    def __len__(self):
        # Return the number of sequences
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        # Get the sequence at the given index
        tokenized_sequence = self.tokenized_data[idx]
        
        # Pad or truncate the sequence to ensure it's exactly max_seq_len tokens
        if len(tokenized_sequence) < self.max_seq_len:
            # Padding (if the sequence is too short)
            tokenized_sequence += [0] * (self.max_seq_len - len(tokenized_sequence))  # Assuming 0 is the padding token
        else:
            # Truncate (if the sequence is too long)
            tokenized_sequence = tokenized_sequence[:self.max_seq_len]

        # Convert the token list to a tensor
        return torch.tensor(tokenized_sequence)

# Example usage
if __name__ == "__main__":
    dataset = TextDataset(file_path=r"(Path of tokenised data from processed dataset)", max_seq_len=1024)
    print(f"Number of sequences: {len(dataset)}")
    print(f"First sequence: {dataset[0]}")
