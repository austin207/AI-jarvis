# AI Project - GPT Model

# AI Project Overview

## Directory Structure
- `data/`: Stores raw and processed datasets.
- `models/`: Stores model code and checkpoints.
- `scripts/`: Contains training, evaluation, and inference scripts.
- `results/`: Stores logs, figures, and other results.
- `config/`: Contains model and training configuration files.


This project implements a custom GPT (Generative Pre-trained Transformer) model for text generation using the PyTorch framework. The project covers all steps from data preprocessing to training, evaluation, inference, and visualizing training performance.

## Project Structure

```
ai_project/
│
├── config/
│   └── config.json
├── data/
│   ├── raw/
│   │   └── dataset.txt           # Raw dataset for tokenization
│   └── processed/
│       ├── tokenized_data.txt     # Tokenized dataset
│       └── tokenizer/
│           ├── vocab.json         # Tokenizer vocabulary
│           └── merges.txt         # Byte Pair Encoding (BPE) merges
├── models/
│   ├── checkpoints/
│   │   └── gpt_model.pth          # Trained model weights
│   └── final/
│   └── model.py                   # GPT model architecture
├── results/
│   ├── logs/
│   │   └── training_log.txt       # Training logs for tracking loss per epoch
│   └── figures/
│       ├── Training_Loss_Curve_N.png  # Loss curve plots per training session
│       └── plot_counter.txt       # Counter to track the plot numbering
├── scripts/
│   ├── data_preprocessing.py      # Data preprocessing and tokenization script
│   ├── dataset.py                 # Custom dataset class for tokenized data
│   ├── evaluate.py                # Model evaluation script for loss and perplexity
│   ├── inference.py               # Text generation script (with sampling techniques)
│   ├── train.py                   # Model training script
│   ├── Training_loss_graph_plotter.py  # Script to plot training loss
├── README.md                      # Project description and instructions
└── requirements.txt               # Required Python libraries
```

## Requirements(For this project)

To run the project, you need to install the following Python libraries:

- `torch`
- `transformers`
- `matplotlib`

You can install them using the following command:

```bash
pip install torch transformers matplotlib
```

## Data Preparation

### 1. Dataset

The raw dataset should be stored in `data/raw/dataset.txt`. This file will be tokenized and prepared for training.

### 2. Tokenization

The `data_preprocessing.py` script tokenizes the raw dataset using the GPT-2 Byte-Level BPE tokenizer. It also saves the tokenizer files (`vocab.json`, `merges.txt`) in the `data/processed/tokenizer/` folder and the tokenized data in `data/processed/tokenized_data.txt`.

**Usage**:

```bash
python scripts/data_preprocessing.py
```

### Tokenizer Output

- `vocab.json`: Vocabulary file for the tokenizer.
- `merges.txt`: BPE merges file.
- `tokenized_data.txt`: The tokenized dataset.

## Model Architecture

The model is defined in `models/model.py`. It is a custom GPT architecture that includes the following components:

- **Token Embedding Layer**: Maps token IDs to embeddings.
- **Positional Embedding Layer**: Provides positional information for the input tokens.
- **Transformer Layers**: A stack of transformer encoder layers that process the input tokens.
- **Output Layer**: A final linear layer to project the outputs to the vocabulary size for token prediction.

### Hyperparameters

- `vocab_size`: The size of the vocabulary (from the tokenizer).
- `embedding_dim`: Dimension of token embeddings.
- `num_heads`: Number of attention heads.
- `num_layers`: Number of transformer layers.
- `max_seq_len`: Maximum sequence length.
- `dropout`: Dropout rate.

## Training the Model

The training script `train.py` is used to train the GPT model on the tokenized data.

### Training Parameters

- **Learning rate**: 5e-5
- **Batch size**: 8
- **Number of epochs**: 5
- **Optimizer**: Adam optimizer with weight decay
- **Scheduler**: A linear learning rate scheduler with warm-up steps

Training logs are saved in `results/logs/training_log.txt`, where each line contains the epoch number and the corresponding loss.

**Usage**:

```bash
python scripts/train.py
```

The model weights are saved in `models/checkpoints/gpt_model.pth` after training.

## Evaluation

The `evaluate.py` script calculates the average loss and perplexity of the model on the test dataset.

**Usage**:

```bash
python scripts/evaluate.py
```

The model evaluation outputs the average loss and perplexity after processing the dataset.

## Text Generation

The `inference.py` script generates text based on a given prompt. It uses **temperature scaling** and **top-k sampling** for diverse text generation.

### Parameters for Text Generation

- **Temperature**: Controls the randomness of predictions by scaling logits before applying softmax. A lower temperature (e.g., 0.8) makes the model more conservative, while a higher temperature (e.g., 1.0 or higher) produces more random text.
- **Top-k Sampling**: Limits the sampling pool to the top-k tokens with the highest probability.

**Usage**:

```bash
python scripts/inference.py
```

The script interactively asks for a prompt and generates a continuation of the input text.

## Plotting Training Loss

The `Training_loss_graph_plotter.py` script generates plots for the training loss recorded in `training_log.txt`. The plots are saved as images in `results/figures/`.

**Usage**:

```bash
python scripts/Training_loss_graph_plotter.py
```

Each plot represents the loss over a session of 5 epochs. The filenames are automatically generated, and the counter for the plot numbers is tracked in `plot_counter.txt`.

## Custom Dataset Class

The `dataset.py` script defines the `TextDataset` class, which loads the tokenized dataset, pads/truncates sequences to a fixed length, and converts them into PyTorch tensors for training.

## How to Run the Project

1. **Tokenize the Dataset**: Run the data preprocessing script to tokenize your dataset.

   ```bash
   python scripts/data_preprocessing.py
   ```

2. **Train the Model**: Run the training script to train the model. Training logs will be saved for later use.

   ```bash
   python scripts/train.py
   ```

3. **Evaluate the Model**: Evaluate the trained model to get the average loss and perplexity.

   ```bash
   python scripts/evaluate.py
   ```

4. **Generate Text**: Interactively generate text based on user input.

   ```bash
   python scripts/inference.py
   ```

5. **Plot the Training Loss**: Generate plots of training loss for each session.

   ```bash
   python scripts/Training_loss_graph_plotter.py
   ```

## How to Extend the Project

This project is highly modular and can be extended or customized in various ways, depending on your requirements. Below are some common ways you can extend or modify the project:

### 1. Training on Custom Datasets

To train the model on your own dataset, simply replace the `dataset.txt` file in the `data/raw/` directory with your new text data.

- Ensure the new dataset follows a similar format (i.e., plain text).
- Run the `data_preprocessing.py` script to tokenize the new dataset:

  ```bash
  python scripts/data_preprocessing.py
  ```

### 2. Changing Model Hyperparameters

You can modify key model parameters such as the number of layers, the size of embeddings, and the number of attention heads by editing the `config/config.json` file. This allows you to experiment with different model architectures without changing the code.

For example, to increase the number of transformer layers or heads, update the following parameters in `config/config.json`:

```json
{
  "num_layers": 24,
  "num_heads": 16,
  "embedding_dim": 1024
}
```

After making the changes, rerun the `train.py` script to train the model with the updated configuration.

### 3. Modifying the Model Architecture

The `models/model.py` file contains the architecture for the GPT model. You can add more layers, modify the feedforward dimension, or experiment with different activation functions.

For example, if you want to change the feedforward dimension in the transformer layers, locate this section in `model.py`:

```python
nn.TransformerEncoderLayer(
    d_model=embedding_dim,
    nhead=num_heads,
    dim_feedforward=embedding_dim * 4,  # You can adjust this value
    dropout=dropout
)
```

Change the `dim_feedforward` parameter to a value that suits your model's complexity.

### 4. Experimenting with Dropout and Regularization

Dropout helps prevent overfitting, and you can experiment with different dropout rates by modifying the `dropout` parameter in `config/config.json`. A typical value to start with is `0.1`, but you can adjust it depending on how well the model generalizes.

```json
{
  "dropout": 0.2
}
```

### 5. Changing Optimization Strategy

The current project uses the Adam optimizer and a linear learning rate scheduler. If you wish to experiment with different optimization strategies (e.g., RMSprop or using learning rate schedules), you can modify the `train.py` script.

For instance, to switch to the RMSprop optimizer, change this section of the `train.py` script:

```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
```

Similarly, you can experiment with learning rate schedulers, like cosine annealing:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
```

### 6. Fine-Tuning a Pre-Trained GPT Model

If you want to fine-tune a pre-trained GPT model (e.g., GPT-2), you can load the pre-trained weights from Hugging Face's `transformers` library and then further train it on your dataset.

You can modify `model.py` to load a pre-trained model like this:

```python
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2')
```

### 7. Extending the Inference Capabilities

The `inference.py` script currently supports temperature scaling and top-k sampling for text generation. You can extend it to include top-p (nucleus) sampling, beam search, or other decoding strategies.

For example, to add **top-p sampling** in `inference.py`, add the following:

```python
outputs = model.generate(input_ids, max_length=100, do_sample=True, top_p=0.9, temperature=0.8)
```

### 8. Model Checkpointing and Resuming Training

To save time and resources, especially during long training sessions, you can modify the `train.py` script to save checkpoints at regular intervals and resume training from a checkpoint.

You can use PyTorch's `torch.save()` and `torch.load()` functions for this purpose:

```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, f'checkpoint_{epoch}.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## Conclusion

This project demonstrates how to build, train, and evaluate a GPT model for text generation. You can fine-tune the model, try different datasets, or experiment with hyperparameters to see how the model performs. The project also includes a script to visualize training loss over time for performance monitoring.

Feel free to modify and extend the project as per your needs!

## Requirements

To set up and run this project, you need to install the following dependencies:

### Python Version
- Python 3.8 or higher

### Libraries and Packages

The required Python libraries are listed below. You can install them using `pip`:

```bash
pip install torch torchvision torchaudio transformers tokenizers matplotlib
```

#### Explanation of Required Libraries:

- **torch**: PyTorch, the deep learning framework used for building and training the GPT model.
- **torchvision** and **torchaudio**: Standard libraries that complement PyTorch (optional but commonly included).
- **transformers**: Provides pre-trained models and tokenizers, specifically for GPT and other transformer-based models.
- **tokenizers**: Library to efficiently tokenize text data, used for loading and handling the Byte-Level BPE tokenizer.
- **matplotlib**: Used to generate training loss curves and visualizations.

### Installing Requirements

To ensure all dependencies are installed, you can also create a `requirements.txt` file with the following content:

```txt
torch
transformers
tokenizers
matplotlib
```

Then, run the following command to install all required packages:

```bash
pip install -r requirements.txt
```

### Additional Requirements

- **GPU (optional)**: For faster model training, a CUDA-enabled GPU is recommended.
- **CUDA Toolkit (optional)**: If you are using a GPU, make sure you have CUDA drivers installed.

## Configuration

The project includes a `config/config.json` file that stores various hyperparameters and settings for training the GPT model. This allows you to easily modify the configuration without changing the code.

### Example `config.json`:

```json
{
  "learning_rate": 5e-5,
  "num_epochs": 5,
  "batch_size": 8,
  "max_seq_len": 1024,
  "vocab_size": 50257,
  "embedding_dim": 768,
  "num_heads": 12,
  "num_layers": 12,
  "dropout": 0.1,
  "num_warmup_steps": 500
}
```

### Configuration Parameters:

- `learning_rate`: The learning rate for the optimizer.
- `num_epochs`: Number of epochs for training.
- `batch_size`: Size of each mini-batch during training.
- `max_seq_len`: Maximum length of input sequences.
- `vocab_size`: The size of the tokenizer vocabulary.
- `embedding_dim`: Dimensionality of the token and positional embeddings.
- `num_heads`: Number of attention heads in each transformer layer.
- `num_layers`: Number of transformer layers.
- `dropout`: Dropout rate used in the model.
- `num_warmup_steps`: Number of warm-up steps for the learning rate scheduler.

### Using the Configuration

To use the configuration in your scripts, simply load the `config.json` file and extract the necessary parameters. This ensures that all key hyperparameters and settings are stored in one place, making the training process more flexible and easier to manage.
