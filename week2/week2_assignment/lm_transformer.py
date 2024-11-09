import re
import time
from collections import Counter

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class PositionalEncoding(nn.Module):
    """
    Applies positional encoding to the input sequence for the transformer model.
    """
    def __init__(self, max_len, d_model):
        """
        Initializes the positional encoding.

        Args:
        - max_len: Maximum length of the input sequence.
        - d_model: Dimensionality of the model embeddings.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input and applies dropout.
        """
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x


class TransformerArchitecture(nn.Module):
    """
    Defines the Transformer model for next-word prediction.
    """
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, context_size, ff_size=512):
        """
        Initializes the transformer architecture.

        Args:
        - vocab_size: Size of the vocabulary.
        - embed_dim: Dimensionality of embeddings.
        - num_layers: Number of transformer layers.
        - num_heads: Number of attention heads in each layer.
        - context_size: Maximum sequence length the model will process.
        - ff_size: Dimensionality of the feedforward network. Default is 512.
        """
        super(TransformerArchitecture, self).__init__()
        self.vocab_size = vocab_size
        self.pos_encoder = PositionalEncoding(max_len=context_size, d_model=embed_dim)
        self.emb = nn.Embedding(self.vocab_size, embed_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            batch_first=True,
            dropout=0.0
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embed_dim, self.vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """
        Generates a square mask to prevent attending to future tokens.
        This mask ensures that predictions for a given token do not use information from subsequent tokens.
        """
        mask = torch.triu(torch.ones(sz, sz)) == 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        """
        Forward pass through the transformer model.
        Encodes the input sequence, applies masking, and passes it through the transformer decoder.
        """
        emb = self.emb(x)
        input_mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
        x = self.pos_encoder(emb)
        x = self.decoder(x, memory=x, tgt_mask=input_mask, tgt_is_causal=True)
        out = self.linear(x)
        return out


class TransformerLanguageModel:
    """
    Language model class that integrates training, evaluation, and text generation using a transformer model.
    """
    def __init__(self, model, tok2idx, idx2tok, context_size, learning_rate, epochs, device):
        """
        Initializes the transformer-based language model.

        Args:
        - model: The transformer model to use.
        - tok2idx: Mapping from token to index.
        - idx2tok: Mapping from index to token.
        - context_size: Context size for the model.
        - learning_rate: Learning rate for optimization.
        - epochs: Number of training epochs.
        - device: Device for model training (e.g., 'cpu' or 'cuda').
        """
        self.model = model.to(device)
        self.tok2idx = tok2idx
        self.idx2tok = idx2tok
        self.context_size = context_size
        self.epochs = epochs
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_function = nn.CrossEntropyLoss()

    def tokenize_text(self, text, context_size=None):
        """
        Converts text to a list of token indices using the model's vocabulary.
        """
        words = text.split()
        if context_size is not None:
            return [self.tok2idx.get(word, self.tok2idx["<UNK>"]) for word in words[-context_size:]]
        else:
            return [self.tok2idx.get(word, self.tok2idx["<UNK>"]) for word in words]

    def sample_next(self, logits, temperature=1.0):
        """
        Samples the next token from the model's output logits based on the provided temperature.
        """
        scaled_logits = logits / temperature
        probabilities = F.softmax(scaled_logits[:, -1, :], dim=-1).cpu()
        predicted_index = torch.multinomial(probabilities, num_samples=1)
        return self.idx2tok[int(predicted_index.item())]

    def generate_text(self, context, max_length=100, temperature=1.0):
        """
        Generates text starting from the provided context up to a given maximum length.
        """
        sample = context
        for _ in range(max_length):
            int_vector = torch.tensor(self.tokenize_text(sample, context_size=self.context_size),
                                      dtype=torch.long).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out_logits = self.model(int_vector)
            next_token = self.sample_next(out_logits, temperature)
            sample += ' ' + next_token
            if next_token == "</s>":
                break
        return re.sub(r'\s([^\w\s](?:\s|$))', r'\1', sample)

    def perplexity(self, sequence):
        """
        Computes the perplexity of a given sequence using the trained model.
        """
        self.model.eval()
        int_vector = self.tokenize_text(sequence)
        text_samples = [int_vector[0:i + 1] for i in range(1, self.context_size)]
        text_samples.extend(
            [int_vector[i:i + self.context_size + 1] for i in range(len(int_vector) - self.context_size)])

        total_loss = 0
        for sample in text_samples:
            sample_tensor = torch.tensor(sample, dtype=torch.long).unsqueeze(0).to(self.device)
            target = sample_tensor.clone()

            input = sample_tensor[:, :-1]
            with torch.no_grad():
                out_logits = self.model(input)
            target = target[0, 1:]
            target[:-1] = -100
            loss = F.cross_entropy(out_logits.view(-1, out_logits.size(-1)), target.view(-1), ignore_index=-100)
            total_loss += loss

        total_perplexity = torch.exp(total_loss / len(text_samples))
        self.model.train()
        return total_perplexity.item()

    def train(self, dataloader, test_sentences):
        """
        Trains the transformer model on the provided data.
        """
        eval_perpl = float('inf')
        for epoch in range(self.epochs):
            start_time = time.time()
            total_loss = 0
            self.model.train()
            for context, target in dataloader:
                context, target = context.to(self.device), target.to(self.device)
                outputs = self.model(context)
                target = target.contiguous().view(-1)
                outputs = outputs.view(-1, self.model.vocab_size)
                loss = self.loss_function(outputs, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.detach().cpu().numpy()

            perplexity = sum([self.perplexity(sentence) for sentence in test_sentences]) / len(test_sentences)
            epoch_time = time.time() - start_time
            print(
                f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(dataloader)}, Perplexity: {perplexity:.2f}, Time: {epoch_time:.2f} s")

            INPUT = ["<s>", "<s>", "<s>"]
            for input in INPUT:
                generated_text = self.generate_text(
                    context=input,
                    max_length=30,
                    temperature=1.0
                )
                # Output the generated text
                print(f"Generated text: {generated_text}")

            if perplexity < eval_perpl:
                eval_perpl = perplexity
                self.save_model(f'transformer_model_context{self.context_size}_layers{self.model.decoder.num_layers}.pth')

    def save_model(self, path):
        """
        Saves the trained model to the specified file path.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path, map_location):
        """
        Loads a pre-trained model from the specified file path.
        """
        self.model.load_state_dict(torch.load(path, map_location=map_location))
        print(f"Model loaded from {path}")


class Dataset4Transformer(Dataset):
    """
    Dataset class for loading and preparing data for the transformer model.
    """
    def __init__(self, words, tok2idx, context_size):
        """
        Initializes the dataset with words, token mappings, and context size.

        Args:
        - words: List of tokenized words from the dataset.
        - tok2idx: Mapping from token to index.
        - context_size: Context size for input sequences.
        """
        self.words = words
        self.tok2idx = tok2idx
        self.context_size = context_size
        self.samples = [self.words[i:i + context_size + 1] for i in range(len(self.words) - context_size)]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns the input and target sequences for the given index.
        """
        sample = self.samples[idx]
        input_seq = torch.LongTensor([self.tok2idx[word] for word in sample[:-1]])
        target_seq = torch.LongTensor([self.tok2idx[word] for word in sample[1:]])
        return input_seq, target_seq


if __name__ == '__main__':
    # TODO: Modify these hyperparameters as part of the assignment
    N = 5 # TODO: change the value of this
    LAYERS = 2 # TODO: change the value of this

    ######

    # Additional hyperparameters and settings
    TRAIN = False # Set to True if training is needed, False for inference only
    CONTEXT_SIZE = N - 1
    MAX_LENGTH = 60
    TEMPERATURE = 1.0
    EPOCHS = 30
    LEARNING_RATE = 5e-05
    BATCH_SIZE = 32
    EMBED_DIM = 512
    NUM_HEADS = 1
    FF_SIZE = 512
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test sentences and training data
    with open("eval_berkeley_restaurant.txt", 'r') as f:
        test_sentences = f.read().splitlines()

    with open('processed_berkeley_restaurant.txt', 'r') as f:
        words = f.read().split()

    # Prepare vocabulary and mappings
    vocab = Counter(words)
    vocab = list(vocab.keys()) + ['<UNK>']
    tok2idx = {word: i for i, word in enumerate(vocab)}
    idx2tok = {i: word for word, i in tok2idx.items()}

    if TRAIN:
        # Create dataset and data loader
        dataset = Dataset4Transformer(words, tok2idx, context_size=CONTEXT_SIZE)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize transformer model
    transformer_model = TransformerArchitecture(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        num_layers=LAYERS,
        num_heads=NUM_HEADS,
        context_size=CONTEXT_SIZE,
        ff_size=FF_SIZE)

    # Initialize the language model with the transformer
    language_model = TransformerLanguageModel(transformer_model,
                                              tok2idx,
                                              idx2tok,
                                              CONTEXT_SIZE,
                                              LEARNING_RATE,
                                              EPOCHS,
                                              device=DEVICE)

    if TRAIN:
        # If TRAIN is set to True, the model will be trained using the provided data.
        language_model.train(dataloader, test_sentences)
    else:
        # If TRAIN is False, a pre-trained model will be loaded, and evaluation will be performed.
        language_model.load_model(f"transformer_model_N{N}_layers{LAYERS}.pth", DEVICE)
        # Calculate total parameters
        total_params = sum(p.numel() for p in language_model.model.parameters())
        print("\nTotal parameters in the model: ", total_params)
        # Evaluate the perplexity of the model on the test set
        perplexities = [language_model.perplexity(sentence) for sentence in test_sentences]
        avg_perplexity = sum(perplexities) / len(perplexities)
        print(f"\nPerplexity: {avg_perplexity:.2f}")
        print()

    print("Generated examples:\n")
    # Generate text using the trained or loaded model
    INPUT = ["<s>", "<s>", "<s>", "<s>", "<s>", "<s>", "<s>", "<s>", "<s>", "<s>"]
    for input in INPUT:
        generated_text = language_model.generate_text(context=input,
                                                      max_length=MAX_LENGTH,
                                                      temperature=TEMPERATURE)
        print(f"{generated_text}")
        print()
