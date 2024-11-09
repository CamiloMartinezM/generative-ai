import random
import re
from collections import Counter, defaultdict

class Ngram:
    def __init__(self, n, corpus_file_path):
        """
        Initializes the N-gram model.
        Args:
        - n: N in the N-gram model (e.g., 3 for trigrams).
        - corpus_file_path: Path to the input corpus file.
        """
        self.n = n
        self.all_tokens = []  # Stores all tokens
        self.context_counters = defaultdict(Counter)  # Tracks counts of word sequences (contexts)

        with open(corpus_file_path, 'r') as file:
            self.text = file.readlines()

    def train(self):
        """
        Trains the N-gram model by counting how often each word follows a given context.
        """
        for line in self.text:
            tokens = line.split()
            self.all_tokens.extend(tokens)

            for position in range(len(tokens)):
                for order in range(1, self.n + 1):
                    if position - order + 1 >= 0:
                        context = ' '.join(tokens[(position - order + 1):position])
                        word = tokens[position]
                        self.context_counters[context][word] += 1

        self.context_counters['']['<s>'] = 0  # Prevent '<s>' from being generated

    def get_context(self, position, source):
        """
        Extracts the context preceding the current position, based on the N-gram order
        Uses backoff to find the longest context that has been seen during training.
        """
        context_length = self.n - 1
        if position < context_length:
            context_length = position

        context = ' '.join(source[(position - context_length): position])
        while context not in self.context_counters and context_length > 0:
            context_length -= 1
            context = ' '.join(source[(position - context_length): position])

        return context

    def sample_word_given_context(self, context):
        """
        Samples a word from the given context.
        """
        words = list(self.context_counters[context].keys())  # List of words for the given context
        weights = list(self.context_counters[context].values())  # Corresponding frequencies (weights)

        return random.choices(words, weights=weights, k=1)[0]

    def sample_word(self, position, source):
        """
        Samples the next word at a given position.
        """
        context = self.get_context(position, source)
        return self.sample_word_given_context(context)

    def generate_sentence(self, context, max_length=100):
        """
        Generates a sentence starting from the given context.
        """
        output = context.split()
        for position in range(len(output), len(output) + max_length):
            token = self.sample_word(position, source=output)
            output.append(token)
            if token == '</s>':
                break

        output = " ".join(output)
        return re.sub(r'\s([^\w\s](?:\s|$))', r'\1', output)  # Fix punctuation spacing


if __name__ == '__main__':
    # Number of parameters for a n=1 model, V=50,000
    V = 50_000
    n = 3
    print(f"Number of parameters for a n={n} model with V={V}: {V**n}")
    exponents = V**n / 10**12
    print(f"Number of parameters for a n={n} model with V={V}: {V**n / 10**12} x 10^12")
    
    # Initialize the N-gram model with the n-gram order and corpus file path
    ngram = Ngram(n=3, corpus_file_path='processed_berkeley_restaurant.txt')

    # Train the n-gram model on the input text
    ngram.train()

    # Generate and print multiple sentences using the trained model
    for i in range(5):
        print(ngram.generate_sentence('<s>'))  # Generate a sentence starting with the start token '<s>'
        print()
