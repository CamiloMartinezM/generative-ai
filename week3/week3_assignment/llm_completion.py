import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_device(use_gpu=False):
    """Set up the device for computation."""
    return torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(model_name, device):
    """Load the model and tokenizer, and move the model to the specified device."""
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=os.path.join(os.getcwd(), "cache"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.path.join(os.getcwd(), "cache"))
    model.to(device)
    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer


def read_prefixes(file_path):
    """Read prefixes from a file."""
    with open(file_path, "r") as file:
        return [line.strip() for line in file.readlines()]


def generate_completion(model, tokenizer, prefix, device, max_length=100):
    """Generate text completion for a given prefix."""
    input_ids = tokenizer.encode(prefix, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def write_completions_to_file(prefixes, model, tokenizer, device, output_file_path):
    """Generate completions for each prefix and write them to a file."""
    with open(output_file_path, "w+") as output_file:
        for prefix in prefixes:
            completion = generate_completion(model, tokenizer, prefix, device)
            output_file.write(
                f"===PREFIX===\n{prefix}\n\n===COMPLETION===\n{completion}\n\n====================================\n\n"
            )

    print("Completions written to file.")


if __name__ == "__main__":
    print(f"Using GPU: {torch.cuda.is_available()}")
    MODEL_NAME = "openai-community/gpt2-xl"
    # MODEL_NAME = "microsoft/phi-1_5"

    device = setup_device(use_gpu=True)
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device)
    prefixes = read_prefixes("input_shakespeare_prefixes.txt")
    write_completions_to_file(
        prefixes, model, tokenizer, device, f'output_shakespeare_completions_{MODEL_NAME.split("/")[-1]}.txt'
    )
