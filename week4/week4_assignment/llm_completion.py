import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_device(use_gpu=False):
    """Set up the device for computation."""
    return torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_name, device):
    """Load the model and tokenizer, and move the model to the specified device."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer

def read_prefixes(file_path):
    """Read prefixes from a file."""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def generate_completion(model, tokenizer, prefix, device, max_length=100, num_generations=3):
    """Generate text completion for a given prefix."""  
    
    """
    You can learn more about the parameters used below at the following link: https://huggingface.co/blog/how-to-generate   
    """
    
    input_ids = tokenizer.encode(prefix, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=num_generations,
        do_sample=True,
        temperature=0.5,
        repetition_penalty=1.2,
    )
    # Remove the prefix from the generated completions
    prefix_length = input_ids.shape[-1]
    completions_without_prefix = [tokenizer.decode(output[i][prefix_length:], skip_special_tokens=True) for i in range(num_generations)]
    return completions_without_prefix

def write_completions_to_file(prefixes, model, tokenizer, device, output_file_path, num_generations):
    """Generate completions for each prefix and write them to a file."""
    
    enum = ["A", "B", "C"]
    
    with open(f'{output_file_path}.txt', 'w+') as output_file:
        for j, prefix in enumerate(prefixes):
            output_file.write(f"\n\n=== PREFIX {j+1} ===\n{prefix}")
            completions = generate_completion(model, tokenizer, prefix, device, num_generations=num_generations)
            for i, completion in enumerate(completions):
                output_file.write(f"\n\n====================================\n\n"
                                    f"=== COMPLETION {j+1}-{enum[i]} ===\n{completion}")     
            output_file.write("\n====================================\n====================================")

    print(f"Completions written to file.")

if __name__ == "__main__":
    MODEL_NAME = "microsoft/phi-1_5"
    # MODEL_NAME = "course-genai-w24/week4-phi-1.5-sft-shakespeare"

    num_generations = 3 # You will generate completions this number of times

    device = setup_device(use_gpu=True)
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device)
    prefixes = read_prefixes('input_shakespeare_prefixes.txt')
    
    if "sft" in MODEL_NAME:
        prefixes = [prefix + " <complete>" for prefix in prefixes]  
    
    write_completions_to_file(prefixes, model, tokenizer, device, f'output_shakespeare_completions_{MODEL_NAME.split("/")[-1]}', num_generations)
