import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import evaluate

def setup_device(use_gpu=False):
    """Set up the device for computation."""
    return torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_name, device):
    """Load the model and tokenizer, and move the model to the specified device."""
    if "lora-sft" in model_name:
        peft_model_id = model_name
        peft_config = PeftConfig.from_pretrained(peft_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path,
                                                          quantization_config=None,
                                                          device_map=None,
                                                          trust_remote_code=True,
                                                          attn_implementation="eager")
        tokenizer = AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path,
            padding_side="left")

        tokenizer.pad_token = tokenizer.eos_token
        model = PeftModel.from_pretrained(base_model, peft_model_id).to(device)

    elif "CarperAI" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model.to(device)

    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model.to(device)
    return model, tokenizer

def read_posts(post_path):
    """Read reddit posts and their human generated summaries from a file."""
    with open(post_path, 'r') as f:
        reddit_posts = f.read().split('\n\n')
    return reddit_posts

def generate_summary(model, tokenizer, posts, n_generations, device, max_length=1024):
    """Generate a summary for a given post."""
    post_encodings = tokenizer(posts,
                               truncation=True,
                               padding=True,
                               max_length=max_length,
                               return_tensors='pt').to(device)

    # You can learn more about the generation parameters at the following link: https://huggingface.co/blog/how-to-generate
    outputs = model.generate(**post_encodings,
                            max_new_tokens=180,
                            num_return_sequences=n_generations,
                            pad_token_id=tokenizer.eos_token_id,
                            do_sample=True,
                            temperature=0.2)

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    summaries = []
    for i, output in enumerate(decoded):
        # Split the output into the prompt and the generated response in the TL;DR: format
        output_split = output.split("TL;DR:")
        summaries.append(output_split[1])
    return summaries

def write_summaries_to_file(posts, model, tokenizer, n_generations, device, output_file_path):
    """Generate summaries for each prefix and write them to a file."""

    enum = ["A", "B", "C"]
    with open(output_file_path, 'w+') as output_file:
        all_summaries = []
        for i, post in enumerate(posts):
            output_file.write(f"\n\n=== POST {i+1} ===\n{post}")
            summaries = generate_summary(model, tokenizer, post, n_generations, device)
            for n, summary in enumerate(summaries):
                output_file.write(f"\n\n====================================\n\n"
                                  f"=== SUMMARY {i+1}-{enum[n]} ===\n{summary}")
            output_file.write("\n\n====================================\n====================================")
            all_summaries.append(summaries)
    print("Summaries written to file.")

    return all_summaries

def compute_rouge_scores_and_write_to_file(predictions, references, n_generations, output_file_path):
    """Compute ROUGE scores for a list of hypotheses and references."""
    rouge = evaluate.load('rouge')
    per_generation_rouge = []
    for i in range(n_generations):
        per_generation_predictions = []
        for j in range(len(predictions)):
            per_generation_predictions.append(predictions[j][i])
        scores = rouge.compute(predictions=per_generation_predictions, references=references, rouge_types=['rouge1'])
        per_generation_rouge.append(scores["rouge1"])
    mean_rouge_score = sum(per_generation_rouge) / len(per_generation_rouge)
    with open(output_file_path, 'w+') as output_file:
        output_file.write("ROUGE is a set of metrics that are used to evaluate summarization in natural language processing. "
                          "They are used to compare an automatically produced summary against a human-produced summary. "
                          "The higher score the better and if the score is 1 then it matches exactly the human reference summary."
                          "\n\n====================================\n\n")
        output_file.write(f"ROUGE-1 mean score: {mean_rouge_score}")

    print("ROUGE-1 score is written to file.")

if __name__ == "__main__":

    ############## Put a message to the user to let them know that the script will not run because of resources so we already have the output files for them
    print("\n====================================\n====================================")
    print("This script will not run in Colab because of the resources required to run it.\n"
          "The output summaries for SFT and PPO models are already provided to you.\n"
          "If you have the resources uncomment the exit() and run it.")
    print("====================================\n====================================")
    exit()
    ##############
    n_generations = 3 # This is the number of summaries you will generate for each post and each model

    MODEL_NAME = "CarperAI/openai_summarize_tldr_sft"
    #MODEL_NAME = "CarperAI/openai_summarize_tldr_ppo"

    reddit_posts = read_posts('input_reddit_posts.txt')
    device = setup_device(use_gpu=True)
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, device)

    if "sft" in MODEL_NAME:
        output_file_path = 'output_reddit_summaries_week5-sft-tldr.txt'
    else:
        output_file_path = 'output_reddit_summaries_week5-ppo-tldr.txt'

    all_summaries = write_summaries_to_file(reddit_posts, model, tokenizer, n_generations, device,
                                      output_file_path)