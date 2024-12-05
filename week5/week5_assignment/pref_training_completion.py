from datasets import load_dataset
from trl import ORPOConfig, ORPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the output directory for the trained model
output_dir="./models/week5-phi-1.5-pref-shakespeare"

# Load the pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained("course-genai-w24/week4-phi-1.5-sft-shakespeare")
tokenizer = AutoTokenizer.from_pretrained("course-genai-w24/week4-phi-1.5-sft-shakespeare")
tokenizer.pad_token = tokenizer.eos_token

train_dataset = load_dataset("json", data_files="pref_data_completion.jsonl", split="train")

def pad_batch(batch):
    """
    Pads the 'chosen' sequences in the batch to match the length of the 'rejected' sequences.
    This is done by appending the end-of-sequence token to the 'chosen' sequences until they are the same length as the 'rejected' sequences.
    
    Args:
        batch (dict): A dictionary containing 'chosen' and 'rejected' sequences.
    
    Returns:
        dict: The input batch dictionary with padded 'chosen' sequences.
    """
    padded_chosen = []
    for chosen, rejected in zip(batch['chosen'], batch['rejected']):
        length_diff = len(rejected) - len(chosen)
        padded_chosen.append(chosen + tokenizer.eos_token * length_diff)
    batch['chosen'] = padded_chosen
    batch['rejected'] = batch['rejected']
    return batch

# Load and preprocess the training dataset
padded_train_dataset = train_dataset.map(pad_batch, batched=True)

# Set up the training arguments
training_args = ORPOConfig(
    output_dir=output_dir, 
    logging_steps=10,
    per_device_train_batch_size=1,
    learning_rate=1e-6,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    save_strategy="epoch",
)

# Initialize the Trainer
trainer = ORPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=padded_train_dataset)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)