import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, default_data_collator, Trainer, TrainingArguments
from datasets import load_dataset

# Set a fixed seed for reproducibility
set_seed(42)

# Define the output directory for the trained model
output_dir = "models/sft_shakespeare"

# Load the pre-trained model and tokenizer
model_id = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map='auto', 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
)

def get_custom_dataset(tokenizer, split):
    """
    Loads and preprocesses the dataset for training.
    Args:
        tokenizer (AutoTokenizer): The tokenizer to use for encoding the data.
        split (str): The dataset split to load (e.g., 'train').
    Returns:
        Dataset: The tokenized dataset ready for training.
    """
    if split == "train":
        dataset = load_dataset(
            "json",
            data_files="training_data_completion.jsonl",
            split="train"
        )
    else:
        raise ValueError(f"Invalid split: {split}")

    # Determine the maximum sequence length in the dataset
    max_length = max(
        len(tokenizer.encode(
            tokenizer.bos_token + sample['input'] + sample['output'] + tokenizer.eos_token, 
            add_special_tokens=False
        )) for sample in dataset
    )

    def tokenize_add_label(sample):
        """
        Tokenizes the input and output, and prepares the labels for training.
        Args:
            sample (dict): A single data sample containing 'input' and 'output'.
        Returns:
            dict: A dictionary with tokenized input_ids, attention_mask, and labels.
        """
        prompt = tokenizer.encode(
            tokenizer.bos_token + sample['input'], 
            add_special_tokens=False
        )
        answer = tokenizer.encode(
            sample['output'] + tokenizer.eos_token, 
            add_special_tokens=False
        )
        padding = [tokenizer.eos_token_id] * (max_length - len(prompt + answer))
        
        sample = {
            "input_ids": prompt + answer + padding,
            "attention_mask": [1] * (len(prompt) + len(answer) + len(padding)),
            "labels": [-100] * len(prompt) + answer + [-100] * len(padding),
        }

        return sample

    # Apply the tokenization and label preparation to the dataset
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset

# Load and preprocess the training dataset
train_dataset = get_custom_dataset(tokenizer, 'train')

# Set the model to training mode
model.train()

# Set up the training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=1,
    gradient_checkpointing=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    weight_decay=0.1,
    bf16=True,  
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch_fused",
    adam_beta1=0.9,
    adam_beta2=0.95,
    report_to="none", # No reporting to tensorboard or wandb
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=default_data_collator,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)