import torch
from unsloth import FastLanguageModel
from datasets import load_dataset

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

def get_custom_dataset(tokenizer, split, data_file):
    """
    Loads and preprocesses the dataset for training.
    Args:
        tokenizer (AutoTokenizer): The tokenizer to use for encoding the data.
        split (str): The dataset split to load (e.g., 'train').
        data_file (str): The file name of the dataset.
    Returns:
        Dataset: The tokenized dataset ready for training.
    """
    if split == "train":
        dataset = load_dataset(
            "json",
            data_files=data_file,
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

if __name__ == "__main__":
    lora_r = 16
    lora_alpha = 32

    data_file = "project_part2_dataset_training_repair.jsonl"  # Change this to the path of the training dataset (i.e. the jsonl file you assembled)
    output_dir = f"project_part2_models/Phi-3-SFT-Repair_r{lora_r}_alpha{lora_alpha}" # Change this to the desired output directory

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Phi-3-mini-4k-instruct",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = lora_alpha,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    dataset = get_custom_dataset(tokenizer, 'train', data_file)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 1,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)

    print(f"GPU = {gpu_stats.name}. Max memory = {gpu_stats.total_memory / 1024 / 1024 / 1024} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    current_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    lora_memory = round(current_memory - start_gpu_memory, 3)

    print(f"Training runtime: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
    print(f"Training memory: {lora_memory} GB")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

