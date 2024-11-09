import os
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
import time
from transformers import set_seed
import argparse
import json

SEED=2024

MODEL_ARGS = {
    "model_name_or_path":"gpt2-large",

    # Lora parameters
    "use_peft_lora": True, ### SET TO TRUE TO ENABLE PEFT LoRA TRAINING
    "lora_alpha": 64,
    "lora_dropout": 0.1,
    "lora_r": 256,
    "lora_target_modules":None, # If None then all available modules are targeted

    # You will have to use_peft_lora to fine-tune the model with quantization.
    "use_nested_quant":True,
    "use_4bit_quantization":True, ### SET TO TRUE TO ENABLE 4-BIT QUANTIZATION
    "bnb_4bit_compute_dtype":"float16", # Half-precision floating point
    "bnb_4bit_quant_type":"nf4",
    "use_8bit_quantization":False,
}

DATA_ARGS = {
    "data_splits": "train",
    "add_special_tokens": False,
}

TRAINING_ARGS = {
    "output_dir": "models/sft_tldr",
    "seed": SEED,
    "num_train_epochs": 1,
    "max_seq_length": 650,

    # Batch size for training and evaluation
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 1,
    # Logging is done every 100 steps
    "logging_dir": "logs",
    "logging_strategy": "steps",
    "logging_steps": 200,

    # Saving strategy
    "save_strategy": "no",

    # Evaluation strategy (requires a validation dataset)
    "eval_strategy": "no",
    #"eval_strategy": "steps",
    #"eval_steps": 400,

    "learning_rate": 1e-5,

    "lr_scheduler_type": "cosine", # Learning rate scheduler with cosine annealing
    "report_to": "none", # No reporting to tensorboard or wandb
}

def get_dataset_from_jsonl(split):
    """
    Loads a dataset from a JSONL file and preprocesses the data into a dictionary with 'prompt' and 'label' keys.
    Args:
        split (str): The name of the dataset split to load (e.g. 'train', 'valid').
    Returns:
        dict: A dictionary containing the preprocessed dataset, with 'prompt' and 'label' keys.
    """

    if split=="train":
        dataset_name = "training_data_summarization.jsonl"
    elif split=="valid":
        dataset_name = "validation_data_summarization.jsonl"
    else:
        raise ValueError(f"Split type {split} not recognized as one of train or valid.")

    with open(dataset_name, "r") as f:
        dataset = [json.loads(line) for line in f]
    post_dict = {"prompt": [], "label": []}
    
    for d in dataset:
        post_dict["prompt"].append(f"SUBREDDIT: r/{d['subreddit']}\nTITLE: {d['title']}\nPOST: {d['post']}\nTL;DR: ")
        post_dict["label"].append(f"{d['summary']}")
    return post_dict

def formatting_prompts_func_summarize(example):
    """
    Formats the input prompts and labels into a summarization format.
    Args:
        example (dict): A dictionary containing the 'prompt' and 'label' keys, where 'prompt' is a list of input texts and 'label' is a list of corresponding labels.
    Returns:
        list: A list of formatted prompt strings, where each string contains the prompt text followed by the label text.
    """
    output_texts = []

    for i in range(len(example['prompt'])):
        text = f"Summarize:\n{example['prompt'][i]}{example['label'][i]}"
        output_texts.append(text)
    return output_texts


def create_dataset_raw(data_args):
    """
    Creates a dataset from JSONL files for the specified data splits (e.g. "train,valid").
    Args:
        data_args (argparse.Namespace): An object containing the data arguments, including the `data_splits` attribute which specifies the data splits to load.
    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the train and validation datasets.
    """
    train_data, valid_data = None, None

    for split in data_args.data_splits.split(","):
        data = get_dataset_from_jsonl(split=split)
        data = Dataset.from_dict(data)
        if split == "train":
            train_data = data
        elif split == "valid":
            valid_data = data
        else:
            f"Split type {split} not recognized as one of valid or train."
    
    print(
        f"Size of the train set: {len(train_data)}. Size of the valid set: {len(valid_data) if valid_data is not None else 0}"
    )

    return train_data, valid_data


def create_and_prepare_model(model_args):
    """
    Creates and prepares the model for training and evaluation.
    Args:
        model_args (argparse.Namespace): An object containing the model arguments, including options for 4-bit and 8-bit quantization, PEFT LoRA configuration, and special tokens.
    Returns:
        Tuple[AutoModelForCausalLM, LoraConfig, AutoTokenizer]: A tuple containing the created and prepared model, the PEFT LoRA configuration, and the tokenizer.
    """
    bnb_config = None
    device_map = None

    # assert if both 4bit and 8bit quantization are enabled
    assert not (model_args.use_4bit_quantization and model_args.use_8bit_quantization), "Both 4bit and 8bit quantization cannot be enabled at the same time"

    # Set up quantization configuration for 4-bit or 8-bit quantization
    if model_args.use_4bit_quantization:
        compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.use_4bit_quantization,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.use_nested_quant,
        )

    elif model_args.use_8bit_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=model_args.use_8bit_quantization,
        )

    # Set device mapping for distributed training with quantization
    if model_args.use_4bit_quantization or model_args.use_8bit_quantization:
        device_map = (
            int(os.environ.get("LOCAL_RANK", -1))
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else "auto"
        )

    # Load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    # Configure PEFT (Parameter-Efficient Fine-Tuning)
    peft_config = None
    if model_args.use_peft_lora:
        if model_args.lora_target_modules is not None:
            lora_target_modules = model_args.lora_target_modules.split(",")
        else:
            lora_target_modules = None

        peft_config = LoraConfig(
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            r=model_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules,
        )

    special_tokens = data_args.add_special_tokens
    # Load the tokenizer
    if special_tokens:
        eos_token = "</s>"
        bos_token = "<s>"
        pad_token = "<pad>"
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            trust_remote_code=True,
        )
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer


def train(model_args, data_args, training_args):

    set_seed(training_args.seed)
    # model
    model, peft_config, tokenizer = create_and_prepare_model(model_args)

    # data
    train_dataset, eval_dataset = create_dataset_raw(data_args)

    print("First two examples of the train dataset:")
    for i in range(2):
        print(f"Example {i}: {train_dataset[i]}")

    # template
    response_template = "TL;DR:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        formatting_func=formatting_prompts_func_summarize,
        peft_config=peft_config,
    )

    # Report the number of trainable parameters and memory allocation in GPU for the model
    trainer.accelerator.print(f"Memory allocated for the model: {torch.cuda.memory_allocated() / 1e9} GB")
    if model_args.use_peft_lora:
        trainer.model.print_trainable_parameters()
    # train
    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time
    # print the elapsed time in minutes
    print(f"Elapsed time: {elapsed_time / 60:.2f} minutes")
    # saving final model
    trainer.save_model()

if __name__ == '__main__':

    model_args = argparse.Namespace(**MODEL_ARGS)
    data_args = argparse.Namespace(**DATA_ARGS)
    sft_config = SFTConfig(**TRAINING_ARGS)
    train(model_args, data_args, sft_config)
