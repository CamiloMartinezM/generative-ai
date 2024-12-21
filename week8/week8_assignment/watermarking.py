from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, T5ForConditionalGeneration
from datasets import load_dataset
import torch
import numpy as np
import hashlib
import math
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("vicgalle/gpt2-alpaca")
    model = AutoModelForCausalLM.from_pretrained("vicgalle/gpt2-alpaca").to(device)
    model.max_length = 512

    def hard_red_list_generation(input, tokenizer, model, max_new_tokens=20):
        for tok in range(max_new_tokens):
            pass
            # Generate next probability token
            tokens = tokenizer(input, return_tensors="pt").to(device)
            probs = model(**tokens).logits[0, -1]

            # compute hash of previous token, use sha256 (https://docs.python.org/3/library/hashlib.html) and use this as a seed
            prev_token = tokens["input_ids"][0, -1]
            prev_token = tokenizer.decode(prev_token)
            seed = hashlib.sha256(prev_token.encode("utf-8")).hexdigest()
            seed = int(seed, 16)

            # reduce size of seed, needed for torch
            seed = seed & ((1<<63)-1)

            # randomly partion vocabulary into a green list G and a red list R of equal size
            torch.random.manual_seed(seed)
            vocab = torch.randperm(tokenizer.vocab_size).squeeze()
            green_list, red_list = vocab[:len(vocab)//2].to(device), vocab[len(vocab)//2+1:].to(device)

            # reset seed for sampling
            torch.random.manual_seed(np.random.randint(2**32))
            # sample next token from green list and add it to the input
            probs = torch.softmax(probs[green_list], dim=-1)
            next_token = torch.multinomial(probs, 1)
            next_token = tokenizer.decode(green_list[next_token])
            input += next_token

        return input
    
    def compute_perplexity(answer, tokenizer, model):
        with torch.no_grad():
            # compute log-probabilities for each token
            answer = answer.split("Response: ")[1]
            tokens = tokenizer(answer, return_tensors="pt").to(device)
            logits = model(**tokens).logits
            logprobs = torch.log_softmax(logits, dim = -1)
            logprobs = logprobs[0, torch.arange(0,len(tokens["input_ids"])), tokens["input_ids"]]

            # compute perplexity
            perplexity = -logprobs.mean()

            return perplexity
        
    prompts = ["### Instruction: Give three tips for staying healthy. ### Response: \n",
          "### Instruction: Generate a list of ten items a person might need for a camping trip ### Response: \n",
          "### Instruction: Find the capital of Spain. ### Response: \n"
    ]

    exercise_1_res = "====================================\n====================================\n\n"
    for i, p in enumerate(prompts):
        exercise_1_res += f"=== {i+1}. Instruction ===\n" + p + "\n====================================\n\n"
        
        response_non_watermarked = tokenizer.decode(model.generate(**tokenizer(p, return_tensors="pt").to(device), max_new_tokens=64, do_sample=True)[0])
        exercise_1_res += f"=== {i+1}. Non-Watermarked Response ===\n"
        exercise_1_res += response_non_watermarked
        exercise_1_res += f"\n\n=== Perplexity ===\n{compute_perplexity(response_non_watermarked, tokenizer, model)}"
        exercise_1_res += "\n\n====================================\n\n"

        response_watermarked = hard_red_list_generation(p, tokenizer, model, max_new_tokens=64)
        exercise_1_res += f"=== {i+1}. Watermarked Response ===\n"
        exercise_1_res += response_watermarked
        exercise_1_res += f"\n\n=== Perplexity ===\n{compute_perplexity(response_watermarked, tokenizer, model)}"
        exercise_1_res += "\n\n====================================\n====================================\n\n"

    with open("I1.txt", "w") as f:
        f.write(exercise_1_res)

    def detect_watermark(text, z=4):
        num_green_words = 0
        # Count the number of red words in the text
        prompt, answer = text.split("### Response:", 1)
        answer_tokens = tokenizer.encode(":" + answer)
        for i in range(1, len(answer_tokens)):
            last_word = tokenizer.decode(answer_tokens[i-1])
            seed = hashlib.sha256(last_word.encode("utf-8")).hexdigest()
            seed = int(seed, 16)
            seed = seed & ((1<<63)-1)
            torch.random.manual_seed(seed)
            vocab = torch.randperm(tokenizer.vocab_size).squeeze()
            green_list, red_list = vocab[:len(vocab)//2], vocab[len(vocab)//2+1:]
            num_green_words += answer_tokens[i] in green_list

        # Compute z-score and detect if the input includes the watermark
        T = len(answer_tokens) - 1
        z_score = 2*((num_green_words)-T/2)/(math.sqrt(T))
        return z_score > z
    
    # Generate test data
    # load test dataset
    data = load_dataset("vicgalle/alpaca-gpt4", split="train").select(range(50))
    # generate watermarked and non-watermarked examples
    watermarked, non_watermarked = [], []
    for prompt in tqdm(data["instruction"]):
        prompt = "### Instruction: " + prompt + "### Response: \n"
        watermarked.append(hard_red_list_generation(prompt, tokenizer, model, 128))
        non_watermarked.append(tokenizer.decode(model.generate(**tokenizer(prompt, return_tensors="pt").to(device), max_new_tokens=128, do_sample=True)[0]))


    # Test watermark detection
    def test_watermarks(watermarked, non_watermarked, z_values = [0.5, 4, 20]):
        res = ""
        for z in z_values:
            res += f"{z=}\n"
            fp, fn, tp, tn = 0, 0, 0, 0
            for watermarked_response in watermarked:
                if detect_watermark(watermarked_response, z):
                    tp += 1
                else:
                    fn += 1
            for non_watermarked_response in non_watermarked:
                if detect_watermark(non_watermarked_response, z):
                    fp += 1
                else:
                    tn += 1
            res += f"{tp=}\t{fn=}\n{fp=}\t{tn=}\n\n"
        return res

    res = test_watermarks(watermarked, non_watermarked)
    with open("I2.txt", "w") as f:
        f.write(res)

    # Test Perplexity
    ppl_watermarked, ppl_non_watermarked = 0, 0
    for watermarked_response, non_watermarked_response in tqdm(zip(watermarked, non_watermarked)):
        ppl_watermarked += compute_perplexity(watermarked_response, tokenizer, model)
        ppl_non_watermarked += compute_perplexity(non_watermarked_response, tokenizer, model)

    res = f"Perplexity (Non-Watermarked): {ppl_non_watermarked/50}\nPerplexity (Watermarked): {ppl_watermarked/50}"
    with open("I3.txt", "w") as f:
        f.write(res)

    from transformers import T5Tokenizer, T5ForConditionalGeneration
    # See https://huggingface.co/docs/transformers/en/model_doc/t5
    # The "training" section is especially important
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

    def t5_span_attack(input, t5_tokenizer, t5_model, eps=0.1, num_iters=50):
        prompt, answer = input.split("### Response: ")
        num_replaced = 0
        for t in range(num_iters):
            tokens = t5_tokenizer.encode(answer, return_tensors="pt").to(device)
            # Randomly replace one word with the mask
            replaced_word = np.random.randint(len(tokens[0])-1)
            prev_word = tokens[0, replaced_word]
            tokens[0, replaced_word] = t5_tokenizer.convert_tokens_to_ids("<extra_id_0>")

            # generate alternatives using T5, instead of beam search, you can use simpler sampling techniques (e.g. greedy sampling)
            alternatives = t5_model(tokens, decoder_input_ids=tokens)
            new_word = tokens[0, replaced_word]
            tokens[0, replaced_word] = torch.multinomial(torch.softmax(alternatives.logits[0, replaced_word], dim=0), 1).item()
            answer = t5_tokenizer.decode(tokens[0]).replace("</s>", "")

            # Test if the replaced token is different than the previous token, stop the attack if more than \epsilon*T tokens have been replaced
            num_replaced += new_word != prev_word
            if num_replaced > eps*len(tokens[0]):
                break
        return prompt + "### Response: " + answer
    
    # Generate data
    watermarked_attacked, non_watermarked_attacked = [], []
    for response in tqdm(watermarked):
        watermarked_attacked.append(t5_span_attack(response, t5_tokenizer, t5_model))
    for response in tqdm(non_watermarked):
        non_watermarked_attacked.append(t5_span_attack(response, t5_tokenizer, t5_model))

    res = test_watermarks(watermarked_attacked, non_watermarked_attacked, z_values=[4])
    with open("I4.txt", "w") as f:
        f.write(res)

    # Test Perplexity
    ppl_watermarked, ppl_non_watermarked = 0, 0
    for watermarked_response, non_watermarked_response in tqdm(zip(watermarked_attacked, non_watermarked_attacked)):
        ppl_watermarked += compute_perplexity(watermarked_response, tokenizer, model)
        ppl_non_watermarked += compute_perplexity(non_watermarked_response, tokenizer, model)

    res = f"Perplexity (Non-Watermarked): {ppl_non_watermarked/50}\nPerplexity (Watermarked): {ppl_watermarked/50}"
    with open("I5.txt", "w") as f:
        f.write(res)

if __name__ == "__main__":
    main()