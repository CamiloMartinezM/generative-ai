import openai
import project_part1_prompts as prompts
import project_part1_utils as utils

import os
import json

import torch

if torch.cuda.is_available():
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template


# Class for handling program repair using language models
class Repair:
    # Initialize the Repair class with model details
    def __init__(self, model_name, is_huggingface):
        self.model_name = model_name
        self.system_prompt = prompts.system_message_nus
        self.user_prompt = prompts.user_message_nus_repair_basic
        self.is_huggingface = is_huggingface

        if self.is_huggingface:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template="chatml",
            )
            FastLanguageModel.for_inference(self.model)
        else:
            with open("./8_Martinez_openai.txt") as f:
                content = f.read().strip()
            openai_key = content.split("\n")[2]
            openai.api_key = openai_key

    # Extract fixed code from the generated text
    def extract_fixed_code(self, text):
        start_tag = "[FIXED]"
        end_tag = "[/FIXED]"

        start_index = text.find(start_tag)
        if start_index == -1:
            return ""

        start_index += len(start_tag)
        end_index = text.find(end_tag, start_index)
        if end_index == -1:
            end_index = len(text)

        extracted_text = text[start_index:end_index].strip()

        if extracted_text.startswith("```"):
            extracted_text = extracted_text[3:]
        if extracted_text.startswith("python"):
            extracted_text = extracted_text[6:]
        if extracted_text.endswith("```"):
            extracted_text = extracted_text[:-3]

        return extracted_text

    # Save the transcript to a JSON file at "project_part1_transcripts/transcript.json". This file contains all prompts and LLM responses which can be used for debugging.
    def save_transcript_to_json(self, transcript):
        os.makedirs("project_part1_transcripts", exist_ok=True)
        file_path = os.path.join("project_part1_transcripts", "transcript.json")

        # Read existing data
        if os.path.exists(file_path):
            with open(file_path, "r") as json_file:
                try:
                    existing_data = json.load(json_file)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        # Append new transcript data
        existing_data.extend(transcript)

        # Write back to the file
        with open(file_path, "w") as json_file:
            json.dump(existing_data, json_file, indent=4)

    # Call the OpenAI language model
    def call_llm_openai(self, system_prompt_formatted, user_prompt_formatted, temperature=0):
        # [x] TODO: Added ability to set temperature
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt_formatted},
                {"role": "user", "content": user_prompt_formatted},
            ],
            temperature=temperature if temperature is not None else 0,
        )

        return response.choices[0].message.content

    # Call the Hugging Face language model
    def call_llm_huggingface(self, system_prompt_formatted, user_prompt_formatted, do_sample=None, temperature=None):
        prompt_string = (
            f"""<|system|>\n{system_prompt_formatted}<|end|>\n<|user|>\n{user_prompt_formatted}<|end|>\n<|assistant|>"""
        )
        inputs = self.tokenizer(prompt_string, return_tensors="pt", padding=True, truncation=True).to("cuda")

        # [x] TODO: Added ability to set do_sample and temperature
        additional_kwargs = {}
        # Only set if do_sample is True, since I don't know model.generate default argument value
        if do_sample is not None and do_sample:
            additional_kwargs["do_sample"] = do_sample
        if temperature is not None:
            additional_kwargs["temperature"] = temperature

        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=2048,
            use_cache=True,
            # You can use the parameters below to set the temperature for sampling. To learn more about these parameters, you can refer to https://huggingface.co/blog/how-to-generate.
            # do_sample=True,
            # temperature=0.7
            **additional_kwargs,
        )

        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)

        return response_text

    # Call the appropriate language model based on configuration
    def call_llm(self, problem_data, buggy_program, do_sample=None, temperature=None):
        system_prompt_formatted = self.system_prompt
        user_prompt_formatted = self.user_prompt.format(problem_data=problem_data, buggy_program=buggy_program)

        transcript = []

        if self.is_huggingface:
            generated_response = self.call_llm_huggingface(
                system_prompt_formatted, user_prompt_formatted, do_sample, temperature
            )
        else:
            generated_response = self.call_llm_openai(system_prompt_formatted, user_prompt_formatted, temperature)

        transcript.append(
            {"input_prompt": system_prompt_formatted + user_prompt_formatted, "output": generated_response}
        )

        self.save_transcript_to_json(transcript)

        return generated_response

    # Generate a repair for the given problem and program
    def generate_repair(
        self, problem_data, buggy_program, testcases, n=1, do_sample=None, temperature=None
    ) -> tuple[str, list[tuple[bool, str]]]:
        # Part 1.b: Improved Program Repairs [2 Point]
        # [x] TODO (I.5): Modify the scripts to query the model to generate n=3 repair candidates with a temperature
        # of 0.7 and select the best candidate.
        compiler = utils.Compiler()
        distance = utils.Distance()

        # Save a list of [(fixed_code, num_correct_testcases, edit_distance), ...]
        fixed_codes_stats = []
        for _ in range(n):
            generated_response = self.call_llm(
                problem_data, buggy_program, do_sample=do_sample, temperature=temperature
            )
            fixed_code = self.extract_fixed_code(generated_response)

            # Results look like [(testcase_correct, output), ...]
            _, results = compiler.run_program_with_testcases(fixed_code, testcases)

            # Compute stats
            num_correct_testcases = sum([1 for result in results if result[0]])
            current_distance = distance.get_edit_distance(fixed_code, buggy_program)

            fixed_codes_stats.append((fixed_code, num_correct_testcases, current_distance, results))

        # Sort by number of correct testcases and edit distance at the same time
        # If fixed_codes_stats =      [("", 5, 14), ("", 5, 2), ("best", 10, 2), ("3 best", 10, 18), ("2 best", 10, 3)]
        # then sorted should be       [('best', 10, 2), ('2 best', 10, 3), ('3 best', 10, 18), ('', 5, 2), ('', 5, 14)]

        # Sort by number of correct testcases (descending) and edit distance (ascending)
        fixed_codes_stats.sort(key=lambda x: (-x[1], x[2]))

        # Make an assertion with the first and second item just to be safe
        assert fixed_codes_stats[0][1] >= fixed_codes_stats[1][1], (
            f"Got 0th item with: {fixed_codes_stats[0][1]} and 1st item with: {fixed_codes_stats[1][1]}"
        )  # Assert more correct testcases

        # If the number of correct testcases are the same, then assert that the edit distance is lower for the 1st item
        if fixed_codes_stats[0][1] == fixed_codes_stats[1][1]:
            assert fixed_codes_stats[0][2] <= fixed_codes_stats[1][2], (
                f"Got 0th item with: {fixed_codes_stats[0][2]} and 1st item with: {fixed_codes_stats[1][2]}"
            )

        # [x] TODO: For Part 1.c, changed the return statement to return the fixed code and the results of the test 
        # cases, which will be used for the prompt to generate a hint
        return fixed_codes_stats[0][0], fixed_codes_stats[0][3]
