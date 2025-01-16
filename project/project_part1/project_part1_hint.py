import json
import os

import openai
import torch

import project_part1_prompts as prompts
import project_part1_repair as repair
import project_part1_utils as utils

if torch.cuda.is_available():
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template


# Class for generating hints using language models
class Hint:
    # Initialize the Hint class with model details
    def __init__(self, model_name, is_huggingface, use_advanced_workflow=False):
        self.model_name = model_name
        self.system_prompt = prompts.system_message_nus
        # [x] TODO: Added the possibility of turning on/off the advanced workflow Part 1.c
        self.use_advanced_workflow = use_advanced_workflow

        if self.use_advanced_workflow:
            self.user_prompt = prompts.user_message_nus_hint_advanced
        else:
            self.user_prompt = prompts.user_message_nus_hint_basic

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

    # Extract hint from the generated text
    def extract_hint(self, text):
        start_tag = "[HINT]"
        end_tag = "[/HINT]"

        start_index = text.find(start_tag)
        if start_index == -1:
            return ""

        start_index += len(start_tag)
        end_index = text.find(end_tag, start_index)
        if end_index == -1:
            end_index = len(text)

        extracted_text = text[start_index:end_index].strip()
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
    def call_llm_openai(self, system_prompt_formatted, user_prompt_formatted):
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt_formatted},
                {"role": "user", "content": user_prompt_formatted},
            ],
        )

        return response.choices[0].message.content

    # Call the Hugging Face language model
    def call_llm_huggingface(self, system_prompt_formatted, user_prompt_formatted):
        prompt_string = (
            f"""<|system|>\n{system_prompt_formatted}<|end|>\n<|user|>\n{user_prompt_formatted}<|end|>\n<|assistant|>"""
        )
        inputs = self.tokenizer(prompt_string, return_tensors="pt", padding=True, truncation=True).to("cuda")

        outputs = self.model.generate(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=2048, use_cache=True
        )

        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)

        return response_text

    # Call the appropriate language model based on configuration
    def call_llm(self, problem_data, buggy_program, repaired_program=None, testcases_results=None):
        system_prompt_formatted = self.system_prompt

        # [x] TODO: Added the possiblity of using the repaired program as part of the prompt for the model
        if self.use_advanced_workflow:
            assert repaired_program is not None and repaired_program.strip() != "" and testcases_results is not None, (
                f"repaired_program is required for advanced workflow, but got {repaired_program}"
            )

            # Re-format the testcases_results which is a list of tuples to a prompt-engineered string
            reformatted_testcases_results = self.__reformat_testcases_results(testcases_results)
            user_prompt_formatted = self.user_prompt.format(
                problem_data=problem_data.strip(),
                buggy_program=buggy_program.strip(),
                repaired_program=repaired_program,
                testcases_results=reformatted_testcases_results,
            )
        else:
            user_prompt_formatted = self.user_prompt.format(problem_data=problem_data, buggy_program=buggy_program)

        transcript = []

        if self.is_huggingface:
            generated_response = self.call_llm_huggingface(system_prompt_formatted, user_prompt_formatted)
        else:
            generated_response = self.call_llm_openai(system_prompt_formatted, user_prompt_formatted)

        transcript.append(
            {"input_prompt": system_prompt_formatted + user_prompt_formatted, "output": generated_response}
        )

        self.save_transcript_to_json(transcript)

        return generated_response

    # Generate a hint for the given problem and program
    def generate_hint(self, problem_data, buggy_program, testcases, repair_agent=None):
        # Part 1.c: Advanced Workflow [6 Points]
        # [x] TODO (I.7): Modify required scripts to improve the quality of the hint generation process of the model
        additional_kwargs = {}
        if self.use_advanced_workflow:
            assert repair_agent is not None, "repair_agent is required for advanced workflow"
            extracted_repair, testcases_results = repair_agent.generate_repair(
                problem_data, buggy_program, testcases, n=3, do_sample=True, temperature=0.7
            )
            additional_kwargs["repaired_program"] = extracted_repair.strip()
            additional_kwargs["testcases_results"] = testcases_results

        generated_response = self.call_llm(problem_data, buggy_program, **additional_kwargs)
        hint = self.extract_hint(generated_response)
        return hint

    def __reformat_testcases_results(self, testcases_results: list[tuple[bool, str]]) -> str:
        """Reformat the testcases_results to a prompt-engineered string.

        Parameters
        ----------
        testcases_results : list[tuple[bool, str]]
            Example: `[(False, '[2, 3]'), (False, '[5, 3, 2]'), (True, '[]'), (False, 'SyntaxError: ...')]`

        Returns
        -------
        str
            A list beginning with '*' for each testcase result, with the testcase number, pass/fail status, and error
            message (if any). For example,
            ```
            * Testcase 4 failed with error: SyntaxError: ...
            * 1 out of 4 testcases passed.
            ```
        """
        reformatted = ""
        testcases_passed = 0
        for i, (passed, result_msg) in enumerate(testcases_results):
            if passed:
                testcases_passed += 1
            elif "error" in result_msg.lower() or "exception" in result_msg.lower():
                reformatted += f"* Testcase {i + 1} failed error: {result_msg}\n"
        reformatted += f"* {testcases_passed} out of {len(testcases_results)} testcases passed."
        return reformatted.strip()
