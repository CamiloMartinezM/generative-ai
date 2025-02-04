import argparse
import json
import os
import random

import project_part1_prompts as prompts


# Retrieve problem data based on problem_id
def get_problem_data(problem_id):
    # Get the directory containing problem content
    def get_content_directory():
        return os.path.join("./project_part1_datasets/problems/")

    # Find the problem file for a given problem_id
    def find_problem_file(content_directory, problem_id):
        problem_file = next((f for f in os.listdir(content_directory) if f.startswith(f"{problem_id}_")), None)
        if not problem_file:
            raise FileNotFoundError(f"No file found for problem_id {problem_id}")
        return os.path.join(content_directory, problem_file)

    # Load test cases from a problem file
    def load_testcases(problem_file):
        try:
            with open(problem_file, "r") as file:
                data = json.load(file)
                return data["tests"]
        except FileNotFoundError:
            raise
        except json.JSONDecodeError:
            raise

    # Create a sample testcase string from testcases
    def create_sample_testcase(testcases, problem_id):
        sample_testcase = f"Input: \n{testcases[0]['input']}\nExpected Output: \n{testcases[0]['output']}\n"
        if problem_id == "2":
            sample_testcase += f"\nInput: \n{testcases[1]['input']}\nExpected Output: \n{testcases[1]['output']}\n"
            sample_testcase += f"\nInput: \n{testcases[3]['input']}\nExpected Output: \n{testcases[3]['output']}\n"
        return sample_testcase

    # Construct problem data string from file and testcases
    def construct_problem_data(problem_file, testcases, problem_id):
        with open(problem_file, "r") as file:
            data = json.load(file)
            title = data["title"]
            problem_description = data["description"]
            if data["additional_description"]:
                problem_description += "\n" + data["additional_description"]
            sample_testcase = create_sample_testcase(testcases, problem_id)
            problem_data = f"{title} - {problem_description}\n\n\nSample Testcase - \n{sample_testcase}\n"
            return problem_data

    # Fetch and return problem data for a given problem_id
    def fetch_problem_data(problem_id):
        content_directory = get_content_directory()
        problem_file = find_problem_file(content_directory, problem_id)
        testcases = load_testcases(problem_file)
        return construct_problem_data(problem_file, testcases, problem_id)

    return fetch_problem_data(problem_id)


# Shuffle lines in a JSONL file
def shuffle_jsonl_file(file_name, seed):
    with open(file_name, "r") as file:
        lines = file.readlines()

    random.seed(seed)
    random.shuffle(lines)

    with open(file_name, "w") as file:
        file.writelines(lines)


# Generate training dataset based on mode
def generate_train_dataset(output_file_name, mode):
    with open("./project_part2_dataset_training_raw.json", "r") as raw_file:
        raw_data = json.load(raw_file)

    with open(output_file_name, "w") as file:
        for problem_id, examples in raw_data.items():
            for example in examples:
                problem_data = get_problem_data(problem_id)
                if mode == "repair":
                    buggy_code = example["buggy_code"]
                    repaired_code = example["repaired_code"]

                    system_prompt_formatted = prompts.system_message_nus
                    user_prompt_formatted = prompts.user_message_nus_repair_basic.format(
                        problem_data=problem_data, buggy_program=buggy_code
                    )
                    input = f"""<|system|>\n{system_prompt_formatted}<|end|>\n<|user|>\n{user_prompt_formatted}<|end|>\n<|assistant|>"""
                    output = f"[FIXED]\n{repaired_code}\n[/FIXED]<|endoftext|>"

                    json_line = json.dumps({"input": input, "output": output})
                    file.write(json_line + "\n")
                elif mode == "hint":
                    buggy_code = example["buggy_code"]
                    hint = example["hint"]

                    system_prompt_formatted = prompts.system_message_nus
                    user_prompt_formatted = prompts.user_message_nus_hint_basic.format(
                        problem_data=problem_data, buggy_program=buggy_code
                    )
                    input = f"""<|system|>\n{system_prompt_formatted}<|end|>\n<|user|>\n{user_prompt_formatted}<|end|>\n<|assistant|>"""
                    output = f"[HINT]\n{hint}\n[/HINT]<|endoftext|>"

                    json_line = json.dumps({"input": input, "output": output})
                    file.write(json_line + "\n")

    shuffle_jsonl_file(output_file_name, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate repair or hint dataset")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["repair", "hint"],
        default="repair",
        help="Mode to generate dataset: repair or hint",
    )

    args = parser.parse_args()
    mode = args.mode

    generate_train_dataset(f"./project_part2_dataset_training_{mode}.jsonl", mode)
