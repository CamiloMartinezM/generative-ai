import openai
import os
import base64

def encode_image(image_path):
    """
    Encode an image file into a Base64 string.

    Parameters:
    - image_path (str): Path to the image file.

    Returns:
    - str: Base64-encoded string representation of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_prompt(file_path):
    """
    Load a prompt template from a file.

    Parameters:
    - file_path (str): Path to the file containing the system prompt.

    Returns:
    - str: The content of the prompt file.
    """
    with open(file_path, "r") as file:
        return file.read()

def task_grid_to_elements_ascii(grid_text, system_prompt_path, num_responses=3):
    """
    Process an ASCII grid to identify its elements using OpenAI API.

    Parameters:
    - grid_text (str): The textual representation of the grid.
    - system_prompt_path (str): Path to the system prompt file.
    - num_responses (int): Number of responses to generate. Defaults to 3.

    Returns:
    - list: A list of model-generated responses describing the grid elements.
    """
    system_prompt = load_prompt(system_prompt_path)
    user_prompt = f"Identify the elements in the following grid and provide them in this exact format:\n\n" \
                  f"avatar: row:col:dir (where dir is one of east, north, west, or south)\n" \
                  f"goal: row:col\n" \
                  f"walls: [list of wall positions as row:col]\n\n" \
                  f"Grid:\n{grid_text}"
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        n=num_responses
    )
    return [choice.message.content for choice in response.choices]

def task_grid_to_elements_vis(image_base64, system_prompt_path, num_responses=3):
    """
    Process a visual grid image to identify its elements using OpenAI API.

    Parameters:
    - image_base64 (str): Base64-encoded string representation of the image.
    - system_prompt_path (str): Path to the system prompt file.
    - num_responses (int): Number of responses to generate. Defaults to 3.

    Returns:
    - list: A list of model-generated responses describing the grid elements.
    """
    system_prompt = load_prompt(system_prompt_path)
    user_prompt = f"Analyze the provided image of the grid and identify the elements in the following format:\n\n" \
                  f"avatar: row:col:dir (where dir is one of east, north, west, or south based on the blue dart's direction)\n" \
                  f"goal: row:col (red star)\n" \
                  f"walls: [list of wall positions as row:col (gray cells)]\n\n"
    image_data = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [user_prompt, image_data]}
        ],
        n=num_responses
    )
    return [choice.message.content for choice in response.choices]

def main(input_type="ascii", num_responses=3):
    """
    Main function to process a grid (ASCII or visual) and identify its elements.

    Parameters:
    - input_type (str): Type of input to process ('ascii' or 'vis'). Defaults to 'ascii'.
    - num_responses (int): Number of responses to generate. Defaults to 3.

    Depending on the input type:
    - 'ascii': Reads an ASCII grid from a file and processes it.
    - 'vis': Reads a grid image, encodes it, and processes it.

    Saves the generated responses to the 'output_grid_to_elements' directory.
    """
    os.makedirs("output_grid_to_elements", exist_ok=True)

    if input_type == "ascii":
        # ASCII Task
        with open("input_visualprog_task1_grid.txt") as f:
            grid_text = f.read()
        responses = task_grid_to_elements_ascii(grid_text, "system_prompt_ascii.txt", num_responses)
        for i, response in enumerate(responses):
            with open(f"output_grid_to_elements/I3_{chr(97 + i)}.txt", "w") as file:
                file.write(response)
                print(f"Response generated and saved to I3_{chr(97 + i)}.txt")
    elif input_type == "vis":
        # Visual Task
        image_base64 = encode_image("input_visualprog_task1_grid.png")
        responses = task_grid_to_elements_vis(image_base64, "system_prompt_vis.txt", num_responses)
        for i, response in enumerate(responses):
            with open(f"output_grid_to_elements/I4_{chr(97 + i)}.txt", "w") as file:
                file.write(response)
                print(f"Response generated and saved to I4_{chr(97 + i)}.txt")
    else:
        print("Invalid input_type. Choose either 'ascii' or 'vis'.")

if __name__ == "__main__":
    # Load the OpenAI API key from a file.
    with open("<<OPENAI_API_KEY_FILE>>") as f:
        content = f.read().strip()
    # Extract the API key from the third line of the file.
    openai_key = content.split("\n")[2]
    openai.api_key = openai_key

    # Choose 'ascii' for text-based grid or 'vis' for image-based grid.
    input_type = "ascii"
    # input_type = "vis"
    main(input_type=input_type)
