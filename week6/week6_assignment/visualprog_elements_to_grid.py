import openai
import os
import base64

def load_prompt(file_path):
    """
    Load a system prompt template from a file.

    Parameters:
    - file_path (str): Path to the file containing the system prompt.

    Returns:
    - str: Content of the prompt file.
    """
    with open(file_path, "r") as file:
        return file.read()

def save_image_from_b64(b64_data, output_path):
    """
    Save an image from Base64-encoded data to a file.

    Parameters:
    - b64_data (str): Base64-encoded image data.
    - output_path (str): Path to save the decoded image file.

    Returns:
    - None
    """
    img_data = base64.b64decode(b64_data)
    with open(output_path, "wb") as img_file:
        img_file.write(img_data)

def task_elements_to_grid_ascii(elements_text, system_prompt_path, num_responses=3):
    """
    Generate an ASCII grid based on provided elements using OpenAI API.

    Parameters:
    - elements_text (str): Text describing the grid elements.
    - system_prompt_path (str): Path to the system prompt file.
    - num_responses (int): Number of responses to generate. Defaults to 3.

    Returns:
    - list: A list of ASCII grid representations as strings.
    """
    system_prompt = load_prompt(system_prompt_path)
    user_prompt = f"Create an ASCII grid based on the following elements:\n\n" \
                  f"{elements_text}\n\n" \
                  f"Use '>', '<', '^', 'v' for avatar directions, '+' for the goal, and '#' for walls."
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        n=num_responses
    )
    return [choice.message.content for choice in response.choices]

def task_elements_to_grid_vis(elements_text, system_prompt_path, num_responses=3):
    """
    Generate visual grids based on provided elements using OpenAI API.

    Parameters:
    - elements_text (str): Text describing the grid elements.
    - system_prompt_path (str): Path to the system prompt file.
    - num_responses (int): Number of images to generate. Defaults to 3.

    Returns:
    - list: A list of responses containing Base64-encoded image data.
    """
    system_prompt = load_prompt(system_prompt_path)
    user_prompt = f"Generate a 4x4 visual grid based on the elements provided:\n\n{elements_text}\n\n" \
                  f"Use a blue dart for the avatar in the correct direction, a red star for the goal, and gray cells for walls."
    responses = []
    for _ in range(num_responses):
        response = openai.images.generate(
            model="dall-e-3",
            prompt=system_prompt + "\n\n" + user_prompt,
            n=1,
            size="1024x1024",
            quality="standard",
            response_format="b64_json"
        )
        responses.append(response.data[0])
    return responses

def main(input_type="ascii", num_responses=3):
    """
    Main function to generate grids (ASCII or visual) based on grid elements.

    Parameters:
    - input_type (str): Type of grid to generate ('ascii' or 'vis'). Defaults to 'ascii'.
    - num_responses (int): Number of responses to generate. Defaults to 3.

    Depending on the input type:
    - 'ascii': Generates ASCII grids and saves them as text files.
    - 'vis': Generates visual grids and saves them as image files.

    Saves the outputs to the 'output_elements_to_grid' directory.
    """
    os.makedirs("output_elements_to_grid", exist_ok=True)

    # Load grid elements from file
    with open("input_visualprog_task1_gridelements.txt") as f:
        elements_text = f.read()

    if input_type == "ascii":
        # ASCII grid generation
        responses = task_elements_to_grid_ascii(elements_text, "system_prompt_ascii.txt", num_responses)
        for i, response in enumerate(responses):
            with open(f"output_elements_to_grid/I9_{chr(97 + i)}.txt", "w") as file:
                file.write(response)
                print(f"Grid generated and saved to output_elements_to_grid/I9_{chr(97 + i)}.txt")
    elif input_type == "vis":
        # Visual grid generation
        responses = task_elements_to_grid_vis(elements_text, "system_prompt_vis.txt", num_responses)
        for i, choice in enumerate(responses):
            image_data = choice.b64_json
            save_image_from_b64(image_data, f"output_elements_to_grid/I10_{chr(97 + i)}.png")
            print(f"Grid generated and saved to output_elements_to_grid/I10_{chr(97 + i)}.png")
    else:
        print("Invalid input_type. Choose either 'ascii' or 'vis'.")

if __name__ == "__main__":
    # Load the OpenAI API key from a file.
    with open("<<OPENAI_API_KEY_FILE>>") as f:
        content = f.read().strip()
    # Extract the API key from the third line of the file
    openai_key = content.split("\n")[2]
    openai.api_key = openai_key

    # Choose 'ascii' for text-based grid or 'vis' for image-based grid.
    input_type = "ascii"
    # input_type = "vis"
    main(input_type=input_type)
