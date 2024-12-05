import openai
import os
import base64

def load_keywords(file_path):
    """
    Load keywords from a file.

    Parameters:
    - file_path (str): Path to the file containing keywords.

    Returns:
    - list: A list of keywords extracted from the file.

    The file should contain keywords separated by commas and a space (", ").
    """
    with open(file_path, "r") as file:
        keywords = file.read().strip().split(", ")
    return keywords

def generate_image_from_keywords(keywords, output_dir="output_demo_text2image", num_images=3):
    """
    Generate images based on provided keywords using the OpenAI API.

    Parameters:
    - keywords (list): A list of keywords to form the basis of the image prompt.
    - output_dir (str): Directory to save the generated images. Defaults to 'output_demo_text2image'.
    - num_images (int): Number of images to generate. Defaults to 3.

    This function creates a textual prompt from the keywords, generates images using the OpenAI API,
    and saves them as PNG files in the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    prompt = "A visual representation featuring: " + ", ".join(keywords)

    responses = []
    for i in range(num_images):
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="standard",
            response_format="b64_json"
        )

        # Access the base64 image data and save it as a PNG file
        image_data = response.data[0].b64_json
        image_path = os.path.join(output_dir, f"I2_{chr(97 + i)}.png")

        with open(image_path, "wb") as img_file:
            img_file.write(base64.b64decode(image_data))

        responses.append(image_path)
        print(f"Image generated and saved to {image_path}")

    return responses

def main():
    """
    Main function to generate images from keywords provided in a file.

    It reads the keywords from 'input_demo_text2image.txt', generates images
    based on the keywords, and saves them to the output directory.
    """
    keywords = load_keywords("input_demo_text2image.txt")
    generate_image_from_keywords(keywords)

if __name__ == "__main__":
    # Load the OpenAI API key from a file.
    with open("<<OPENAI_API_KEY_FILE>>") as f:
        content = f.read().strip()
    # Extract the API key from the third line of the file.
    openai_key = content.split("\n")[2]
    openai.api_key = openai_key

    main()