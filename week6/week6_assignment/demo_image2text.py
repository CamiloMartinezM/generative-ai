import openai
import base64
import os

def encode_image(image_path):
    """Encode an image file into a Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image_path, output_dir="output_demo_image2text", num_responses=3):
    """
    Analyze an image to extract descriptive keywords.

    Parameters:
    - image_path (str): Path to the input image file.
    - output_dir (str): Directory to save the output files. Defaults to 'output_demo_image2text'.
    - num_responses (int): Number of responses to generate. Defaults to 3.

    This function encodes the image into Base64 format, sends it to the OpenAI API
    for analysis, and saves the descriptive keywords to text files in the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    encoded_image = encode_image(image_path)

    system_prompt = "You are an image analysis assistant. Analyze the visual input and provide a list of descriptive keywords for the main elements in the image."
    user_prompt = "Analyze this image and provide descriptive keywords."

    image_data = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
    }

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [user_prompt, image_data]}
        ],
        n=num_responses
    )

    for i, choice in enumerate(response.choices):
        keywords = choice.message.content.strip()
        output_file = os.path.join(output_dir, f"I1_{chr(97 + i)}.txt")

        with open(output_file, "w") as file:
            file.write(keywords)

        print(f"Keywords extracted and saved to {output_file}")

def main():
    """Main function to analyze a demo image."""
    analyze_image("input_demo_image2text.png")

if __name__ == "__main__":
    # Load the OpenAI API key from a file.
    with open("<<OPENAI_API_KEY_FILE>>") as f:
        content = f.read().strip()
    # Extract the API key from the third line of the file.
    openai_key = content.split("\n")[2]
    openai.api_key = openai_key

    main()