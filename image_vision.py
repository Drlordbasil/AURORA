import os
import requests
import ollama
from io import BytesIO
from PIL import Image

class ImageVision:
    def download_image(self, image_url):
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error downloading image: {str(e)}")
        except IOError as e:
            raise Exception(f"Error opening image: {str(e)}")

    def save_image(self, image, image_path):
        try:
            if image.mode == "RGBA":
                # Convert RGBA image to RGB mode
                image = image.convert("RGB")
            image.save(image_path)
        except IOError as e:
            raise Exception(f"Error saving image: {str(e)}")

    def analyze_image(self, image_url):
        try:
            image = self.download_image(image_url)
            image_path = "temp_image.jpg"
            self.save_image(image, image_path)

            res = ollama.chat(
                model="llava-llama3",
                messages=[
                    {
                        'role': 'user',
                        'content': 'Describe this image in extremely verbose robust detail using a large vocabulary:',
                        'images': [image_path]
                    }
                ]
            )

            image_description = res['message']['content']
            os.remove(image_path)  # Remove the temporary image file

            return image_description
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
