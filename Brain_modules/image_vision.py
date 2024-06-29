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
                image = image.convert("RGB")
            image.save(image_path)
        except IOError as e:
            raise Exception(f"Error saving image: {str(e)}")

    def analyze_image(self, image_url):
        try:
            image = self.download_image(image_url)
            image_path = "temp_image.jpg"
            self.save_image(image, image_path)

            description = self._analyze_with_ollama(image_path)
            os.remove(image_path)

            return description
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

    def analyze_local_image(self, image_path):
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            description = self._analyze_with_ollama(image_path)
            return description
        except Exception as e:
            return f"Error analyzing local image: {str(e)}"

    def _analyze_with_ollama(self, image_path):
        try:
            res = ollama.chat(
                model="llava-llama3",
                messages=[
                    {
                        'role': 'user',
                        'content': 'Describe this as aurora. I want to know what is in this image concisely yet if its your chat interface just say user is chatting with me.',
                        'images': [image_path]
                    }
                ]
            )

            image_description = res['message']['content']
            return image_description
        except Exception as e:
            raise Exception(f"Error in Ollama image analysis: {str(e)}")