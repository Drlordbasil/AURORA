import json
import time
import logging
from queue import Queue
from collections import defaultdict
from datasets import load_dataset
from Brain_modules.lobes.frontal_lobe import FrontalLobe
from Brain_modules.lobes.parietal_lobe import ParietalLobe
from Brain_modules.lobes.temporal_lobe import TemporalLobe
from Brain_modules.lobes.occipital_lobe import OccipitalLobe
from Brain_modules.lobes.limbic_lobe import LimbicLobe
from Brain_modules.lobes.cerebellar_lobe import CerebellarLobe
from Brain_modules.lobes.insular_cortex import InsularCortex
from Brain_modules.lobes.association_areas import AssociationAreas
from Brain_modules.lobes.wernickes_area import WernickesArea

class LobesProcessing:
    def __init__(self, image_vision):
        """
        Initializes the LobesProcessing class and sets up the individual lobes.
        """
        self.image_vision = image_vision
        self.lobes = self._initialize_lobes()
        self.responses = Queue()

    def _initialize_lobes(self):
        """
        Initializes all the lobes and returns a dictionary of lobe instances.
        """
        lobe_classes = {
            "frontal": FrontalLobe,
            "parietal": ParietalLobe,
            "temporal": TemporalLobe,
            "occipital": OccipitalLobe,
            "limbic": LimbicLobe,
            "cerebellar": CerebellarLobe,
            "wernickes_area": WernickesArea,
            "insular": InsularCortex,
            "association_areas": AssociationAreas
        }
        return {name: lobe_class() for name, lobe_class in lobe_classes.items()}

    def process_lobe(self, lobe_name, prompt):
        """
        Processes the given prompt using the specified lobe.

        Args:
            lobe_name (str): The name of the lobe to process the prompt.
            prompt (str): The input prompt.

        Returns:
            str: The response from the lobe.
        """
        lobe = self.lobes.get(lobe_name)
        if lobe:
            return lobe.process(prompt)
        else:
            return f"Error: {lobe_name} processing method not found."

    def process_all_lobes(self, prompt):
        """
        Processes the given prompt using all lobes and aggregates their responses.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The combined response from all lobes.
        """
        combined_responses = []
        for lobe_name, lobe in self.lobes.items():
            response = lobe.process(prompt)
            self.responses.put((lobe_name, response))
            combined_responses.append(f"{lobe_name.capitalize()} Lobe: {response}")
        
        combined_thought = self._generate_inner_voice(combined_responses)
        return combined_thought

    def _generate_inner_voice(self, responses):
        """
        Generates a cohesive inner voice from the aggregated lobe responses.

        Args:
            responses (list): The list of responses from each lobe.

        Returns:
            str: The cohesive inner voice.
        """
        cohesive_thought = ". ".join(responses)
        return f"Inner Voice: {cohesive_thought}"

    def load_and_preprocess_dataset(self, dataset_name, split='train', percentage=0.01):
        """
        Loads and preprocesses a dataset for training.

        Args:
            dataset_name (str): The name of the dataset to load.
            split (str): The dataset split to load (default is 'train').
            percentage (float): The percentage of the dataset to load (default is 0.01).

        Returns:
            list: A list of preprocessed data points.
        """
        dataset = load_dataset(dataset_name, split=f"{split}[:{int(percentage * 100)}%]")
        processed_data = [{"text": item["text"], "label": item.get("label", None)} for item in dataset]
        return processed_data

    def train_lobes_with_dataset(self, dataset_name, split='train', percentage=0.01):
        """
        Trains the lobes using a specified dataset.

        Args:
            dataset_name (str): The name of the dataset to train with.
            split (str): The dataset split to use (default is 'train').
            percentage (float): The percentage of the dataset to use (default is 0.01).
        """
        dataset = self.load_and_preprocess_dataset(dataset_name, split, percentage)
        for data in dataset:
            prompt = data["text"]
            for lobe_name in self.lobes.keys():
                self.process_lobe(lobe_name, prompt)

if __name__ == "__main__":
    lobes_processor = LobesProcessing(image_vision=False)
    
    # Example to train with a small percentage of a Hugging Face dataset
    lobes_processor.train_lobes_with_dataset('ag_news', percentage=0.01)
    
    test_prompts = [
        "The box is above the table, near the window",
        "I feel a rough texture and cold temperature",
        "Navigate to the nearest exit using the map",
        "Calculate the distance between points A (2,3) and B (5,7)",
        "The room temperature is 72 degrees",
        "Process this sentence without any spatial or numerical content",
        "Error: setting an array element with a sequence",
        "sup man its me Anthony"
    ]
    
    for prompt in test_prompts:
        final_thought = lobes_processor.process_all_lobes(prompt)
        print(f"\nFinal Combined Thought for prompt '{prompt}':\n{final_thought}\n")
