# lobes_processing.py

import json
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from textblob import TextBlob
from nltk import word_tokenize, pos_tag
import pyautogui
import os
from Brain_modules.image_vision import ImageVision
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

# Importing the lobes from their respective files
from Brain_modules.lobes.frontal_lobe import FrontalLobe
from Brain_modules.lobes.parietal_lobe import ParietalLobe
from Brain_modules.lobes.temporal_lobe import TemporalLobe
from Brain_modules.lobes.occipital_lobe import OccipitalLobe
from Brain_modules.lobes.limbic_lobe import LimbicLobe
from Brain_modules.lobes.cerebellar_lobe import CerebellarLobe
from Brain_modules.lobes.brocas_area import BrocasArea as BrocasAreaLobe
from Brain_modules.lobes.insular_cortex import InsularCortex as InsularCortexLobe
from Brain_modules.lobes.association_areas import AssociationAreas as AssociationAreasLobe
from Brain_modules.lobes.wernickes_area import WernickesArea

class LobesProcessing:
    def __init__(self, image_vision):
        self.image_vision = image_vision
        self.lobes = self._initialize_lobes()

    def _initialize_lobes(self):
        lobe_classes = {
            "frontal": FrontalLobe,
            "parietal": ParietalLobe,
            "temporal": TemporalLobe,
            "occipital": OccipitalLobe,
            "limbic": LimbicLobe,
            "cerebellar": CerebellarLobe,
            "brocas_area": BrocasAreaLobe,
            "wernickes_area": WernickesArea,  # Assuming listen_lobe corresponds to wernickes_area
            "insular": InsularCortexLobe,
            "association_areas": AssociationAreasLobe
        }
        return {name: lobe_class() for name, lobe_class in lobe_classes.items()}

    def process_lobe(self, lobe_name, prompt):
        lobe = self.lobes.get(lobe_name)
        if lobe:
            return lobe.process(prompt)
        else:
            return f"Error: {lobe_name} processing method not found."

# Example usage:
# image_vision = ImageVision()
# lobes_processor = LobesProcessing(image_vision)
# result = lobes_processor.process_lobe("frontal", "Analyze this text")
# print(result)
