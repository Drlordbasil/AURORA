# frontal_lobe.py

import numpy as np
import json
import time
from collections import deque

class FrontalLobe:
    def __init__(self):
        self.working_memory = deque(maxlen=7)
        self.attention_focus = None
        self.emotional_state = 'neutral'
        self.inhibition_threshold = 0.7
        self.planning_depth = 3
        self.priority_words = set(['urgent', 'important', 'critical', 'immediately', 'deadline', 'crucial'])

    def process(self, prompt):
        print(f"Frontal lobe processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            self.update_working_memory(prompt)
            self.focus_attention(prompt)
            plan = self.executive_function(prompt)
            self.regulate_emotion(prompt)
            decision = self.make_decision(plan)

            for i in range(3):
                time.sleep(0.5)
                print(f"Frontal lobe processing: {time.strftime('%Y-%m-%d %H:%M:%S')} - Step {i+1}")

            analysis = {
                "Working Memory": list(self.working_memory),
                "Attention Focus": self.attention_focus,
                "Emotional State": self.emotional_state,
                "Executive Function Plan": plan,
                "Decision": decision
            }

            return f"Frontal Lobe Response: {decision} Full analysis: {json.dumps(analysis, indent=2)}"
        except Exception as e:
            return f"Frontal Lobe Response: Error in processing: {str(e)}. Redirecting to problem-solving circuits."

    def update_working_memory(self, prompt):
        words = prompt.split()
        for word in words:
            self.working_memory.append(word)

    def focus_attention(self, prompt):
        words = prompt.lower().split()
        for word in words:
            if word in self.priority_words:
                self.attention_focus = word
                return
        self.attention_focus = words[0] if words else None

    def executive_function(self, prompt):
        plan = []
        words = prompt.split()
        priority = 1.0 if self.attention_focus in self.priority_words else 0.5
        for i in range(min(self.planning_depth, len(words))):
            if self.should_inhibit(words[i]):
                plan.append(f"Inhibit response to '{words[i]}'")
            else:
                plan.append(f"Process '{words[i]}' with priority {priority:.1f}")
            priority *= 0.9  # Decrease priority for subsequent actions
        return plan

    def should_inhibit(self, word):
        return (len(word) / 10 > self.inhibition_threshold) and (word.lower() not in self.priority_words)

    def regulate_emotion(self, prompt):
        emotion_words = {
            'positive': ['happy', 'good', 'excellent', 'wonderful', 'excited', 'optimistic'],
            'negative': ['sad', 'bad', 'terrible', 'awful', 'anxious', 'frustrated'],
            'neutral': ['consider', 'analyze', 'evaluate', 'assess', 'review']
        }
        
        for emotion, words in emotion_words.items():
            if any(word in prompt.lower() for word in words):
                self.emotional_state = emotion
                return

    def make_decision(self, plan):
        inhibition_count = sum(1 for action in plan if 'Inhibit' in action)
        priority_sum = sum(float(action.split()[-1]) for action in plan if 'priority' in action)

        if self.attention_focus in self.priority_words:
            return f"High priority task detected: {self.attention_focus}. Allocating maximum resources."
        elif inhibition_count > len(plan) / 2:
            return "Multiple inhibitions required. Proceed with caution and reassess."
        elif priority_sum > 2.0:
            return "Important tasks identified. Focusing on high-priority actions."
        elif self.emotional_state != 'neutral':
            return f"Emotional context detected: {self.emotional_state}. Adjusting cognitive approach."
        else:
            return "Standard processing. Execute planned actions sequentially."

if __name__ == "__main__":
    frontal_lobe = FrontalLobe()
    test_prompts = [
        "Urgently need to finish the project report",
        "Feeling happy about the team's progress",
        "Consider the long-term consequences of this decision",
        "Resist the impulse to react negatively",
        "Multitask between coding and attending the meeting",
        "Critical bug detected in production environment",
        "Analyze the market trends for the next quarter",
        "Feeling anxious about the upcoming presentation"
    ]
    for prompt in test_prompts:
        print(f"\nTesting prompt: '{prompt}'")
        result = frontal_lobe.process(prompt)
        print(result)
