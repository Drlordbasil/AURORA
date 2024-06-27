import numpy as np
import json
import time
from typing import Dict, Any, List, Tuple
import random

class FrontalLobe:
    def __init__(self):
        self.working_memory = []
        self.attention_focus = None
        self.emotional_state = 'neutral'
        self.inhibition_control = 0.7
        self.planning_depth = 5
        self.priority_words = set(['urgent', 'important', 'critical', 'immediately', 'deadline', 'crucial'])
        self.decision_history = []
        self.tools = [
            {"name": "run_local_command", "description": "Execute a local command on the system"},
            {"name": "web_research", "description": "Perform a web research query"},
            {"name": "analyze_image", "description": "Analyze an image from a URL or local path"},
            {"name": "check_os_default_calendar", "description": "Check the calendar or create a calendar event"},
            {"name": "text_response", "description": "Provide a regular text response"}
        ]

    def process(self, prompt: str) -> Dict[str, Any]:
        self.update_working_memory(prompt)
        self.focus_attention(prompt)
        plan = self.executive_function(prompt)
        self.regulate_emotion(prompt)
        
        action, confidence = self.choose_action(prompt, plan)
        decision = self.make_decision(plan, action)
        
        self.update_decision_history(decision)

        analysis = self.generate_analysis(prompt, plan, decision, action, confidence)
        return {
            "response": decision["response"],
            "tool_call": decision.get("tool_call"),
            "analysis": analysis
        }

    def update_working_memory(self, prompt: str) -> None:
        self.working_memory = prompt.split()[-10:]  # Keep last 10 words

    def focus_attention(self, prompt: str) -> None:
        words = prompt.lower().split()
        self.attention_focus = next((word for word in words if word in self.priority_words), words[0] if words else None)

    def executive_function(self, prompt: str) -> List[Dict[str, Any]]:
        return [{'word': word, 'priority': 1.0 * (0.9 ** i), 'inhibit': self.should_inhibit(word), 
                 'action': 'Inhibit' if self.should_inhibit(word) else 'Process'}
                for i, word in enumerate(prompt.split()[:self.planning_depth])]

    def should_inhibit(self, word: str) -> bool:
        return (len(word) / 10 > self.inhibition_control) and (word.lower() not in self.priority_words)

    def regulate_emotion(self, prompt: str) -> None:
        emotion_words = {
            'positive': ['happy', 'good', 'excellent', 'wonderful', 'excited', 'optimistic'],
            'negative': ['sad', 'bad', 'terrible', 'awful', 'anxious', 'frustrated'],
            'neutral': ['consider', 'analyze', 'evaluate', 'assess', 'review']
        }
        self.emotional_state = next((emotion for emotion, words in emotion_words.items() 
                                     if any(word in prompt.lower() for word in words)), 'neutral')

    def choose_action(self, prompt: str, plan: List[Dict[str, Any]]) -> Tuple[str, float]:
        if any(word in prompt.lower() for word in ['run', 'execute', 'command']):
            return "run_local_command", 0.9
        elif any(word in prompt.lower() for word in ['research', 'search', 'find information']):
            return "web_research", 0.9
        elif any(word in prompt.lower() for word in ['image', 'picture', 'photo']):
            return "analyze_image", 0.9
        elif any(word in prompt.lower() for word in ['schedule', 'calendar', 'event']):
            return "check_os_default_calendar", 0.9
        else:
            return "text_response", 0.8

    def make_decision(self, plan: List[Dict[str, Any]], action: str) -> Dict[str, Any]:
        if action == "text_response":
            return self.generate_text_response(plan)
        return self.generate_tool_call(action, plan)

    def generate_text_response(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        priority_actions = [a for a in plan if not a['inhibit']]
        if priority_actions:
            return {"response": f"The frontal lobe suggests focusing on '{priority_actions[0]['word']}' as a key action."}
        return {"response": "The frontal lobe recommends careful consideration before proceeding."}

    def generate_tool_call(self, action: str, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        tool_call = {"name": action, "arguments": {}}
        priority_words = " ".join([a['word'] for a in plan if not a['inhibit']][:5])
        
        if action == "run_local_command":
            tool_call["arguments"]["command"] = priority_words
        elif action == "web_research":
            tool_call["arguments"]["query"] = priority_words
        elif action == "analyze_image":
            tool_call["arguments"]["image_url"] = "path/to/image.jpg"  # Placeholder
        elif action == "check_os_default_calendar":
            tool_call["arguments"]["date"] = time.strftime("%Y-%m-%d")
        
        return {
            "response": f"The frontal lobe recommends using the {action} tool for effective task execution.",
            "tool_call": tool_call
        }

    def update_decision_history(self, decision: Dict[str, Any]) -> None:
        self.decision_history.append({'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'decision': decision})
        if len(self.decision_history) > 50:
            self.decision_history.pop(0)

    def generate_analysis(self, prompt: str, plan: List[Dict[str, Any]], decision: Dict[str, Any], action: str, confidence: float) -> Dict[str, Any]:
        return {
            "Working Memory": self.working_memory,
            "Attention Focus": self.attention_focus,
            "Emotional State": self.emotional_state,
            "Executive Function Plan": plan,
            "Chosen Action": action,
            "Decision": decision,
            "Confidence": confidence,
            "Decision History": self.decision_history[-5:],
            "Input Prompt": prompt
        }

# test = FrontalLobe()
# print(test.process("Please run the command to execute the program"))
# print(test.process("I am feeling sad today"))