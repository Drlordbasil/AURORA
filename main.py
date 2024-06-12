import os
import threading
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.clock import Clock
from kivy.utils import get_color_from_hex
from brain import Brain

class BubbleLabel(BoxLayout):
    def __init__(self, text, background_color, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.size_hint_y = None
        self.padding = 10
        self.spacing = 5

        with self.canvas.before:
            Color(*background_color)
            self.rect = RoundedRectangle(size=self.size, pos=self.pos, radius=[10])
            self.bind(size=self._update_rect, pos=self._update_rect)

        label = Label(
            text=text,
            size_hint_y=None,
            halign="left",
            valign="middle",
            text_size=(600, None),  # Set a maximum width for text wrapping
            font_size="18sp",
            color=(1, 1, 1, 1),
            bold=True,
        )
        label.bind(texture_size=label.setter("size"))

        self.add_widget(label)
        self.bind(minimum_height=self.setter("height"))

    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

class ChatbotApp(App):
    def build(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            self.show_error("Error", "GROQ_API_KEY environment variable not set.")
            return

        self.brain = Brain(api_key)
        
        self.root = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        self.chat_display = ScrollView(size_hint=(1, 0.8))
        self.chat_content = BoxLayout(orientation='vertical', size_hint_y=None, spacing=10, padding=10)
        self.chat_content.bind(minimum_height=self.chat_content.setter('height'))
        self.chat_display.add_widget(self.chat_content)
        
        self.prompt_input = TextInput(size_hint=(1, 0.1), multiline=False, font_size=20, background_color=(0.1, 0.1, 0.1, 1), foreground_color=(1, 1, 1, 1), cursor_color=(1, 1, 1, 1))
        self.prompt_input.bind(on_text_validate=self.on_send_button_press)
        
        self.buttons_layout = BoxLayout(size_hint=(1, 0.1), spacing=10)
        self.send_button = Button(text="Send", size_hint=(0.5, 1), background_color=(0.2, 0.6, 0.86, 1), font_size=20)
        self.send_button.bind(on_press=self.on_send_button_press)
        self.clear_button = Button(text="Clear Chat", size_hint=(0.5, 1), background_color=(0.86, 0.36, 0.36, 1), font_size=20)
        self.clear_button.bind(on_press=self.clear_chat)
        
        self.buttons_layout.add_widget(self.send_button)
        self.buttons_layout.add_widget(self.clear_button)
        
        self.root.add_widget(self.chat_display)
        self.root.add_widget(self.prompt_input)
        self.root.add_widget(self.buttons_layout)
        
        with self.root.canvas.before:
            Color(0.15, 0.15, 0.15, 1)  # Background color
            self.rect = Rectangle(size=self.root.size, pos=self.root.pos)
            self.root.bind(size=self._update_rect, pos=self._update_rect)
        
        return self.root

    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

    def show_error(self, title, message):
        popup = Popup(title=title, content=Label(text=message), size_hint=(None, None), size=(400, 200))
        popup.open()

    def on_send_button_press(self, instance):
        prompt = self.prompt_input.text
        if prompt.lower() in ["exit", "quit"]:
            App.get_running_app().stop()
        else:
            self.display_message(f"You: {prompt}", user=True)
            threading.Thread(target=self.send_message, args=(prompt,)).start()
            self.prompt_input.text = ""

    def send_message(self, prompt):
        try:
            response = self.brain.central_processing_agent(prompt)
            Clock.schedule_once(lambda dt: self.display_message(f"AURORA: {response}", user=False), 0)
        except Exception as e:
            Clock.schedule_once(lambda dt: self.display_message(f"Error processing prompt: {e}", user=False), 0)

    def display_message(self, message, user=True):
        background_color = get_color_from_hex("#1e1e1e") if user else get_color_from_hex("#2e7d32")
        bubble = BubbleLabel(text=message, background_color=background_color)
        self.chat_content.add_widget(bubble)
        self.chat_display.scroll_y = 0

    def clear_chat(self, instance):
        self.chat_content.clear_widgets()

if __name__ == "__main__":
    ChatbotApp().run()
