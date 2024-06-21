import os
import threading
import json
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.uix.spinner import Spinner
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.utils import get_color_from_hex

from theme import ThemeManager
from status_manager import StatusManager
from recording_manager import RecordingManager
from ui_components import BubbleLabel
from Brain_modules.brain import Brain
from Brain_modules.llm_api_calls import LLM_API_Calls
from Brain_modules.listen_lobe import AuroraRecorder
from speaker import text_to_speech  # Add this import

class AuroraApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme_manager = ThemeManager()
        self.status_manager = None
        self.recording_manager = None

    def build(self):
        Window.bind(on_request_close=self.on_request_close)

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            api_key = self.prompt_for_api_key()
        if not api_key:
            self.show_error("Error", "GROQ_API_KEY is required to run the application.")
            return

        self.root = BoxLayout(orientation='vertical', padding=10, spacing=10)

        self.header = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))

        self.body = BoxLayout(orientation='horizontal', size_hint=(1, 0.9), padding=10, spacing=10)
        
        left_layout = BoxLayout(orientation='vertical', size_hint=(0.75, 1), spacing=10)
        right_layout = BoxLayout(orientation='vertical', size_hint=(0.25, 1), spacing=10)

        self.chat_display = ScrollView(size_hint=(1, 0.8))
        self.chat_content = BoxLayout(orientation='vertical', size_hint_y=None, spacing=10, padding=10)
        self.chat_content.bind(minimum_height=self.chat_content.setter('height'))
        self.chat_display.add_widget(self.chat_content)
        
        self.prompt_input = TextInput(
            size_hint=(1, 0.1), multiline=False, font_size=20,
            background_color=(0, 0, 0, 1), foreground_color=(0.2, 1, 0.2, 1),
            cursor_color=(0.2, 1, 0.2, 1)
        )
        self.prompt_input.bind(on_text_validate=self.on_send_button_press)
        
        self.buttons_layout = BoxLayout(size_hint=(1, 0.1), spacing=10)
        self.send_button = Button(
            text="Send", size_hint=(0.3, 1),
            background_color=(0.2, 0.6, 0.86, 1), font_size=20, color=(1, 1, 1, 1)
        )
        self.send_button.bind(on_press=self.on_send_button_press)
        self.clear_button = Button(
            text="Clear Chat", size_hint=(0.3, 1),
            background_color=(0.86, 0.36, 0.36, 1), font_size=20, color=(1, 1, 1, 1)
        )
        self.clear_button.bind(on_press=self.clear_chat)
        self.record_button = Button(
            text="Record", size_hint=(0.3, 1),
            background_color=(0.36, 0.86, 0.36, 1), font_size=20, color=(1, 1, 1, 1)
        )
        
        self.buttons_layout.add_widget(self.send_button)
        self.buttons_layout.add_widget(self.clear_button)
        self.buttons_layout.add_widget(self.record_button)

        left_layout.add_widget(self.chat_display)
        left_layout.add_widget(self.prompt_input)
        left_layout.add_widget(self.buttons_layout)

        info_label = Label(
            text=(
                "AURORA: Artificial Unified Responsive Optimized Reasoning Agent\n\nFeatures:\n"
                "- Execute local commands\n- Perform web research\n- Analyze images\n"
                "- Extract text from PDFs\n- Analyze sentiment\n- Voice recognition\n\n"
                "Aurora assists you with a variety of tasks by using its advanced AI capabilities."
            ),
            size_hint=(1, 0.6), font_size="16sp", color=(0.2, 1, 0.2, 1),
            halign="center", valign="top", text_size=(300, None)
        )

        self.status_label = Label(
            size_hint=(1, 0.1), font_size=16, color=(0.2, 1, 0.2, 1),
            halign="center", valign="middle"
        )
        self.status_label.bind(size=self._update_status_rect)

        logo = Image(source='aurora.png', size_hint=(1, 0.3))

        right_layout.add_widget(logo)
        right_layout.add_widget(info_label)
        right_layout.add_widget(self.status_label)

        self.body.add_widget(left_layout)
        self.body.add_widget(right_layout)

        self.theme_spinner = Spinner(
            text='Dark Theme',
            values=('Dark Theme', 'Light Theme'),
            size_hint=(None, None),
            size=(150, 44),
            pos_hint={'center_x': 0.9, 'center_y': 0.5}
        )
        self.theme_spinner.bind(text=self.on_theme_change)
        self.header.add_widget(self.theme_spinner)

        self.root.add_widget(self.header)
        self.root.add_widget(self.body)

        with self.root.canvas.before:
            Color(0, 0, 0, 1)
            self.rect = Rectangle(size=self.root.size, pos=self.root.pos)
            self.root.bind(size=self._update_rect, pos=self._update_rect)

        self.brain = Brain(api_key, self.update_status)
        self.llm_api_call = LLM_API_Calls(self.update_status)
        self.status_manager = StatusManager(self.status_label)
        self.recording_manager = RecordingManager(
            AuroraRecorder(callback=self.on_transcription),
            self.record_button,
            self.on_transcription
        )
        self.record_button.bind(on_press=lambda x: self.recording_manager.toggle_recording())
        self.spinner = None

        # Set UI elements in the theme manager
        self.theme_manager.set_ui_elements(
            self.root, self.prompt_input, self.status_label,
            self.send_button, self.clear_button, self.record_button
        )

        self.load_chat_history()
        return self.root

    def prompt_for_api_key(self):
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.api_key_input = TextInput(hint_text='Enter your GROQ API key', multiline=False)
        submit_button = Button(text='Submit', size_hint_y=None, height=44)
        
        layout.add_widget(Label(text='GROQ API Key not found in environment variables.'))
        layout.add_widget(self.api_key_input)
        layout.add_widget(submit_button)

        popup = Popup(title='Enter API Key', content=layout, size_hint=(None, None), size=(400, 200))
        
        submit_button.bind(on_press=lambda x: self.set_api_key(popup))
        popup.open()
        return self.api_key_input.text

    def set_api_key(self, popup):
        os.environ["GROQ_API_KEY"] = self.api_key_input.text
        popup.dismiss()

    def on_theme_change(self, spinner, text):
        self.set_theme(text.split()[0])  # 'Dark Theme' -> 'Dark'

    def set_theme(self, theme_name):
        self.theme_manager.set_theme(theme_name)

    def show_help(self, instance):
        help_popup = Popup(
            title="Help",
            content=Label(text="This is the help section. You can find information about how to use the application here.", color=(0.2, 1, 0.2, 1)),
            size_hint=(None, None),
            size=(400, 400)
        )
        help_popup.open()

    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

    def _update_status_rect(self, instance, value):
        instance.text_size = instance.size

    def show_error(self, title, message):
        popup = Popup(
            title=title, content=Label(text=message, color=(1, 0, 0, 1)),
            size_hint=(None, None), size=(400, 200)
        )
        popup.open()

    def on_send_button_press(self, instance):
        prompt = self.prompt_input.text
        if prompt.lower() in ["exit", "quit"]:
            App.get_running_app().stop()
        else:
            self.display_message(f"You: {prompt}", user=True)
            self.show_loading_spinner()
            threading.Thread(target=self.send_message, args=(prompt,)).start()
            self.prompt_input.text = ""

    def show_loading_spinner(self):
        if not self.spinner:
            self.spinner = Spinner(
                text='Loading...', size_hint=(None, None), size=(50, 50),
                pos_hint={'center_x': 0.5, 'center_y': 0.5}, color=(0.2, 1, 0.2, 1)
            )
        self.root.add_widget(self.spinner)

    def hide_loading_spinner(self):
        if self.spinner:
            self.root.remove_widget(self.spinner)
            self.spinner = None

    def send_message(self, prompt):
        try:
            self.update_status("Processing...", animation=True)
            response = self.brain.central_processing_agent(prompt)
            Clock.schedule_once(lambda dt: self.display_message(f"AURORA: {response}", user=False), 0)
            Clock.schedule_once(lambda dt: self.update_status("Completed"), 0)
            # Use text_to_speech to play the response
            text_to_speech(response)
        except Exception as e:
            Clock.schedule_once(lambda dt: self.display_message(f"Error processing prompt: {e}", user=False), 0)
            Clock.schedule_once(lambda dt: self.update_status("Error"), 0)
        finally:
            Clock.schedule_once(lambda dt: self.hide_loading_spinner(), 0)

    def display_message(self, message, user=True):
        background_color = get_color_from_hex("#000000") if user else get_color_from_hex("#008000")
        bubble = BubbleLabel(text=message, background_color=background_color)
        self.chat_content.add_widget(bubble)

    def clear_chat(self, instance):
        self.chat_content.clear_widgets()

    def update_status(self, message, animation=False):
        Clock.schedule_once(lambda dt: self.status_manager.update(message, animation), 0)

    def on_transcription(self, text):
        Clock.schedule_once(lambda dt: self.display_message(f"You (voice): {text}", user=True), 0)
        Clock.schedule_once(lambda dt: self.send_message(text), 0)

    def on_request_close(self, *args):
        if hasattr(self, 'recording_manager'):
            self.recording_manager.stop_recording()
        self.save_chat_history()
        App.get_running_app().stop()
        return True

    def save_chat_history(self):
        history = [{'text': child.children[0].text, 'user': isinstance(child.canvas.before.children[1], Color)} for child in self.chat_content.children]
        with open('chat_history.json', 'w') as f:
            json.dump(history, f)

    def load_chat_history(self):
        try:
            with open('chat_history.json', 'r') as f:
                history = json.load(f)
            for message in reversed(history):
                self.display_message(message['text'], user=message['user'])
        except FileNotFoundError:
            pass

    def on_stop(self):
        self.save_chat_history()
