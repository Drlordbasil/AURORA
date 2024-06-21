from kivy.utils import get_color_from_hex

class Theme:
    def __init__(self, name, colors):
        self.name = name
        self.colors = colors

class ThemeManager:
    def __init__(self):
        self.themes = {
            'Dark': Theme('Dark', {
                'background': '#000000',
                'text': '#33ff33',
                'send_button': '#1E90FF',
                'clear_button': '#DC143C',
                'record_button': '#36D936',
            }),
            'Light': Theme('Light', {
                'background': '#FFFFFF',
                'text': '#000000',
                'send_button': '#4169E1',
                'clear_button': '#B22222',
                'record_button': '#2ECC71',
            })
        }
        self.current_theme = self.themes['Dark']
        self.root = None
        self.prompt_input = None
        self.status_label = None
        self.send_button = None
        self.clear_button = None
        self.record_button = None

    def set_theme(self, theme_name):
        self.current_theme = self.themes[theme_name]
        self.apply_theme()

    def apply_theme(self):
        if self.root is None:
            return  # UI elements not yet set
        colors = self.current_theme.colors
        self.root.canvas.before.children[0].rgba = get_color_from_hex(colors['background'])
        self.prompt_input.background_color = get_color_from_hex(colors['background'])
        self.prompt_input.foreground_color = get_color_from_hex(colors['text'])
        self.status_label.color = get_color_from_hex(colors['text'])
        self.send_button.background_color = get_color_from_hex(colors['send_button'])
        self.clear_button.background_color = get_color_from_hex(colors['clear_button'])
        self.record_button.background_color = get_color_from_hex(colors['record_button'])

    def set_ui_elements(self, root, prompt_input, status_label, send_button, clear_button, record_button):
        self.root = root
        self.prompt_input = prompt_input
        self.status_label = status_label
        self.send_button = send_button
        self.clear_button = clear_button
        self.record_button = record_button
        self.apply_theme()  # Apply the current theme to the newly set UI elements
