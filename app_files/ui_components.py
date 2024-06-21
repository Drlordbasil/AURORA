from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.graphics import Color, RoundedRectangle

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
            text_size=(600, None),
            font_size="18sp",
            color=(0.2, 1, 0.2, 1),
            bold=True,
        )
        label.bind(texture_size=label.setter("size"))

        self.add_widget(label)
        self.bind(minimum_height=self.setter("height"))

    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos
