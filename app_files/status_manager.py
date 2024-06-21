from kivy.animation import Animation

class StatusManager:
    def __init__(self, status_label):
        self.status_label = status_label
        self.default_color = (0.2, 1, 0.2, 1)
        self.animation = None

    def update(self, message, animate=False):
        self.status_label.text = message
        if animate:
            if self.animation:
                self.animation.cancel(self.status_label)
            self.animation = Animation(color=self.default_color, duration=0.5) + Animation(color=self.default_color, duration=0.5)
            self.animation.repeat = True
            self.animation.start(self.status_label)
        else:
            if self.animation:
                self.animation.cancel(self.status_label)
            self.status_label.color = self.default_color
