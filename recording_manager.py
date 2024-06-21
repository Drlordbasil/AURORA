class RecordingManager:
    def __init__(self, recorder, record_button, callback):
        self.recorder = recorder
        self.record_button = record_button
        self.callback = callback

    def toggle_recording(self):
        if self.record_button.text == "Record":
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.recorder.start_recording()
        self.record_button.text = "Stop Recording"
        self.record_button.background_color = (0.86, 0.36, 0.36, 1)

    def stop_recording(self):
        self.recorder.stop_recording()
        self.record_button.text = "Record"
        self.record_button.background_color = (0.36, 0.86, 0.36, 1)

    def on_transcription(self, text):
        self.callback(text)
