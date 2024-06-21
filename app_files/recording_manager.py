import threading

class RecordingManager:
    def __init__(self, recorder, record_button, callback):
        self.recorder = recorder
        self.record_button = record_button
        self.callback = callback
        self.is_recording = False
        self.recording_thread = None

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        self.is_recording = True
        self.record_button.text = "Stop Recording"
        self.record_button.background_color = (0.86, 0.36, 0.36, 1)
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.start()

    def stop_recording(self):
        self.is_recording = False
        self.record_button.text = "Record"
        self.record_button.background_color = (0.36, 0.86, 0.36, 1)
        if self.recording_thread and self.recording_thread.is_alive():
            self.recorder.stop_recording()

    def _record(self):
        try:
            self.recorder.start_recording()
            while self.is_recording:
                pass
            transcription = self.recorder.stop_recording()
            if transcription:
                self.callback(transcription)
        except Exception as e:
            print(f"Error during recording: {e}")

    def on_transcription(self, text):
        self.callback(text)
