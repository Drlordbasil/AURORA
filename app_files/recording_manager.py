import threading

class RecordingManager:
    def __init__(self, recorder, callback):
        self.recorder = recorder
        self.callback = callback
        self.is_recording = False
        self.recording_thread = None

    def start_recording(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.start()

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join()
        transcription = self.recorder.transcription
        if transcription:
            self.callback(transcription)
        return transcription

    def _record(self):
        try:
            self.recorder.start_recording()
            while self.is_recording:
                pass
            self.recorder.stop_recording()
        except Exception as e:
            print(f"Error during recording: {e}")
