import os
import sounddevice as sd
import soundfile as sf
from groq import Groq
import threading

class AuroraRecorder:
    def __init__(self):
        """Initialize the AuroraRecorder with default settings."""
        self.client = Groq()
        self.recording = False
        self.audio_path = "output.wav"
        self.transcription = None
        self._recording_thread = None
        self._lock = threading.Lock()

    def start_recording(self):
        """Start recording audio."""
        with self._lock:
            if self.recording:
                print("Already recording")
                return
            self.recording = True
            self.transcription = None
            self._recording_thread = threading.Thread(target=self._record_audio)
            self._recording_thread.start()

    def stop_recording(self):
        """Stop recording audio."""
        with self._lock:
            if not self.recording:
                print("Not recording")
                return
            self.recording = False
            if self._recording_thread:
                self._recording_thread.join()  # Ensure the recording thread has finished

    def _record_audio(self):
        """Record audio in a separate thread."""
        print("Recording audio...")
        samplerate = 16000  # 16kHz
        channels = 1  # mono
        recording = sd.rec(int(60 * samplerate), samplerate=samplerate, channels=channels)
        while self.recording:
            sd.sleep(100)
        sd.stop()
        sf.write(self.audio_path, recording, samplerate)
        print(f"Audio recorded and saved to {self.audio_path}.")
        self.transcribe_audio(self.audio_path)

    def transcribe_audio(self, file_path):
        """Transcribe the recorded audio using Groq."""
        if not os.path.isfile(file_path):
            print("The provided file path is not valid.")
            return None
        
        with open(file_path, "rb") as file:
            response = self.client.audio.transcriptions.create(
                file=(file_path, file.read()),
                model="whisper-large-v3",
                response_format="json",
                language="en",
                temperature=0.0
            )
        
        self.transcription = response.text
        print(f"Transcription: {self.transcription}")
