import os
import sounddevice as sd
import soundfile as sf
from groq import Groq
import threading

class AuroraRecorder:
    def __init__(self):
        self.client = Groq()
        self.recording = False
        self.audio_path = "output.wav"
        self.transcription = None

    def start_recording(self):
        if self.recording:
            print("Already recording")
            return
        self.recording = True
        threading.Thread(target=self._record_audio).start()

    def stop_recording(self):
        if not self.recording:
            print("Not recording")
            return
        self.recording = False

    def _record_audio(self):
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
