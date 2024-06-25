from deepgram import DeepgramClient, SpeakOptions
import os

def text_to_speech(text):
    if os.environ.get("DEEPGRAM_API_KEY") is None:
        print("Please set the DEEPGRAM_API_KEY environment variable.")
        return
    else:
        DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")

    FILENAME = "combined_audio.mp3"

    try:
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        options = SpeakOptions(
            model="aura-asteria-en",
        )

        if len(text) > 1999:
            chunks = [text[i:i + 1800] for i in range(0, len(text), 1800)]
        else:
            chunks = [text]

        with open(FILENAME, "wb") as combined_audio:
            for i, chunk in enumerate(chunks):
                chunk_filename = f"audio_chunk_{i}.mp3"
                response = deepgram.speak.v("1").save(chunk_filename, {"text": chunk}, options)
                with open(chunk_filename, "rb") as chunk_file:
                    combined_audio.write(chunk_file.read())
                os.remove(chunk_filename)  # Remove chunk file after processing

        print(f"Audio saved as {FILENAME}")
        return FILENAME

    except Exception as e:
        print(f"Exception: {e}")
        return f"Error converting text to speech: {str(e)}"
