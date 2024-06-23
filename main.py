import os
import threading
import time
import webbrowser
from flask import Flask, render_template, request, jsonify, g, send_file
from Brain_modules.brain import Brain
from Brain_modules.lobes.listen_lobe import AuroraRecorder
from app_files.speaker import text_to_speech
from app_files.status_manager import StatusManager

app = Flask(__name__)

# Initialize components
brain = Brain(api_key=os.environ.get("GROQ_API_KEY"), status_update_callback=lambda x: print(f"Status: {x}"))
aurora_recorder = AuroraRecorder()
status_manager = StatusManager(None)  # We'll update this with a proper status update function

def process_transcription(transcription):
    response = brain.central_processing_agent(transcription)
    audio_file = text_to_speech(response)
    return response, audio_file

@app.before_request
def before_request():
    g.start_time = time.time()

@app.after_request
def after_request(response):
    diff = time.time() - g.start_time
    print(f"Request processed in {diff:.2f} seconds")
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    prompt = data.get('message')
    
    if not prompt:
        return jsonify({'response': 'No prompt provided.', 'status': 'Error'})
    
    print(f"Received prompt: {prompt}")  # Debugging statement

    try:
        response = brain.central_processing_agent(prompt)
        audio_file = text_to_speech(response)
        return jsonify({'response': response, 'status': 'Completed', 'audio_file': audio_file})
    except Exception as e:
        print(f"Error processing prompt: {str(e)}")  # Debugging statement
        return jsonify({'response': f"Error processing prompt: {str(e)}", 'status': 'Error'})

@app.route('/toggle_tts', methods=['POST'])
def toggle_tts():
    brain.toggle_tts()
    status = "enabled" if brain.tts_enabled else "disabled"
    return jsonify({'status': f'Text-to-Speech {status}'})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    try:
        aurora_recorder.start_recording()
        return jsonify({'status': 'Recording started'})
    except Exception as e:
        return jsonify({'status': f"Error starting recording: {str(e)}"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    try:
        aurora_recorder.stop_recording()
        transcription = aurora_recorder.transcription
        response, audio_file = process_transcription(transcription)
        return jsonify({'transcription': transcription, 'response': response, 'audio_file': audio_file})
    except Exception as e:
        return jsonify({'status': f"Error stopping recording: {str(e)}"})

@app.route('/get_audio/<filename>', methods=['GET'])
def get_audio(filename):
    return send_file(filename, mimetype="audio/mp3")

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    threading.Timer(1.25, open_browser).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
