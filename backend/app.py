from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
import uuid
import requests
import tempfile
from watermark_remover import TikTokWatermarkRemover
import cv2
from dotenv import load_dotenv
import json
import time
from queue import Queue
from threading import Thread

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Firebase config from environment variables
FIREBASE_CONFIG = {
    "apiKey": os.environ.get("FIREBASE_API_KEY", ""),
    "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", "crema-takehome.firebaseapp.com"),
    "projectId": os.environ.get("FIREBASE_PROJECT_ID", "crema-takehome"),
    "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", "crema-takehome.appspot.com"),
    "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID", "561553746659"),
    "appId": os.environ.get("FIREBASE_APP_ID", "1:561553746659:web:4907152bdd320c1ae7f3d6")
}

# Store progress updates for each task
progress_queues = {}

@app.route('/api/progress/<task_id>', methods=['GET'])
def progress(task_id):
    """SSE endpoint for progress updates"""
    def generate():
        q = Queue()
        progress_queues[task_id] = q
        try:
            while True:
                progress = q.get()
                if progress == 'DONE':
                    del progress_queues[task_id]
                    break
                yield f"data: {json.dumps(progress)}\n\n"
        except GeneratorExit:
            if task_id in progress_queues:
                del progress_queues[task_id]
    
    return Response(generate(), mimetype='text/event-stream')

def process_video_task(video_url, task_id):
    """Background task to process video and send progress updates"""
    q = progress_queues.get(task_id)
    if not q:
        return
    
    try:
        # Create temporary files for processing
        temp_input = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        # Update progress - Downloading
        q.put({"status": "downloading", "progress": 0})
        
        # Download the video
        response = requests.get(video_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(temp_input, 'wb') as f:
            for data in response.iter_content(block_size):
                downloaded += len(data)
                f.write(data)
                if total_size:
                    progress = int((downloaded / total_size) * 100)
                    q.put({"status": "downloading", "progress": progress})
        
        # Update progress - Analyzing
        q.put({"status": "analyzing", "progress": 0})
        
        # Initialize watermark remover
        remover = TikTokWatermarkRemover(debug=True)
        
        # Analyze video
        cap = cv2.VideoCapture(temp_input)
        if not cap.isOpened():
            raise Exception('Could not open video file')
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        first_section_frames = min(int(5 * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        # Analyze first section
        first_section_elements = remover._analyze_section(cap, 0, first_section_frames, "first_section")
        cap.release()
        
        if not first_section_elements or not any(key in first_section_elements for key in ['logo', 'logo_text']):
            q.put({"status": "complete", "message": "No TikTok watermarks detected", "video_url": video_url})
            q.put("DONE")
            os.unlink(temp_input)
            os.unlink(temp_output)
            return
        
        # Update progress - Processing
        q.put({"status": "processing", "progress": 0})
        
        # Process video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        def progress_callback(frame_num):
            progress = int((frame_num / total_frames) * 100)
            q.put({"status": "processing", "progress": progress})
        
        remover.process_video(temp_input, temp_output, progress_callback=progress_callback)
        
        # Update progress - Complete
        q.put({
            "status": "complete",
            "message": "Video processed successfully",
            "video_url": video_url  # Frontend will handle final upload
        })
        q.put("DONE")
        
        # Clean up
        os.unlink(temp_input)
        os.unlink(temp_output)
        
    except Exception as e:
        q.put({"status": "error", "message": str(e)})
        q.put("DONE")
        if 'temp_input' in locals() and os.path.exists(temp_input):
            os.unlink(temp_input)
        if 'temp_output' in locals() and os.path.exists(temp_output):
            os.unlink(temp_output)

@app.route('/api/process-video', methods=['POST'])
def api_process_video():
    if 'video_url' not in request.json:
        return jsonify({'error': 'No video URL provided'}), 400
    
    video_url = request.json['video_url']
    task_id = str(uuid.uuid4())
    
    # Start processing in background thread
    thread = Thread(target=process_video_task, args=(video_url, task_id))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'task_id': task_id,
        'message': 'Processing started'
    })

@app.route('/api/remove-watermark', methods=['POST'])
def remove_watermark():
    """
    Remove TikTok watermark from a video.
    
    Request body:
    {
        "video_url": "URL of the video to process"
    }
    
    Returns:
    {
        "success": true/false,
        "message": "Status message",
        "video_url": "URL of the processed video" (if successful)
    }
    """
    try:
        # Get the video URL from the request
        data = request.get_json()
        video_url = data.get('video_url')
        
        if not video_url:
            return jsonify({
                'success': False,
                'message': 'No video URL provided'
            }), 400
        
        # Create a temporary file for the input video
        temp_input = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        # Download the video from Firebase Storage URL
        print(f"Downloading video from: {video_url}")
        response = requests.get(video_url, stream=True)
        if response.status_code != 200:
            return jsonify({
                'success': False,
                'message': f'Failed to download video: {response.status_code}'
            }), 400
            
        # Save the video to the temporary file
        with open(temp_input, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Create a temporary file for the output video
        temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        # Remove the watermark
        remover = TikTokWatermarkRemover(debug=True)
        remover.process_video(temp_input, temp_output)
        
        # Clean up the input file
        os.unlink(temp_input)
        
        return jsonify({
            'success': True,
            'message': 'Watermark removed successfully',
            'video_path': temp_output
        })
    except Exception as e:
        print(f"Error in remove_watermark: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True) 