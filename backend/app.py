from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import requests
import tempfile
from watermark_remover import TikTokWatermarkRemover
import cv2
from dotenv import load_dotenv

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

@app.route('/api/process-video', methods=['POST'])
def api_process_video():
    if 'video_url' not in request.json:
        return jsonify({'error': 'No video URL provided'}), 400
    
    video_url = request.json['video_url']
    
    try:
        # Create temporary files for processing
        temp_input = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        # Download the video
        response = requests.get(video_url)
        if response.status_code != 200:
            return jsonify({'error': f'Failed to download video: {response.text}'}), 500
        
        with open(temp_input, 'wb') as f:
            f.write(response.content)
        
        # Initialize watermark remover in debug mode to check for TikTok watermarks
        remover = TikTokWatermarkRemover(debug=True)
        
        # First, analyze the video to detect if it's a TikTok video
        cap = cv2.VideoCapture(temp_input)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 500
            
        # Get first 5 seconds of frames
        fps = cap.get(cv2.CAP_PROP_FPS)
        first_section_frames = min(int(5 * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        # Analyze first section for TikTok watermarks
        first_section_elements = remover._analyze_section(cap, 0, first_section_frames, "first_section")
        cap.release()
        
        # If no TikTok watermarks detected, return original video
        if not first_section_elements or not any(key in first_section_elements for key in ['logo', 'logo_text']):
            os.unlink(temp_input)
            os.unlink(temp_output)
            return jsonify({
                'message': 'No TikTok watermarks detected',
                'video_url': video_url  # Return original URL
            })
        
        # Process video to remove watermarks
        print("TikTok watermarks detected, processing video...")
        remover.process_video(temp_input, temp_output)
        
        # Return the processed video URL (this will be the Firebase Storage URL from frontend)
        return jsonify({
            'message': 'Video processed successfully',
            'original_url': video_url,
            'processed_url': video_url  # Frontend will handle the Firebase upload
        })
        
    except Exception as e:
        # Clean up temporary files in case of error
        if 'temp_input' in locals() and os.path.exists(temp_input):
            os.unlink(temp_input)
        if 'temp_output' in locals() and os.path.exists(temp_output):
            os.unlink(temp_output)
        
        return jsonify({'error': str(e)}), 500

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
        
        # Download the video
        video_path = download_video(video_url)
        
        if not video_path:
            return jsonify({
                'success': False,
                'message': 'Failed to download video'
            }), 400
        
        # Generate output path
        output_filename = f"processed_{os.path.basename(video_path)}"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        
        # Remove the watermark
        remover = WatermarkRemover(inpainting_method='opencv')
        result = remover.remove_watermark(video_path, output_path)
        
        if not result['success']:
            return jsonify({
                'success': False,
                'message': result['message']
            }), 400
        
        # Generate URL for the processed video
        video_url = url_for('static', filename=f'uploads/{output_filename}', _external=True)
        
        return jsonify({
            'success': True,
            'message': 'Watermark removed successfully',
            'video_url': video_url
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 