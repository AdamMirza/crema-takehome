from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import requests
from functools import wraps
import jwt
from datetime import datetime, timedelta
from dotenv import load_dotenv
from video_processor import process_video

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Firebase config from environment variables
FIREBASE_CONFIG = {
    "apiKey": os.environ.get("FIREBASE_API_KEY", ""),
    "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", "crema-takehome.firebaseapp.com"),
    "projectId": os.environ.get("FIREBASE_PROJECT_ID", "crema-takehome"),
    "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", "crema-takehome.firebasestorage.app"),
    "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID", "561553746659"),
    "appId": os.environ.get("FIREBASE_APP_ID", "1:561553746659:web:4907152bdd320c1ae7f3d6")
}

# Secret key for JWT
SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'your-secret-key-for-development')

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            # You can add user verification here if needed
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
        
        return f(*args, **kwargs)
    
    return decorated

@app.route('/api/login', methods=['POST'])
def login():
    # This is a simple example. In production, verify credentials against a database
    auth = request.authorization
    if auth and auth.username == 'admin' and auth.password == 'password':
        token = jwt.encode({
            'user': auth.username,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, SECRET_KEY)
        
        return jsonify({'token': token})
    
    return jsonify({'message': 'Could not verify!'}), 401

@app.route('/api/process-video', methods=['POST'])
@token_required
def api_process_video():
    if 'video_path' not in request.json:
        return jsonify({'error': 'No video path provided'}), 400
    
    video_path = request.json['video_path']
    
    try:
        # Create a temporary file to store the downloaded video
        temp_file_path = f"/tmp/{uuid.uuid4()}.mp4"
        
        # Download from Firebase Storage using REST API
        # Note: This requires the file to be publicly accessible or you need to implement authentication
        storage_url = f"https://firebasestorage.googleapis.com/v0/b/{FIREBASE_CONFIG['storageBucket']}/o/{video_path.replace('/', '%2F')}?alt=media"
        
        response = requests.get(storage_url)
        if response.status_code != 200:
            return jsonify({'error': f'Failed to download video: {response.text}'}), 500
        
        # Save the downloaded content to a temporary file
        with open(temp_file_path, 'wb') as f:
            f.write(response.content)
        
        # Process the video
        result = process_video(temp_file_path)
        
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return jsonify({'result': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 