# Video Processing API

This is a secure API for processing videos stored in Firebase Storage.

## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   Create a `.env` file in the backend directory with:
   ```
   # JWT Secret
   JWT_SECRET_KEY=your-secure-secret-key
   
   # Firebase Configuration
   FIREBASE_API_KEY=your-firebase-api-key
   FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
   FIREBASE_PROJECT_ID=your-project-id
   FIREBASE_STORAGE_BUCKET=your-project.firebasestorage.app
   FIREBASE_MESSAGING_SENDER_ID=your-messaging-sender-id
   FIREBASE_APP_ID=your-app-id
   ```

4. Run the API:
   ```
   python app.py
   ```

## API Endpoints

### Video Processing

**POST /api/process-video**
- Headers: 
  - Authorization: Bearer {token}
- Body:
  ```json
  {
    "video_url": "https://firebasestorage.googleapis.com/v0/b/your-project.appspot.com/o/videos%2Fsample.mp4?alt=media&token=your-token"
  }
  ```
- Returns processing results
   ```json
   {
      "message": "Video processed successfully",
      "original_url": "https://example.com/video.mp4",
      "processed_url": "https://storage.googleapis.com/crema-takehome.appspot.com/processed_videos/..."
   }
   ```

   ```json
   {
      "message": "No TikTok watermarks detected",
      "video_url": "https://example.com/video.mp4"
   }
   ```

## Features

### TikTok Watermark Detection

The API includes functionality to detect TikTok watermarks in videos. This feature:

- Analyzes video frames to identify TikTok logo watermarks
- Returns whether a watermark was detected
- Will eventually extract usernames from watermarks (placeholder for now)

To use this feature, you need to:

1. Create an `assets` directory in the backend folder
2. Add a cropped image of the TikTok logo named `tiktok_logo.png` to the assets directory

## Deployment

For production, consider deploying to:
- Google Cloud Run
- AWS Lambda
- Heroku 