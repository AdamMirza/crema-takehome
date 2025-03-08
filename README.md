# TikTok Watermark Remover

This project consists of two main components:
- A React Native/Expo mobile app (frontend)
- A Flask-based video processing API (backend)

## Project Structure

```
crema-takehome/
├── frontend/         # React Native/Expo mobile app
└── backend/         # Flask API for video processing
```

Each directory contains its own README with specific setup and running instructions.

## Quick Start

### Frontend
The mobile app allows users to:
1. Select videos from their device
2. Upload them to Firebase Storage
3. Process them to remove TikTok watermarks
4. View the processed result

See `frontend/README.md` for detailed setup instructions.

### Backend
The Flask API handles:
1. Video processing
2. TikTok watermark detection and removal
3. Progress tracking via Server-Sent Events

See `backend/README.md` for detailed setup instructions.

## Testing the API Directly

Since the mobile app and backend are currently separate (backend needs deployment for mobile access), you can test the watermark removal API directly:

1. Start the Flask backend:
```bash
cd backend
python app.py
```

2. Test with a sample TikTok video using curl:
```bash
curl -X POST http://localhost:5000/api/process-video \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://firebasestorage.googleapis.com/v0/b/crema-takehome.firebasestorage.app/o/originals%2F1741443029248-gwut44.mp4?alt=media&token=77297e78-7062-4c23-810b-8e6b59f2ae8d"
  }'
```

3. Monitor progress:
- The API will return a task_id
- Use this to monitor progress at: `http://localhost:5000/api/progress/<task_id>`

## Development Notes

- The frontend currently uploads videos to Firebase Storage successfully
- The backend processes videos and removes watermarks successfully
- Integration between frontend and backend requires:
  1. Deploying the backend to a publicly accessible URL, or
  2. Running the backend locally and updating the frontend API_BASE_URL to your local IP

## Future Improvements

1. Deploy backend to cloud service (e.g., Google Cloud Run, Heroku)
2. Add authentication and rate limiting
3. Implement video compression before upload
4. Add support for batch processing

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request