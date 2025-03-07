from watermark_detector import WatermarkDetector

def process_video(video_path):
    """
    Process a video file.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
        dict: Results of the video processing.
    """
    # Initialize the watermark detector
    detector = WatermarkDetector()
    
    # Detect watermark
    watermark_results = detector.detect_tiktok_watermark(video_path)
    
    # Return combined results
    return {
        "status": "success",
        "message": "Video processed successfully",
        "video_path": video_path,
        "processing_details": {
            "frames_processed": watermark_results.get("video_info", {}).get("frame_count", 0),
            "duration": f"{watermark_results.get('video_info', {}).get('duration_seconds', 0):.2f} seconds",
            "resolution": "1920x1080"  # placeholder
        },
        "watermark_detection": {
            "has_tiktok_watermark": watermark_results.get("has_watermark", False),
            "detected_username": watermark_results.get("username", None)
        }
    } 