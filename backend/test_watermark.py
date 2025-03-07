from watermark_detector import WatermarkDetector
import sys
import argparse

def test_detection(video_path, use_alternative=False):
    detector = WatermarkDetector(use_alternative_detection=use_alternative)
    result = detector.detect_tiktok_watermark(video_path)
    
    print("\n" + "="*50)
    print("Watermark Detection Results:")
    print(f"Has TikTok watermark: {result['has_watermark']}")
    if result['has_watermark']:
        print(f"Username: {result['username']}")
    
    print("\nVideo Info:")
    print(f"Frame count: {result['video_info']['frame_count']}")
    print(f"FPS: {result['video_info']['fps']}")
    print(f"Duration: {result['video_info']['duration_seconds']} seconds")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test TikTok watermark detection')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--alternative', action='store_true', 
                        help='Enable alternative detection method')
    
    args = parser.parse_args()
    
    test_detection(args.video_path, args.alternative) 