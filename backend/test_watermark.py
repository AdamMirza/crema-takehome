from watermark_detector import WatermarkDetector
import argparse
import os

def test_detection(video_path, debug=False):
    detector = WatermarkDetector(debug=debug)
    result = detector.detect_tiktok_watermark(video_path)
    
    print("\n" + "="*50)
    print("Watermark Detection Results:")
    print(f"Has TikTok watermark: {result['has_watermark']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Detection rate: {result.get('detection_rate', 0):.2%}")
    
    if result.get('match_info'):
        print(f"\nBest match details:")
        print(f"  Method: {result['match_info'].get('method', 'unknown')}")
        print(f"  Frame: {result['match_info'].get('frame', 'unknown')}")
        
        # Show template match details if available
        template_matches = result['match_info'].get('template_matches', [])
        if template_matches:
            try:
                best_template = max(
                    template_matches, 
                    key=lambda x: x.get('confidence', 0)
                )
                if best_template.get('detected', False):
                    print(f"  Best template: {best_template.get('template', 'unknown')}")
                    print(f"  Template confidence: {best_template.get('confidence', 0):.4f}")
                    print(f"  Scale: {best_template.get('scale', 'unknown')}")
            except Exception as e:
                print(f"  Error getting template details: {str(e)}")
    
    print("\nVideo Info:")
    print(f"Frame count: {result['video_info']['frame_count']}")
    print(f"FPS: {result['video_info']['fps']}")
    print(f"Duration: {result['video_info']['duration_seconds']} seconds")
    print("="*50)
    
    if debug:
        print("\nDebug information has been saved to the 'debug_frames' directory.")
        print("Check the latest timestamped folder for detailed visualizations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test TikTok watermark detection')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--debug', action='store_true',
                        help='Enable detailed debug visualizations')
    
    args = parser.parse_args()
    
    test_detection(args.video_path, args.debug) 