import argparse
import os
from watermark_remover import WatermarkRemover

def test_watermark_removal(input_path, output_path, inpainting_method="opencv", debug=False):
    """
    Test the watermark removal process.
    
    Args:
        input_path (str): Path to the input video
        output_path (str): Path to save the output video
        inpainting_method (str): Inpainting method to use
        debug (bool): Whether to enable debug mode
    """
    # Create the watermark remover
    remover = WatermarkRemover(inpainting_method=inpainting_method, debug=debug)
    
    # Remove the watermark
    result = remover.remove_watermark(input_path, output_path)
    
    # Print the result
    print("\n" + "="*50)
    print("Watermark Removal Results:")
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    
    if result['success']:
        print(f"\nOutput video saved to: {result['output_path']}")
    
    # Print detection details
    detection_result = result.get('detection_result', {})
    print("\nWatermark Detection Details:")
    print(f"Has TikTok watermark: {detection_result.get('has_watermark', False)}")
    print(f"Confidence: {detection_result.get('confidence', 0):.4f}")
    print(f"Detection rate: {detection_result.get('detection_rate', 0):.2%}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test TikTok watermark removal')
    parser.add_argument('input_path', help='Path to the input video file')
    parser.add_argument('output_path', help='Path to save the output video')
    parser.add_argument('--method', choices=['opencv', 'lama', 'deepfill'], default='deepfill',
                        help='Inpainting method to use (default: deepfill)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    test_watermark_removal(args.input_path, args.output_path, args.method, args.debug) 