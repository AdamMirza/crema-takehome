import argparse
import os
import sys
from watermark_remover import remove_tiktok_watermarks
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_watermark_removal(input_path, output_path, debug=False):
    """
    Test the TikTok watermark removal process.
    
    Args:
        input_path (str): Path to the input video
        output_path (str): Path to save the output video
        debug (bool): Whether to enable debug mode
    """
    # Set default output path if not provided
    if not output_path:
        filename, ext = os.path.splitext(input_path)
        output_path = f"{filename}_processed{ext}"
    
    try:
        # Process the video using our TikTok watermark remover
        logger.info(f"Processing video: {input_path}")
        logger.info(f"Output will be saved to: {output_path}")
        
        # IMPORTANT: Print debug status
        print(f"Debug mode is {'ENABLED' if debug else 'DISABLED'}")
        
        # Set debug level if requested
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("Debug mode enabled")
            
            # Create debug directory explicitly
            debug_dir = os.path.join(os.getcwd(), "debug_frames")
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir, exist_ok=True)
                print(f"Created debug directory at: {debug_dir}")
            else:
                print(f"Using existing debug directory at: {debug_dir}")
            
            # Test if we can write to the directory
            test_file = os.path.join(debug_dir, "test_from_script.txt")
            with open(test_file, 'w') as f:
                f.write("Test file to check write permissions from script")
            if os.path.exists(test_file):
                print(f"Successfully created test file at {test_file}")
            
        # Remove the watermark with debug mode if requested
        # EXPLICITLY pass the debug parameter
        print(f"Calling remove_tiktok_watermarks with debug={debug}")
        output_path = remove_tiktok_watermarks(input_path, output_path, debug=debug)
        
        # Print the result
        print("\n" + "="*50)
        print("TikTok Watermark Removal Results:")
        print(f"Success: True")
        print(f"Output video saved to: {output_path}")
        if debug:
            debug_dir = os.path.join(os.getcwd(), "debug_frames")
            print(f"Debug frames should be saved to: {debug_dir}")
            # List files in debug directory
            if os.path.exists(debug_dir):
                files = os.listdir(debug_dir)
                print(f"Files in debug directory: {len(files)}")
                for file in files[:10]:  # Show first 10 files
                    print(f"  - {file}")
                if len(files) > 10:
                    print(f"  ... and {len(files) - 10} more")
            else:
                print(f"Debug directory does not exist: {debug_dir}")
        print("="*50)
        
        return {
            'success': True,
            'message': 'Watermark removal completed successfully',
            'output_path': output_path
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        print("\n" + "="*50)
        print("TikTok Watermark Removal Results:")
        print(f"Success: False")
        print(f"Error: {str(e)}")
        print("="*50)
        
        return {
            'success': False,
            'message': f'Error: {str(e)}'
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test TikTok watermark removal')
    parser.add_argument('input_path', help='Path to the input video file')
    parser.add_argument('--output_path', help='Path to save the output video (default: input_processed.mp4)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode to save frames with bounding boxes')
    
    args = parser.parse_args()
    
    # IMPORTANT: Print the debug flag value
    print(f"Debug flag from command line: {args.debug}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    
    test_watermark_removal(args.input_path, args.output_path, args.debug) 