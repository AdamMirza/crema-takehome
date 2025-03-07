import cv2
import numpy as np
import os

class WatermarkDetector:
    def __init__(self, use_alternative_detection=False):
        # Path to the TikTok logo template
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "tiktok_logo.png")
        
        # Flag to enable/disable alternative detection
        self.use_alternative_detection = use_alternative_detection
        
        # Check if the logo file exists
        if os.path.exists(logo_path):
            # Read the logo with transparency support
            logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            
            # Handle transparent PNG
            if logo is not None and logo.shape[2] == 4:  # Has alpha channel
                # Extract alpha channel
                alpha = logo[:, :, 3]
                # Convert to grayscale using just the RGB channels
                self.tiktok_logo = cv2.cvtColor(logo[:, :, :3], cv2.COLOR_BGR2GRAY)
                # Apply alpha mask
                _, alpha_mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
                self.tiktok_logo = cv2.bitwise_and(self.tiktok_logo, self.tiktok_logo, mask=alpha_mask)
            else:
                # Regular grayscale conversion if no transparency
                self.tiktok_logo = cv2.imread(logo_path, 0)
            
            print(f"Loaded logo template from {logo_path}")
        else:
            print(f"Warning: TikTok logo template not found at {logo_path}")
            # Create a simple placeholder logo for testing
            self.tiktok_logo = np.zeros((30, 30), dtype=np.uint8)
            print("Using placeholder logo instead")

    def detect_tiktok_watermark(self, video_path):
        """
        Detect if a video has a TikTok watermark.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            dict: Results of the watermark detection
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return {
                "has_watermark": False,
                "error": "Could not open video file"
            }
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"Video info: {frame_count} frames, {fps} FPS, {duration:.2f} seconds")
        
        # We'll check more frames for better detection
        frames_to_check = min(30, frame_count)  # Check up to 30 frames
        interval = max(1, frame_count // frames_to_check)
        
        print(f"Checking {frames_to_check} frames at intervals of {interval} frames")
        
        watermark_detected = False
        username = None
        
        for i in range(0, frame_count, interval):
            print(f"Checking frame {i}/{frame_count}")
            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Error reading frame {i}")
                break
                
            # Check for watermark in this frame
            detected, frame_username = self._check_frame_for_watermark(frame)
            
            if detected:
                watermark_detected = True
                username = frame_username
                print(f"Watermark detected in frame {i}")
                break
            
            if i % 5 == 0:  # Save every 5th frame we check
                debug_dir = os.path.join(os.path.dirname(__file__), "debug_frames")
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"frame_{i}.jpg"), frame)
        
        # Release the video capture
        cap.release()
        
        return {
            "has_watermark": watermark_detected,
            "username": username,
            "video_info": {
                "frame_count": frame_count,
                "fps": fps,
                "duration_seconds": duration
            }
        }
    
    def _check_frame_for_watermark(self, frame):
        """Check a single frame for TikTok watermark"""
        # Try template matching first
        detected, username = self._template_matching(frame)
        
        # If template matching didn't find anything and alternative detection is enabled, try it
        if not detected and self.use_alternative_detection:
            detected, username = self._alternative_watermark_detection(frame)
        
        return detected, username

    def _template_matching(self, frame):
        """Use template matching to find TikTok logo with multi-scale support"""
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get frame dimensions for debugging
        height, width = gray_frame.shape
        print(f"Frame dimensions: {width}x{height}")
        
        # Check if logo template exists and has valid dimensions
        if self.tiktok_logo is None or self.tiktok_logo.shape[0] == 0 or self.tiktok_logo.shape[1] == 0:
            print("Warning: Invalid logo template")
            return False, None
        
        logo_height, logo_width = self.tiktok_logo.shape
        print(f"Logo template dimensions: {logo_width}x{logo_height}")
        
        # Make sure logo is smaller than frame
        if logo_height >= height or logo_width >= width:
            print("Warning: Logo template is larger than frame")
            return False, None
        
        # Try different scales for the template
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        best_match_val = 0
        detected = False
        
        try:
            for scale in scales:
                # Skip if scaled logo would be larger than frame
                scaled_width = int(logo_width * scale)
                scaled_height = int(logo_height * scale)
                
                if scaled_width >= width or scaled_height >= height:
                    continue
                    
                # Resize the logo template
                if scale != 1.0:
                    scaled_logo = cv2.resize(self.tiktok_logo, (scaled_width, scaled_height))
                else:
                    scaled_logo = self.tiktok_logo
                
                # Match template
                res = cv2.matchTemplate(gray_frame, scaled_logo, cv2.TM_CCOEFF_NORMED)
                threshold = 0.65  # Raise threshold for better precision
                
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                print(f"Scale {scale}: Best match confidence: {max_val:.4f} at location {max_loc}")
                
                if max_val > best_match_val:
                    best_match_val = max_val
                
                # Check if we have matches above threshold
                loc = np.where(res >= threshold)
                if len(loc[0]) > 0:
                    detected = True
                    print(f"Detected {len(loc[0])} matches at scale {scale} (confidence: {max_val:.4f})")
                    
                    # Save a debug image showing the match
                    debug_dir = os.path.join(os.path.dirname(__file__), "debug_frames")
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    # Draw rectangle around the best match
                    debug_frame = frame.copy()
                    top_left = max_loc
                    bottom_right = (top_left[0] + scaled_width, top_left[1] + scaled_height)
                    cv2.rectangle(debug_frame, top_left, bottom_right, (0, 255, 0), 2)
                    
                    cv2.imwrite(os.path.join(debug_dir, f"match_scale_{scale}.jpg"), debug_frame)
                    break
            
            print(f"Best overall match confidence: {best_match_val:.4f}")
            
            username = None
            if detected:
                username = "tiktok_user"  # Placeholder
            
            return detected, username
        except Exception as e:
            print(f"Error in template matching: {str(e)}")
            return False, None

    def _alternative_watermark_detection(self, frame):
        """Alternative method to detect TikTok watermarks based on image characteristics"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # TikTok watermarks are usually white text on a semi-transparent background
        # Let's look for bright regions in the bottom right corner
        height, width = frame.shape[:2]
        
        # Define region of interest (bottom right corner)
        roi_x = int(width * 0.7)
        roi_y = int(height * 0.7)
        roi = frame[roi_y:height, roi_x:width]
        
        # Save ROI for debugging
        debug_dir = os.path.join(os.path.dirname(__file__), "debug_frames")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "roi.jpg"), roi)
        
        # Convert ROI to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to find bright regions
        _, thresh = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
        
        # Save thresholded image for debugging
        cv2.imwrite(os.path.join(debug_dir, "roi_thresh.jpg"), thresh)
        
        # Count white pixels
        white_pixel_count = np.sum(thresh == 255)
        white_pixel_percentage = white_pixel_count / (thresh.shape[0] * thresh.shape[1])
        
        print(f"White pixel percentage in ROI: {white_pixel_percentage:.4f}")
        
        # TikTok watermarks typically have a very specific pattern
        # We need to be more strict with our detection
        
        # More strict criteria for TikTok watermark:
        # 1. White pixel percentage should be in a specific range
        # 2. The white pixels should form a specific pattern (e.g., text-like)
        
        # For now, let's just make the threshold range more strict
        if 0.08 < white_pixel_percentage < 0.25:
            # Additional check: Look for text-like patterns
            # This is a simple check for horizontal alignment of white pixels
            
            # Get horizontal projection (sum of white pixels in each row)
            h_projection = np.sum(thresh == 255, axis=1)
            
            # Count rows with significant white pixels
            significant_rows = np.sum(h_projection > (thresh.shape[1] * 0.1))
            
            # TikTok watermark text typically spans multiple rows
            if significant_rows >= 3:
                print("Potential watermark detected using alternative method")
                print(f"Significant rows: {significant_rows}")
                return True, "tiktok_user"
            else:
                print(f"Not enough significant rows: {significant_rows}")
        
        return False, None 