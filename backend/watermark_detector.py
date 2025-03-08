import cv2
import numpy as np
import os
from debug_visualizer import DebugVisualizer

class WatermarkDetector:
    def __init__(self, debug=False):
        # Path to assets directory
        self.assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        
        # Debug mode
        self.debug = debug
        if debug:
            self.visualizer = DebugVisualizer()
        
        # Load TikTok logo components
        self.load_templates()
    
    def load_templates(self):
        """Load all template images for detection"""
        self.templates = []
        
        # Try to load the main logo
        main_logo_path = os.path.join(self.assets_dir, "tiktok_logo.png")
        if os.path.exists(main_logo_path):
            self.load_template(main_logo_path, "main_logo")
        
        # Try to load the TikTok icon (musical note)
        icon_path = os.path.join(self.assets_dir, "tiktok_icon.png")
        if os.path.exists(icon_path):
            self.load_template(icon_path, "icon")
        
        # Try to load the TikTok text
        text_path = os.path.join(self.assets_dir, "tiktok_text.png")
        if os.path.exists(text_path):
            self.load_template(text_path, "text")
        
        if not self.templates:
            print("Warning: No template images found. Detection may not work properly.")
            # Create a placeholder template
            placeholder = np.zeros((30, 30), dtype=np.uint8)
            self.templates.append({
                "name": "placeholder",
                "image": placeholder,
                "type": "grayscale"
            })
    
    def load_template(self, path, name):
        """Load a single template image with proper handling of transparency"""
        print(f"Loading template: {name} from {path}")
        
        # Read the image with transparency support
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            print(f"Warning: Could not load template {name} from {path}")
            return
        
        # Handle transparent PNG
        if img.shape[2] == 4:  # Has alpha channel
            # Extract alpha channel
            alpha = img[:, :, 3]
            # Convert to grayscale using just the RGB channels
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            # Apply alpha mask
            _, alpha_mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
            gray = cv2.bitwise_and(gray, gray, mask=alpha_mask)
            
            self.templates.append({
                "name": name,
                "image": gray,
                "type": "grayscale"
            })
        else:
            # Regular grayscale conversion if no transparency
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.templates.append({
                "name": name,
                "image": gray,
                "type": "grayscale"
            })

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
        
        if self.debug:
            self.visualizer.log(f"Video info: {frame_count} frames, {fps} FPS, {duration:.2f} seconds")
        else:
            print(f"Video info: {frame_count} frames, {fps} FPS, {duration:.2f} seconds")
        
        # We'll check frames at regular intervals
        frames_to_check = min(30, frame_count)  # Check up to 30 frames
        interval = max(1, frame_count // frames_to_check)
        
        # Also check specific time points where the logo might move
        # TikTok logo often moves around 5 seconds in
        specific_time_points = [5, 10, 15]  # seconds
        specific_frames = [int(t * fps) for t in specific_time_points if t * fps < frame_count]
        
        if self.debug:
            self.visualizer.log(f"Checking {frames_to_check} frames at intervals of {interval} frames")
            self.visualizer.log(f"Also checking specific frames at: {specific_frames}")
        else:
            print(f"Checking {frames_to_check} frames at intervals of {interval} frames")
            print(f"Also checking specific frames at: {specific_frames}")
        
        # Track detection results
        detection_results = []
        best_confidence = 0
        best_match_info = None
        positive_frames = 0  # Count frames with positive detection
        
        # First, check the specific time points
        for frame_idx in specific_frames:
            if self.debug:
                self.visualizer.log(f"Checking specific frame {frame_idx}/{frame_count}")
            else:
                print(f"Checking specific frame {frame_idx}/{frame_count}")
            
            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Save frame for debugging
            if self.debug:
                self.visualizer.save_frame(frame, f"specific", frame_idx)
            
            # Check for watermark in this frame
            result = self._analyze_frame(frame, frame_idx)
            detection_results.append(result)
            
            if result["detected"]:
                positive_frames += 1
            
            if result["confidence"] > best_confidence:
                best_confidence = result["confidence"]
                best_match_info = result
        
        # Then check regular intervals
        for i in range(0, frame_count, interval):
            # Skip frames we've already checked
            if i in specific_frames:
                continue
            
            if self.debug:
                self.visualizer.log(f"Checking frame {i}/{frame_count}")
            else:
                print(f"Checking frame {i}/{frame_count}")
            
            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                if self.debug:
                    self.visualizer.log(f"Error reading frame {i}")
                else:
                    print(f"Error reading frame {i}")
                break
            
            # Save frame for debugging
            if self.debug:
                self.visualizer.save_frame(frame, f"original", i)
            
            # Check for watermark in this frame
            result = self._analyze_frame(frame, i)
            detection_results.append(result)
            
            if result["detected"]:
                positive_frames += 1
            
            if result["confidence"] > best_confidence:
                best_confidence = result["confidence"]
                best_match_info = result
            
            # If we have a very confident match, we can stop early
            if best_confidence > 0.8:
                if self.debug:
                    self.visualizer.log(f"Found high-confidence match ({best_confidence:.4f}), stopping early")
                else:
                    print(f"Found high-confidence match ({best_confidence:.4f}), stopping early")
                break
        
        # Release the video capture
        cap.release()
        
        # Calculate percentage of frames with positive detection
        detection_percentage = positive_frames / len(detection_results) if detection_results else 0
        
        if self.debug:
            self.visualizer.log(f"Positive detections: {positive_frames}/{len(detection_results)} frames ({detection_percentage:.2%})")
        else:
            print(f"Positive detections: {positive_frames}/{len(detection_results)} frames ({detection_percentage:.2%})")
        
        # Determine if a watermark was detected based on all frames
        has_watermark = False
        
        # More nuanced decision criteria:
        # 1. Very high confidence match (>0.85) is enough
        if best_confidence >= 0.85:
            has_watermark = True
        # 2. High confidence (>0.75) with decent detection rate (>15%)
        elif best_confidence >= 0.75 and detection_percentage >= 0.15:
            has_watermark = True
        # 3. Moderate confidence (>0.7) with good detection rate (>25%)
        elif best_confidence >= 0.7 and detection_percentage >= 0.25:
            has_watermark = True
        # 4. Multiple consistent detections with moderate confidence
        elif positive_frames >= 3 and best_confidence >= 0.7:
            has_watermark = True
        
        if self.debug:
            self.visualizer.log(f"Final decision - Has watermark: {has_watermark} (Confidence: {best_confidence:.4f}, Detection rate: {detection_percentage:.2%})")
        else:
            print(f"Final decision - Has watermark: {has_watermark} (Confidence: {best_confidence:.4f}, Detection rate: {detection_percentage:.2%})")
        
        final_result = {
            "has_watermark": has_watermark,
            "confidence": best_confidence,
            "detection_rate": detection_percentage,
            "match_info": best_match_info,
            "video_info": {
                "frame_count": frame_count,
                "fps": fps,
                "duration_seconds": duration
            }
        }
        
        # Create summary visualizations if in debug mode
        if self.debug:
            self.visualizer.create_summary(detection_results, final_result)
        
        return final_result
    
    def _analyze_frame(self, frame, frame_number):
        """Analyze a single frame for TikTok watermark using template matching only"""
        results = {
            "frame": frame_number,
            "template_matches": [],
            "detected": False,
            "confidence": 0,
            "method": None
        }
        
        try:
            # Try template matching with all templates
            for template in self.templates:
                match_result = self._template_match(frame, template)
                
                # Validate the match to filter out false positives
                if match_result["detected"]:
                    match_result = self._validate_match(frame, match_result)
                    match_result["frame"] = frame_number  # Add frame number for debugging
                
                results["template_matches"].append(match_result)
                
                # If we found a good match, mark as detected
                if match_result["detected"]:
                    results["detected"] = True
                    results["confidence"] = max(results["confidence"], match_result["confidence"])
                    results["method"] = f"template_{template['name']}"
                
                # Visualize template matching if in debug mode
                if self.debug:
                    self.visualizer.visualize_template_match(frame, template, match_result, frame_number)
            
            # Visualize overall frame analysis if in debug mode
            if self.debug:
                self.visualizer.visualize_frame_analysis(frame, results, frame_number)
        except Exception as e:
            if self.debug:
                self.visualizer.log(f"Error analyzing frame {frame_number}: {str(e)}")
        
        return results
    
    def _template_match(self, frame, template):
        """Match a template against a frame with multi-scale support and preprocessing"""
        result = {
            "template": template["name"],
            "detected": False,
            "confidence": 0,
            "location": None,
            "scale": None,
            "method": f"template_{template['name']}"
        }
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing to enhance contrast
        # This helps with detecting watermarks against complex backgrounds
        preprocessed_frames = [
            gray_frame,  # Original grayscale
            cv2.equalizeHist(gray_frame),  # Histogram equalization
            cv2.GaussianBlur(gray_frame, (3, 3), 0)  # Slight blur to reduce noise
        ]
        
        # Get frame dimensions for debugging
        height, width = gray_frame.shape
        
        # Check if logo template exists and has valid dimensions
        if template["image"] is None or template["image"].shape[0] == 0 or template["image"].shape[1] == 0:
            if self.debug:
                self.visualizer.log(f"Warning: Invalid template {template['name']}")
            return result
        
        logo_height, logo_width = template["image"].shape
        
        # Make sure logo is smaller than frame
        if logo_height >= height or logo_width >= width:
            if self.debug:
                self.visualizer.log(f"Warning: Template {template['name']} is larger than frame")
            return result
        
        # Try different scales for the template
        # Add more scales for finer granularity
        scales = [0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5]
        best_match_val = 0
        best_match_loc = None
        best_scale = None
        best_preprocess = None
        
        try:
            for preprocess_idx, processed_frame in enumerate(preprocessed_frames):
                preprocess_name = ["original", "equalized", "blurred"][preprocess_idx]
                
                for scale in scales:
                    # Skip if scaled logo would be larger than frame
                    scaled_width = int(logo_width * scale)
                    scaled_height = int(logo_height * scale)
                    
                    if scaled_width >= width or scaled_height >= height:
                        continue
                        
                    # Resize the logo template
                    if scale != 1.0:
                        scaled_logo = cv2.resize(template["image"], (scaled_width, scaled_height))
                    else:
                        scaled_logo = template["image"]
                    
                    # Match template
                    res = cv2.matchTemplate(processed_frame, scaled_logo, cv2.TM_CCOEFF_NORMED)
                    threshold = 0.7  # Slightly lower threshold to catch more potential matches
                    
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    
                    if self.debug and max_val > 0.5:  # Only log significant matches
                        self.visualizer.log(f"Preprocess: {preprocess_name}, Scale {scale}: Match confidence: {max_val:.4f} at location {max_loc}")
                    
                    if max_val > best_match_val:
                        best_match_val = max_val
                        best_match_loc = max_loc
                        best_scale = scale
                        best_preprocess = preprocess_name
            
            # Determine if we have a match
            result["confidence"] = best_match_val
            
            if best_match_val >= threshold:
                result["detected"] = True
                result["location"] = best_match_loc
                result["scale"] = best_scale
                
                if self.debug:
                    self.visualizer.log(f"Template {template['name']} matched with confidence {best_match_val:.4f} at scale {best_scale} using {best_preprocess} preprocessing")
            
            return result
        except Exception as e:
            if self.debug:
                self.visualizer.log(f"Error in template matching: {str(e)}")
            return result

    def _validate_match(self, frame, match_result):
        """Validate a template match to filter out false positives"""
        if not match_result["detected"] or match_result["location"] is None:
            return match_result
        
        # Get the location and scale of the matched template
        x, y = match_result["location"]
        scale = match_result["scale"] or 1.0
        template_name = match_result["template"]
        
        # Find the template by name
        template_idx = next((i for i, t in enumerate(self.templates) if t["name"] == template_name), None)
        if template_idx is None:
            return match_result
        
        template = self.templates[template_idx]
        logo_height, logo_width = template["image"].shape
        scaled_height = int(logo_height * scale)
        scaled_width = int(logo_width * scale)
        
        # Extract the region where the match was found
        roi_x = max(0, x)
        roi_y = max(0, y)
        roi_width = min(frame.shape[1] - roi_x, scaled_width)
        roi_height = min(frame.shape[0] - roi_y, scaled_height)
        
        # Make sure we have a valid ROI
        if roi_width <= 0 or roi_height <= 0:
            match_result["detected"] = False
            match_result["confidence"] *= 0.5
            if self.debug:
                self.visualizer.log(f"Match rejected: Too dark (V={avg_v:.1f})")
        
        # Extract the ROI
        roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        
        # Convert to HSV for better color analysis
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate average saturation and value
        avg_s = np.mean(hsv_roi[:, :, 1])
        avg_v = np.mean(hsv_roi[:, :, 2])
        
        # TikTok watermarks typically have moderate to high brightness
        # and low to moderate saturation
        if avg_v < 50:  # Very dark region
            match_result["detected"] = False
            match_result["confidence"] *= 0.7
            if self.debug:
                self.visualizer.log(f"Match rejected: Too dark (V={avg_v:.1f})")
        
        # Save ROI for debugging
        if self.debug:
            self.visualizer.save_frame(roi, f"validation_roi", match_result.get("frame", 0))
        
        return match_result

    def detect_tiktok_watermark_in_frame(self, frame):
        """
        Detect TikTok watermark in a single frame.
        
        Args:
            frame: The frame to analyze
            
        Returns:
            dict: Results of the watermark detection for this frame
        """
        return self._analyze_frame(frame, 0) 