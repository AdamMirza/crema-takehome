import cv2
import numpy as np
import os
import tempfile
from typing import List, Tuple, Dict, Optional
import logging
import tqdm
import sys

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TikTokWatermarkRemover:
    def __init__(self, debug=False):
        """
        Initialize the TikTok watermark remover.
        
        Args:
            debug: Whether to enable debug mode
        """
        self.debug = bool(debug)
        print(f"TikTokWatermarkRemover initialized with debug={self.debug}")
        
        # Create debug directory if in debug mode
        if self.debug:
            self.debug_dir = os.path.abspath(os.path.join(os.getcwd(), "debug_frames"))
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"Debug mode enabled. Saving frames to: {self.debug_dir}")
        
        # Initialize the detector directly with template matching
        self.templates = self._load_templates()
    
    def _load_templates(self):
        """Load TikTok logo templates"""
        templates = []
        template_dir = os.path.join(os.path.dirname(__file__), "assets")
        
        if not os.path.exists(template_dir):
            print(f"Warning: Template directory not found at {template_dir}")
            return templates
        
        # Template file paths with their specific confidence thresholds
        template_files = {
            'logo': {'file': 'tiktok_logo.png', 'threshold': 0.7},
            'icon': {'file': 'tiktok_icon.png', 'threshold': 0.7},
            'logo_text': {'file': 'tiktok_logo_text.png', 'threshold': 0.5}  # Lower threshold for text
        }
        
        for template_type, config in template_files.items():
            template_path = os.path.join(template_dir, config['file'])
            if os.path.exists(template_path):
                print(f"Loading template {template_type} from: {template_path}")
                template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
                if template is not None:
                    print(f"Template {template_type} loaded with shape: {template.shape}")
                    if template.shape[2] == 4:  # Has alpha channel
                        alpha = template[:, :, 3]
                        gray = cv2.cvtColor(template[:, :, :3], cv2.COLOR_BGR2GRAY)
                        _, alpha_mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
                        gray = cv2.bitwise_and(gray, gray, mask=alpha_mask)
                    else:
                        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    templates.append({
                        'name': template_type,
                        'image': gray,
                        'size': gray.shape,
                        'threshold': config['threshold']
                    })
                    print(f"Successfully processed template: {template_type}")
                else:
                    print(f"Failed to load template: {template_type}")
            else:
                print(f"Template file not found: {template_path}")
        
        return templates

    def _detect_watermark(self, frame, section_name="first"):
        """Detect TikTok watermark in a frame using template matching"""
        if not self.templates:
            return None
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best_matches = {}
        
        for template in self.templates:
            template_img = template['image']
            template_h, template_w = template_img.shape
            
            # Different scales for different template types
            scales = [0.5, 0.75, 1.0, 1.25, 1.5] if template['name'] != 'logo_text' else \
                    [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # More scales for text
            
            for scale in scales:
                scaled_w = int(template_w * scale)
                scaled_h = int(template_h * scale)
                
                if scaled_w >= frame.shape[1] or scaled_h >= frame.shape[0]:
                    continue
                
                scaled_template = cv2.resize(template_img, (scaled_w, scaled_h))
                result = cv2.matchTemplate(gray_frame, scaled_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if self.debug:
                    print(f"Template {template['name']} at scale {scale:.2f} - confidence: {max_val:.3f}")
                
                current_best = best_matches.get(template['name'], {'confidence': 0})
                if max_val > current_best['confidence']:
                    best_matches[template['name']] = {
                        'type': template['name'],
                        'location': max_loc,
                        'width': scaled_w,
                        'height': scaled_h,
                        'confidence': max_val,
                        'scale': scale
                    }
        
        # Adjust threshold for logo_text in middle section
        threshold_adjustment = 0.1 if section_name == "middle_section" and \
                            any(m['type'] == 'logo_text' for m in best_matches.values()) else 0
        
        # Filter matches by their specific confidence thresholds
        valid_matches = {}
        for template in self.templates:
            for k, v in best_matches.items():
                if template['name'] == k:
                    adjusted_threshold = template['threshold'] - threshold_adjustment
                    if v['confidence'] > adjusted_threshold:
                        valid_matches[k] = v
        
        if self.debug and valid_matches:
            print(f"\nValid matches found in {section_name}:")
            for match_type, match_data in valid_matches.items():
                print(f"{match_type}: confidence={match_data['confidence']:.3f}, scale={match_data['scale']:.2f}")
        
        if not valid_matches:
            return None
            
        # Return the match with highest confidence
        best_match = max(valid_matches.values(), key=lambda x: x['confidence'])
        return best_match

    def _detect_text_near_logo(self, frame, logo_box):
        """Detect white text near the logo location"""
        if not logo_box:
            return None
            
        # Get logo position
        logo_x, logo_y, logo_w, logo_h = logo_box
        height, width = frame.shape[:2]
        
        # Define two search regions: below and to the right of logo
        search_regions = [
            # Below logo
            {
                'x': max(0, logo_x - int(logo_w * 0.25)),
                'y': logo_y + logo_h,
                'w': min(width - logo_x, int(logo_w * 2)),
                'h': int(logo_h * 1.5),
                'weight': 1.2  # Prefer text below logo
            },
            # Right of logo
            {
                'x': logo_x + logo_w,
                'y': max(0, logo_y - int(logo_h * 0.25)),
                'w': min(width - (logo_x + logo_w), int(logo_w * 2)),
                'h': int(logo_h * 1.5),
                'weight': 1.0  # Normal weight for text to the right
            }
        ]
        
        best_text_box = None
        best_confidence = 0
        
        for region in search_regions:
            # Skip if region is out of bounds
            if region['x'] + region['w'] > width or region['y'] + region['h'] > height:
                continue
                
            # Extract region of interest
            roi = frame[region['y']:region['y']+region['h'], 
                      region['x']:region['x']+region['w']]
            
            # Convert to HSV for better white text detection
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Define range for white color
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 30, 255])
            
            # Create mask for white regions
            mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Apply morphological operations
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=1)
            
            if self.debug:
                self._save_debug_frame(mask, f"text_mask_{region['y']}", None)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
                
            # Filter and sort contours
            valid_contours = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                aspect_ratio = w / float(h) if h > 0 else 0
                
                # Filter based on size and aspect ratio
                if area > 100 and 2 < aspect_ratio < 8:
                    # Calculate relative position score (prefer left-aligned text)
                    position_score = 1.0 - (x / float(region['w']))
                    confidence = (area / (region['w'] * region['h'])) * position_score * region['weight']
                    valid_contours.append((cnt, x, y, w, h, confidence))
            
            if not valid_contours:
                continue
            
            # Get the contour with highest confidence
            best_contour = max(valid_contours, key=lambda x: x[5])
            _, x, y, w, h, confidence = best_contour
            
            # Convert coordinates back to full frame
            text_box = [
                x + region['x'],
                y + region['y'],
                w,
                h
            ]
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_text_box = text_box
        
        if self.debug and best_text_box is not None:
            debug_frame = frame.copy()
            x, y, w, h = best_text_box
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(debug_frame, f"text_conf:{best_confidence:.2f}", (x, y-5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            self._save_debug_frame(debug_frame, "text_detection_debug", None)
        
        return best_text_box

    def _detect_username_below_text(self, frame, logo_text_box):
        """Detect TikTok username below the logo text box"""
        if not logo_text_box:
            return None
            
        # Get logo text position
        x, y, w, h = logo_text_box
        
        # Define search region below logo text (5-15px padding)
        height, width = frame.shape[:2]
        search_x = max(0, x - int(w * 0.5))  # Wider search area horizontally
        search_y = y + h + 5  # Start 5px below logo text
        search_w = min(width - search_x, int(w * 2.5))  # Much wider search area for long usernames
        search_h = 30  # Typical username height
        
        # Ensure search area is within frame bounds
        if search_y + search_h >= height:
            return None
            
        # Extract region of interest
        roi = frame[search_y:search_y+search_h, search_x:search_x+search_w]
        
        # Convert to HSV for better white text detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define range for white/light gray text
        lower_white = np.array([0, 0, 180])  # More permissive threshold
        upper_white = np.array([180, 50, 255])
        
        # Create mask for white regions
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Apply morphological operations to connect text
        kernel = np.ones((2,2), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        if self.debug:
            self._save_debug_frame(mask, f"username_mask", None)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours by size and position
        valid_contours = []
        min_width = 30  # Minimum width for "@" plus a short username
        
        for cnt in contours:
            x_rel, y_rel, w_rel, h_rel = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect_ratio = w_rel / float(h_rel) if h_rel > 0 else 0
            
            # Username should be reasonably sized and start near the left
            # Allow for wider range of aspect ratios since usernames can be any length
            if (w_rel >= min_width and 
                x_rel < w//2 and  # More permissive on starting position
                area > 50 and     # Minimum area to avoid noise
                aspect_ratio > 1.2):  # More permissive aspect ratio
                valid_contours.append((cnt, x_rel, y_rel, w_rel, h_rel))
        
        if not valid_contours:
            return None
        
        # Sort by x position (left to right) and take the leftmost one
        valid_contours.sort(key=lambda x: x[1])
        _, x_rel, y_rel, w_rel, h_rel = valid_contours[0]
        
        # Extend the width significantly to capture the full username
        extended_w = min(search_w - x_rel, int(w_rel * 2.5))  # Much wider extension
        
        # Convert coordinates back to full frame
        username_box = [
            search_x + x_rel,
            search_y + y_rel,
            extended_w,  # Use extended width
            h_rel
        ]
        
        if self.debug:
            debug_frame = frame.copy()
            x, y, w, h = username_box
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 165, 0), 2)
            cv2.putText(debug_frame, "username_debug", (x, y-5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            self._save_debug_frame(debug_frame, "username_detection_debug", None)
        
        return username_box

    def _analyze_section(self, cap, start_frame: int, num_frames: int, section_name: str) -> Dict:
        """Analyze a section of frames for watermark detection."""
        frames_to_analyze = min(20, num_frames)  # Increased from 10 to 20
        frame_step = max(1, num_frames // frames_to_analyze)
        
        if self.debug:
            print(f"\nAnalyzing {section_name}:")
            print(f"Starting at frame {start_frame}")
            print(f"Analyzing {frames_to_analyze} frames with step {frame_step}")
        
        detections = []
        best_detection = None
        best_confidence = 0
        best_frame = None
        
        for i in range(frames_to_analyze):
            frame_idx = start_frame + (i * frame_step)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Detect watermarks with section name
            match = self._detect_watermark(frame, section_name)
            if match:
                detection = {
                    match['type']: {
                        'box': [
                            match['location'][0],
                            match['location'][1],
                            match['width'],
                            match['height']
                        ],
                        'confidence': match['confidence']
                    }
                }
                
                # If we found logo_text, look for username
                if match['type'] == 'logo_text':
                    # Look for username below logo text
                    username_box = self._detect_username_below_text(frame, detection['logo_text']['box'])
                    if username_box:
                        detection['username'] = {
                            'box': username_box,
                            'confidence': 1.0
                        }
                    
                    # Update best detection if this has higher confidence
                    if match['confidence'] > best_confidence:
                        best_confidence = match['confidence']
                        best_detection = detection.copy()
                        best_frame = frame.copy()
                
                detections.append(detection)
                
                if self.debug:
                    debug_frame = frame.copy()
                    # Draw detection boxes with different colors
                    for element_type, element_data in detection.items():
                        x, y, w, h = element_data['box']
                        color = {
                            'logo': (0, 255, 0),      # Green for logo
                            'icon': (0, 255, 255),    # Yellow for icon
                            'logo_text': (255, 0, 0),  # Blue for logo text
                            'username': (255, 165, 0)  # Orange for username
                        }.get(element_type, (0, 255, 0))
                        
                        cv2.rectangle(debug_frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(debug_frame, f"{element_type} ({element_data['confidence']:.2f})", 
                                  (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    self._save_debug_frame(debug_frame, f"{section_name}_detection", i)
        
        if not detections:
            if self.debug:
                print(f"No detections found in {section_name}")
            return {}
        
        # Save best frame with all detections
        if self.debug and best_frame is not None and best_detection is not None:
            debug_frame = best_frame.copy()
            for element_type, element_data in best_detection.items():
                x, y, w, h = element_data['box']
                color = {
                    'logo': (0, 255, 0),
                    'icon': (0, 255, 255),
                    'logo_text': (255, 0, 0),
                    'username': (255, 165, 0)
                }.get(element_type, (0, 255, 0))
                
                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(debug_frame, f"{element_type} ({element_data['confidence']:.2f})", 
                          (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            self._save_debug_frame(debug_frame, f"{section_name}_best_detection", None)
        
        # Combine all unique detections with highest confidence
        combined_detection = {}
        for detection in detections:
            for element_type, element_data in detection.items():
                if element_type not in combined_detection or \
                   element_data['confidence'] > combined_detection[element_type]['confidence']:
                    combined_detection[element_type] = element_data
        
        # Add the best detection and its username if found
        if best_detection:
            for element_type, element_data in best_detection.items():
                combined_detection[element_type] = element_data
        
        if self.debug:
            print(f"\nFinal detections for {section_name}:")
            for element_type, element_data in combined_detection.items():
                print(f"{element_type}: confidence={element_data['confidence']:.3f}")
        
        # Convert to the expected format
        result = {}
        for element_type, element_data in combined_detection.items():
            result[element_type] = element_data['box']
        
        return result

    def process_video(self, input_path: str, output_path: str) -> str:
        """Process a TikTok video to remove watermarks."""
        print(f"Processing video: {input_path}")
        print(f"Debug mode is {'ENABLED' if self.debug else 'DISABLED'}")
        
        # Open the video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Calculate frame ranges
        first_section_frames = min(int(5 * fps), total_frames)  # Exactly 5 seconds or whole video if shorter
        middle_section_start = first_section_frames  # Start middle section right after first section
        middle_section_end = total_frames - int(fps)  # End 1 second before video ends
        
        if self.debug:
            print(f"Frame ranges:")
            print(f"First section: 0 to {first_section_frames} ({first_section_frames/fps:.2f} seconds)")
            print(f"Middle section: {middle_section_start} to {middle_section_end} ({(middle_section_end-middle_section_start)/fps:.2f} seconds)")
        
        # Analyze sections using the watermark detector
        print("Analyzing first 5 seconds of the video...")
        first_section_elements = self._analyze_section(cap, 0, first_section_frames, "first_section")
        
        print("Analyzing middle section of the video...")
        middle_section_elements = self._analyze_section(
            cap, 
            middle_section_start,
            middle_section_end - middle_section_start,
            "middle_section"
        )
        
        # Process frames
        return self._process_frames(
            input_path, output_path,
            first_section_elements,
            middle_section_elements,
            first_section_frames,
            middle_section_end,
            total_frames
        )
    
    def _process_frames(self, input_path: str, output_path: str,
                       first_section_boxes: Dict, middle_section_boxes: Dict,
                       first_section_frames: int, last_section_start: int,
                       total_frames: int) -> str:
        """Process video frames using detected boxes."""
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 
                            cap.get(cv2.CAP_PROP_FPS),
                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        frame_count = 0
        with tqdm.tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Use appropriate boxes based on frame position
                if frame_count < first_section_frames:
                    boxes = first_section_boxes
                    section = "first"
                else:
                    boxes = middle_section_boxes
                    section = "middle"
                
                # Process frame
                processed = self._remove_watermarks_from_frame(frame, boxes)
                
                # Save debug frame occasionally
                if self.debug and frame_count % 300 == 0:
                    self._save_debug_frame(processed, f"{section}_processed", frame_count)
                
                out.write(processed)
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        out.release()
        return output_path
    
    def _remove_watermarks_from_frame(self, frame: np.ndarray, boxes: Dict) -> np.ndarray:
        """Remove watermarks from a frame using the detected boxes."""
        if not boxes:
            return frame
        
        result = frame.copy()
        
        # Process each detected element
        for element_type, box in boxes.items():
            x, y, w, h = box
            
            # Create mask for inpainting
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
            
            # Add padding around the box for better inpainting
            pad = 20
            roi_x = max(0, x - pad)
            roi_y = max(0, y - pad)
            roi_w = min(frame.shape[1] - roi_x, w + 2*pad)
            roi_h = min(frame.shape[0] - roi_y, h + 2*pad)
            
            # Extract ROI
            roi = result[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            roi_mask = mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # Perform inpainting
            inpainted = cv2.inpaint(roi, roi_mask, 3, cv2.INPAINT_NS)
            result[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = inpainted
        
        return result
    
    def _save_debug_frame(self, frame, name, frame_number=None):
        """Save a debug frame."""
        if not self.debug:
            return
        
        filename = f"{name}_frame_{frame_number}.jpg" if frame_number is not None else f"{name}.jpg"
        filepath = os.path.join(self.debug_dir, filename)
        
        try:
            cv2.imwrite(filepath, frame)
            print(f"Saved debug frame: {filepath}")
        except Exception as e:
            print(f"Error saving debug frame: {e}")

def remove_tiktok_watermarks(input_path: str, output_path: str, debug=False) -> str:
    """Remove TikTok watermarks from a video."""
    remover = TikTokWatermarkRemover(debug=debug)
    return remover.process_video(input_path, output_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove TikTok watermarks from videos")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    print(f"Debug flag: {args.debug}")
    remove_tiktok_watermarks(args.input, None, args.debug)
