import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

class DebugVisualizer:
    def __init__(self, base_dir="debug_frames"):
        """Initialize the debug visualizer"""
        self.base_dir = base_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_dir = os.path.join(base_dir, self.session_id)
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Create a log file
        self.log_path = os.path.join(self.debug_dir, "detection_log.txt")
        with open(self.log_path, "w") as f:
            f.write(f"TikTok Watermark Detection Debug Log - {datetime.now()}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message):
        """Add a message to the log file"""
        with open(self.log_path, "a") as f:
            f.write(f"{message}\n")
        print(message)
    
    def save_frame(self, frame, name, frame_number=None):
        """Save a frame for debugging"""
        if frame_number is not None:
            filename = f"{name}_frame_{frame_number}.jpg"
        else:
            filename = f"{name}.jpg"
        
        path = os.path.join(self.debug_dir, filename)
        cv2.imwrite(path, frame)
        return path
    
    def visualize_template_match(self, frame, template, match_result, frame_number):
        """Visualize template matching results"""
        # Create a copy of the frame for drawing
        vis_frame = frame.copy()
        
        # Draw the match location if detected
        if match_result["detected"] and match_result["location"] is not None:
            x, y = match_result["location"]
            scale = match_result["scale"] or 1.0
            
            # Calculate template dimensions
            template_height, template_width = template["image"].shape
            scaled_width = int(template_width * scale)
            scaled_height = int(template_height * scale)
            
            # Draw rectangle around the match
            cv2.rectangle(
                vis_frame, 
                (x, y), 
                (x + scaled_width, y + scaled_height), 
                (0, 255, 0), 
                2
            )
            
            # Add text with match details
            cv2.putText(
                vis_frame, 
                f"{template['name']} ({match_result['confidence']:.4f})", 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
            
            # Add scale information
            cv2.putText(
                vis_frame, 
                f"Scale: {scale}", 
                (x, y + scaled_height + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
        
        # Save the visualization
        self.save_frame(vis_frame, f"template_{template['name']}", frame_number)
        
        return vis_frame
    
    def visualize_roi_analysis(self, frame, roi_result, frame_number):
        """Visualize ROI analysis results"""
        # Create a copy of the frame for drawing
        vis_frame = frame.copy()
        
        # Draw overall ROI result
        detected = roi_result.get("detected", False)
        confidence = roi_result.get("confidence", 0)
        
        cv2.putText(
            vis_frame, 
            f"ROI Analysis - Detected: {detected}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255) if detected else (0, 255, 0), 
            2
        )
        
        cv2.putText(
            vis_frame, 
            f"Confidence: {confidence:.4f}", 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
        
        # Draw rectangles for each region
        regions = roi_result.get("regions", [])
        for region in regions:
            region_name = region.get("region", "unknown")
            region_detected = region.get("detected", False)
            region_confidence = region.get("confidence", 0)
            
            # Get region coordinates based on name
            height, width = frame.shape[:2]
            
            if region_name == "bottom_right":
                x, y = int(width * 0.7), int(height * 0.7)
                w, h = int(width * 0.3), int(height * 0.3)
            elif region_name == "bottom_left":
                x, y = 0, int(height * 0.7)
                w, h = int(width * 0.3), int(height * 0.3)
            elif region_name == "top_right":
                x, y = int(width * 0.7), 0
                w, h = int(width * 0.3), int(height * 0.3)
            elif region_name == "top_left":
                x, y = 0, 0
                w, h = int(width * 0.3), int(height * 0.3)
            else:
                continue
            
            # Draw rectangle around the region
            color = (0, 255, 0) if region_detected else (0, 0, 255)
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # Add text with region info
            cv2.putText(
                vis_frame, 
                f"{region_name}: {region_confidence:.2f}", 
                (x + 5, y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                1
            )
            
            # Log region features
            features = region.get("features", {})
            self.log(f"  Region {region_name}: Confidence={region_confidence:.4f}, " +
                    f"Detected={region_detected}")
            self.log(f"    Features: {features}")
        
        # Save the visualization
        vis_path = os.path.join(
            self.debug_dir, 
            f"roi_analysis_frame_{frame_number}.jpg"
        )
        cv2.imwrite(vis_path, vis_frame)
        
        return vis_path
    
    def visualize_frame_analysis(self, frame, results, frame_number):
        """Visualize the overall frame analysis results"""
        # Create a copy of the frame for drawing
        vis_frame = frame.copy()
        
        # Add text with overall results
        cv2.putText(
            vis_frame, 
            f"Frame: {results.get('frame', frame_number)}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
        
        cv2.putText(
            vis_frame, 
            f"Detected: {results.get('detected', False)}", 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255) if results.get('detected', False) else (0, 255, 0), 
            2
        )
        
        cv2.putText(
            vis_frame, 
            f"Confidence: {results.get('confidence', 0):.4f}", 
            (10, 90), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
        
        cv2.putText(
            vis_frame, 
            f"Method: {results.get('method', 'none')}", 
            (10, 120), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
        
        # Save the visualization
        self.save_frame(vis_frame, "frame_analysis", frame_number)
        
        return vis_frame
    
    def create_summary(self, detection_results, final_result):
        """Create a summary visualization of all frames"""
        # Create a summary image
        plt.figure(figsize=(15, 10))
        
        # Plot confidence values for each frame
        frame_numbers = [r.get("frame", i) for i, r in enumerate(detection_results)]
        confidences = [r.get("confidence", 0) for r in detection_results]
        detected = [r.get("detected", False) for r in detection_results]
        
        # Create color map based on detection
        colors = ['green' if d else 'red' for d in detected]
        
        plt.bar(frame_numbers, confidences, color=colors)
        plt.axhline(y=0.7, color='r', linestyle='-', label='Threshold')
        
        plt.title('Watermark Detection Confidence by Frame')
        plt.xlabel('Frame Number')
        plt.ylabel('Confidence')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        summary_path = os.path.join(self.debug_dir, "detection_summary.png")
        plt.savefig(summary_path)
        plt.close()
        
        # Create a text summary
        with open(os.path.join(self.debug_dir, "summary.txt"), "w") as f:
            f.write("TikTok Watermark Detection Summary\n")
            f.write("="*40 + "\n\n")
            f.write(f"Final result: {final_result['has_watermark']}\n")
            f.write(f"Confidence: {final_result['confidence']:.4f}\n")
            f.write(f"Detection rate: {final_result.get('detection_rate', 0):.2%}\n\n")
            
            f.write("Frame-by-frame results:\n")
            for i, result in enumerate(detection_results):
                f.write(f"Frame {result.get('frame', i)}: ")
                f.write(f"Detected={result.get('detected', False)}, ")
                f.write(f"Confidence={result.get('confidence', 0):.4f}, ")
                f.write(f"Method={result.get('method', 'unknown')}\n")
        
        return summary_path
    
    def _create_comparison(self, template, matched_region):
        """Create a side-by-side comparison of template and matched region"""
        # Convert template to 3-channel if it's grayscale
        if len(template.shape) == 2:
            template_vis = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        else:
            template_vis = template
        
        # Make sure matched region is valid
        if matched_region is None or matched_region.size == 0:
            matched_region = np.zeros_like(template_vis)
        
        # Resize matched region to match template if needed
        if template_vis.shape != matched_region.shape:
            matched_region = cv2.resize(matched_region, (template_vis.shape[1], template_vis.shape[0]))
        
        # Create side-by-side comparison
        comparison = np.hstack((template_vis, matched_region))
        
        # Add labels
        h, w = comparison.shape[:2]
        mid = w // 2
        
        cv2.putText(
            comparison, 
            "Template", 
            (10, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 255), 
            1
        )
        
        cv2.putText(
            comparison, 
            "Matched Region", 
            (mid + 10, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 255), 
            1
        )
        
        return comparison 