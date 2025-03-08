import cv2
import numpy as np
import os
import tempfile
import subprocess
from tqdm import tqdm
from watermark_detector import WatermarkDetector

class WatermarkRemover:
    def __init__(self, inpainting_method="deepfill", debug=False):
        """
        Initialize the watermark remover.
        
        Args:
            inpainting_method (str): The inpainting method to use ('opencv', 'lama', or 'deepfill')
            debug (bool): Whether to enable debug mode
        """
        self.inpainting_method = inpainting_method
        self.debug = debug
        self.detector = WatermarkDetector(debug=debug)
        
        # Create a temporary directory for processing
        self.temp_dir = tempfile.mkdtemp()
        
        if debug:
            print(f"Created temporary directory: {self.temp_dir}")
    
    def remove_watermark(self, input_video_path, output_video_path):
        """
        Remove TikTok watermark from a video.
        
        Args:
            input_video_path (str): Path to the input video
            output_video_path (str): Path to save the output video
            
        Returns:
            dict: Results of the watermark removal process
        """
        # Step 1: Detect the watermark
        detection_result = self.detector.detect_tiktok_watermark(input_video_path)
        
        if not detection_result["has_watermark"]:
            print("No TikTok watermark detected in the video.")
            return {
                "success": False,
                "message": "No TikTok watermark detected",
                "detection_result": detection_result
            }
        
        # Step 2: Extract frames and create masks
        frames_info = self._extract_frames_and_create_masks(input_video_path, detection_result)
        
        if not frames_info["success"]:
            return {
                "success": False,
                "message": frames_info["message"],
                "detection_result": detection_result
            }
        
        # Step 3: Apply inpainting to remove the watermark
        if self.inpainting_method == "lama":
            inpainting_result = self._apply_lama_inpainting(frames_info)
        elif self.inpainting_method == "deepfill":
            inpainting_result = self._apply_deepfill_inpainting(frames_info)
        else:
            inpainting_result = self._apply_opencv_inpainting(frames_info)
        
        if not inpainting_result["success"]:
            return {
                "success": False,
                "message": inpainting_result["message"],
                "detection_result": detection_result
            }
        
        # Step 4: Recompile the frames into a video
        compilation_result = self._compile_video(frames_info, output_video_path)
        
        # Clean up temporary files
        if not self.debug:
            self._cleanup()
        
        return {
            "success": compilation_result["success"],
            "message": compilation_result["message"],
            "detection_result": detection_result,
            "output_path": output_video_path if compilation_result["success"] else None
        }
    
    def _extract_frames_and_create_masks(self, video_path, detection_result):
        """
        Extract frames from the video and create masks for the watermark areas.
        
        Args:
            video_path (str): Path to the video
            detection_result (dict): Results from the watermark detection
            
        Returns:
            dict: Information about the extracted frames and masks
        """
        # Open the video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {
                "success": False,
                "message": f"Could not open video file: {video_path}"
            }
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create directories for frames and masks
        frames_dir = os.path.join(self.temp_dir, "frames")
        masks_dir = os.path.join(self.temp_dir, "masks")
        inpainted_dir = os.path.join(self.temp_dir, "inpainted")
        
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(inpainted_dir, exist_ok=True)
        
        # Get the best match info from detection
        best_match = detection_result.get("match_info", {})
        
        # If we don't have a match, we can't proceed
        if not best_match or not best_match.get("template_matches"):
            return {
                "success": False,
                "message": "No valid watermark match information found"
            }
        
        # Find the best template match
        template_matches = best_match.get("template_matches", [])
        best_template_match = None
        
        for match in template_matches:
            if match.get("detected", False):
                if best_template_match is None or match.get("confidence", 0) > best_template_match.get("confidence", 0):
                    best_template_match = match
        
        if not best_template_match or not best_template_match.get("location"):
            return {
                "success": False,
                "message": "No valid template match found"
            }
        
        # Get the watermark location and size
        watermark_x, watermark_y = best_template_match["location"]
        watermark_scale = best_template_match.get("scale", 1.0)
        template_name = best_template_match.get("template", "")
        
        # Find the template in the detector
        template = None
        for t in self.detector.templates:
            if t["name"] == template_name:
                template = t
                break
        
        if template is None:
            return {
                "success": False,
                "message": f"Template '{template_name}' not found"
            }
        
        # Calculate watermark dimensions
        template_height, template_width = template["image"].shape
        watermark_width = int(template_width * watermark_scale)
        watermark_height = int(template_height * watermark_scale)
        
        # Add padding to ensure we remove the entire watermark and any text below it
        padding_x = int(watermark_width * 0.2)
        padding_y = int(watermark_height * 0.5)  # Extra padding below for username
        
        # Calculate mask coordinates with padding
        mask_x = max(0, watermark_x - padding_x)
        mask_y = max(0, watermark_y - padding_y)
        mask_width = min(width - mask_x, watermark_width + 2 * padding_x)
        mask_height = min(height - mask_y, watermark_height + 2 * padding_y)
        
        # Process each frame
        print(f"Extracting frames and creating masks...")
        frame_paths = []
        mask_paths = []
        
        for frame_idx in tqdm(range(frame_count)):
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Save the frame
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            
            # Create a mask for this frame
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Fill the mask with white in the watermark area
            mask[mask_y:mask_y+mask_height, mask_x:mask_x+mask_width] = 255
            
            # Save the mask
            mask_path = os.path.join(masks_dir, f"mask_{frame_idx:06d}.png")
            cv2.imwrite(mask_path, mask)
            mask_paths.append(mask_path)
        
        # Release the video capture
        cap.release()
        
        return {
            "success": True,
            "frames_dir": frames_dir,
            "masks_dir": masks_dir,
            "inpainted_dir": inpainted_dir,
            "frame_paths": frame_paths,
            "mask_paths": mask_paths,
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "watermark_info": {
                "x": mask_x,
                "y": mask_y,
                "width": mask_width,
                "height": mask_height
            }
        }
    
    def _apply_inpainting(self, frames_info):
        """
        Apply inpainting to remove the watermark from each frame.
        
        Args:
            frames_info (dict): Information about the frames and masks
            
        Returns:
            dict: Results of the inpainting process
        """
        if self.inpainting_method == "lama":
            return self._apply_lama_inpainting(frames_info)
        elif self.inpainting_method == "deepfill":
            return self._apply_deepfill_inpainting(frames_info)
        else:
            # Default to OpenCV inpainting if no specific method is selected
            return self._apply_opencv_inpainting(frames_info)
    
    def _apply_opencv_inpainting(self, frames_info):
        """
        Apply OpenCV's inpainting algorithm to remove the watermark.
        
        Args:
            frames_info (dict): Information about the frames and masks
            
        Returns:
            dict: Results of the inpainting process
        """
        print(f"Applying OpenCV inpainting to {len(frames_info['frame_paths'])} frames...")
        inpainted_paths = []
        
        for i, (frame_path, mask_path) in enumerate(tqdm(zip(frames_info["frame_paths"], frames_info["mask_paths"]))):
            # Read the frame and mask
            frame = cv2.imread(frame_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply inpainting
            inpainted_frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            
            # Save the inpainted frame
            inpainted_path = os.path.join(frames_info["inpainted_dir"], f"inpainted_{i:06d}.png")
            cv2.imwrite(inpainted_path, inpainted_frame)
            inpainted_paths.append(inpainted_path)
        
        return {
            "success": True,
            "inpainted_paths": inpainted_paths,
            "message": "OpenCV inpainting completed successfully"
        }
    
    def _apply_lama_inpainting(self, frames_info):
        """
        Apply LaMa inpainting to remove the watermark.
        This requires the LaMa model to be installed separately.
        
        Args:
            frames_info (dict): Information about the frames and masks
            
        Returns:
            dict: Results of the inpainting process
        """
        # Check if LaMa is installed
        try:
            import torch
            from lama_cleaner.model_manager import ModelManager
            from lama_cleaner.schema import Config, HDStrategy
        except ImportError:
            print("LaMa inpainting requires the lama-cleaner package. Falling back to OpenCV inpainting.")
            return self._apply_opencv_inpainting(frames_info)
        
        print(f"Applying LaMa inpainting to {len(frames_info['frame_paths'])} frames...")
        inpainted_paths = []
        
        try:
            # Initialize LaMa model
            model = ModelManager(name="lama", device="cuda" if torch.cuda.is_available() else "cpu")
            config = Config(
                ldm_steps=25,
                ldm_sampler="plms",
                hd_strategy=HDStrategy.ORIGINAL,
                hd_strategy_crop_margin=32,
                hd_strategy_crop_trigger_size=2048,
                hd_strategy_resize_limit=2048,
            )
            
            for i, (frame_path, mask_path) in enumerate(tqdm(zip(frames_info["frame_paths"], frames_info["mask_paths"]))):
                # Read the frame and mask
                frame = cv2.imread(frame_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                # Apply LaMa inpainting
                inpainted_frame = model(frame, mask, config)
                
                # Save the inpainted frame
                inpainted_path = os.path.join(frames_info["inpainted_dir"], f"inpainted_{i:06d}.png")
                cv2.imwrite(inpainted_path, inpainted_frame)
                inpainted_paths.append(inpainted_path)
            
            return {
                "success": True,
                "inpainted_paths": inpainted_paths,
                "message": "LaMa inpainting completed successfully"
            }
        except Exception as e:
            print(f"Error applying LaMa inpainting: {str(e)}")
            print("Falling back to OpenCV inpainting.")
            return self._apply_opencv_inpainting(frames_info)
    
    def _apply_deepfill_inpainting(self, frames_info):
        """
        Apply DeepFill v2 inpainting to remove the watermark.
        Uses the Transformers library implementation.
        
        Args:
            frames_info (dict): Information about the frames and masks
            
        Returns:
            dict: Results of the inpainting process
        """
        try:
            # Import required libraries
            from transformers import pipeline
            import torch
            
            print(f"Applying DeepFill v2 inpainting to {len(frames_info['frame_paths'])} frames...")
            inpainted_paths = []
            
            # Initialize the inpainting pipeline
            device = 0 if torch.cuda.is_available() else -1
            inpainter = pipeline("inpainting", model="fudan-univeristy/dict-guided-inpainting", device=device)
            
            for i, (frame_path, mask_path) in enumerate(tqdm(zip(frames_info["frame_paths"], frames_info["mask_paths"]))):
                # Read the frame and mask
                frame = cv2.imread(frame_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                # Convert to RGB for the model
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL images
                from PIL import Image
                import numpy as np
                
                frame_pil = Image.fromarray(frame_rgb)
                # For the mask, we need to invert it (255 for the area to inpaint)
                mask_inv = 255 - mask
                mask_pil = Image.fromarray(mask_inv)
                
                # Apply inpainting
                result = inpainter(frame_pil, mask_pil)
                
                # Convert back to OpenCV format
                inpainted_frame = cv2.cvtColor(np.array(result["images"][0]), cv2.COLOR_RGB2BGR)
                
                # Save the inpainted frame
                inpainted_path = os.path.join(frames_info["inpainted_dir"], f"inpainted_{i:06d}.png")
                cv2.imwrite(inpainted_path, inpainted_frame)
                inpainted_paths.append(inpainted_path)
            
            return {
                "success": True,
                "inpainted_paths": inpainted_paths,
                "message": "DeepFill v2 inpainting completed successfully"
            }
        except Exception as e:
            print(f"Error applying DeepFill v2 inpainting: {str(e)}")
            print("Falling back to OpenCV inpainting.")
            return self._apply_opencv_inpainting(frames_info)
    
    def _compile_video(self, frames_info, output_path):
        """
        Compile the inpainted frames into a video.
        
        Args:
            frames_info (dict): Information about the frames
            output_path (str): Path to save the output video
            
        Returns:
            dict: Results of the video compilation
        """
        try:
            # Get the first inpainted frame to determine dimensions
            first_frame = cv2.imread(os.path.join(frames_info["inpainted_dir"], "inpainted_000000.png"))
            height, width, _ = first_frame.shape
            
            # Create a video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, frames_info["fps"], (width, height))
            
            # Add each frame to the video
            print(f"Compiling video from {len(frames_info['frame_paths'])} frames...")
            for i in tqdm(range(frames_info["frame_count"])):
                frame_path = os.path.join(frames_info["inpainted_dir"], f"inpainted_{i:06d}.png")
                
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    out.write(frame)
            
            # Release the video writer
            out.release()
            
            # If ffmpeg is available, use it to improve the output video quality
            if self._is_ffmpeg_available():
                print("Using ffmpeg to improve output video quality...")
                temp_output = output_path + ".temp.mp4"
                os.rename(output_path, temp_output)
                
                # Use ffmpeg to create a higher quality video
                cmd = [
                    "ffmpeg", "-y",
                    "-i", temp_output,
                    "-c:v", "libx264",
                    "-preset", "slow",
                    "-crf", "18",
                    "-c:a", "copy",
                    output_path
                ]
                
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Remove the temporary file
                os.remove(temp_output)
            
            return {
                "success": True,
                "message": "Video compilation completed successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error compiling video: {str(e)}"
            }
    
    def _is_ffmpeg_available(self):
        """Check if ffmpeg is available on the system."""
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary directory: {str(e)}") 