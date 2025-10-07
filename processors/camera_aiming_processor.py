from .base_processor import BaseProcessor
import numpy as np
import cv2
from typing import Dict, Union, Tuple, List, Optional

class CameraAimingProcessor(BaseProcessor):
    """
    Camera Aiming Processor - Similar to Seeing AI Document Mode
    
    Provides audio guidance to help users aim their camera at documents by detecting
    document edges and providing directional feedback (move left, right, up, down, closer, farther).
    
    Uses CPU-only OpenCV operations for edge detection and contour analysis.
    """
    
    def __init__(self, 
                 target_coverage=0.6,  # Target document coverage (60% of frame)
                 min_coverage=0.3,     # Minimum coverage to consider a document
                 max_coverage=0.85,    # Maximum coverage (too close)
                 edge_threshold_low=50,
                 edge_threshold_high=150):
        """
        Initialize Camera Aiming Processor
        
        Args:
            target_coverage (float): Target percentage of frame that document should cover
            min_coverage (float): Minimum coverage to detect document
            max_coverage (float): Maximum coverage before "too close" warning
            edge_threshold_low (int): Lower threshold for Canny edge detection
            edge_threshold_high (int): Higher threshold for Canny edge detection
        """
        super().__init__()
        
        self.target_coverage = target_coverage
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage
        self.edge_threshold_low = edge_threshold_low
        self.edge_threshold_high = edge_threshold_high
        
        # Tolerance for "well-centered" detection
        self.center_tolerance = 0.15  # 15% of frame dimensions
        
        print(f"Camera Aiming processor initialized (CPU-only)")
        print(f"Target coverage: {target_coverage*100}%, Min: {min_coverage*100}%, Max: {max_coverage*100}%")
    
    def _preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess image for edge detection
        
        Args:
            frame (np.ndarray): Input frame in BGR format
            
        Returns:
            np.ndarray: Grayscale image ready for edge detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply Gaussian blur to further reduce noise
        blurred = cv2.GaussianBlur(filtered, (5, 5), 0)
        
        return blurred
    
    def _detect_document_contour(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the largest rectangular contour that likely represents a document
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            Optional[np.ndarray]: Document contour or None if not found
        """
        # Preprocess
        gray = self._preprocess_image(frame)
        
        # Edge detection with adaptive thresholds
        edges = cv2.Canny(gray, self.edge_threshold_low, self.edge_threshold_high)
        
        # Dilate edges to close gaps
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Also try closing to connect nearby edges
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Sort contours by area and get the largest ones
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        frame_area = frame.shape[0] * frame.shape[1]
        
        # Look for rectangular contours with more lenient thresholds
        for contour in contours[:15]:  # Check top 15 largest contours
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # More lenient minimum threshold - even very small documents
            if area < frame_area * 0.05:  # At least 5% of frame
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Document-like aspect ratios (between 0.5 and 2.5 to allow for perspective)
            if not (0.5 <= aspect_ratio <= 2.5):
                continue
            
            # Approximate the contour to a polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            # Check rectangularity - either 4 corners or contour fills bounding box well
            rect_area = w * h
            extent = float(area) / rect_area if rect_area > 0 else 0
            
            # If it's reasonably rectangular (fills >50% of bounding box) or has 4 corners
            if len(approx) >= 4 or extent > 0.5:
                # Additional check: not too irregular
                if extent > 0.3:  # At least 30% filled
                    return contour
        
        return None
    
    def _calculate_document_metrics(self, contour: np.ndarray, frame_shape: Tuple[int, int, int]) -> Dict:
        """
        Calculate metrics for document position and framing
        
        Args:
            contour (np.ndarray): Document contour
            frame_shape (Tuple): Frame dimensions (height, width, channels)
            
        Returns:
            Dict: Metrics including center, coverage, and bounds
        """
        frame_height, frame_width = frame_shape[:2]
        frame_area = frame_height * frame_width
        
        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate center of document
        doc_center_x = x + w / 2
        doc_center_y = y + h / 2
        
        # Calculate frame center
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        
        # Calculate coverage
        doc_area = cv2.contourArea(contour)
        coverage = doc_area / frame_area
        
        # Calculate offsets (normalized)
        offset_x = (doc_center_x - frame_center_x) / frame_width
        offset_y = (doc_center_y - frame_center_y) / frame_height
        
        return {
            'center': (doc_center_x, doc_center_y),
            'frame_center': (frame_center_x, frame_center_y),
            'offset_x': offset_x,
            'offset_y': offset_y,
            'coverage': coverage,
            'bounds': (x, y, w, h),
            'aspect_ratio': w / h if h > 0 else 1.0
        }
    
    def _generate_aiming_guidance(self, metrics: Dict) -> str:
        """
        Generate human-readable aiming guidance
        
        Args:
            metrics (Dict): Document metrics from _calculate_document_metrics
            
        Returns:
            str: Aiming guidance message
        """
        coverage = metrics['coverage']
        offset_x = metrics['offset_x']
        offset_y = metrics['offset_y']
        
        guidance_parts = []
        
        # Distance guidance (coverage-based)
        if coverage < self.min_coverage:
            return "No document detected. Please point your camera at a document."
        elif coverage < self.target_coverage - 0.1:
            guidance_parts.append("Move closer")
        elif coverage > self.max_coverage:
            guidance_parts.append("Move back")
        
        # Horizontal alignment
        if abs(offset_x) > self.center_tolerance:
            if offset_x > 0:
                guidance_parts.append("move left")
            else:
                guidance_parts.append("move right")
        
        # Vertical alignment
        if abs(offset_y) > self.center_tolerance:
            if offset_y > 0:
                guidance_parts.append("move down")
            else:
                guidance_parts.append("move up")
        
        # Perfect alignment
        if not guidance_parts:
            if abs(coverage - self.target_coverage) < 0.05:
                return "Perfect! Document is well-framed."
            elif coverage < self.target_coverage:
                return "Good alignment. Move slightly closer."
            else:
                return "Good alignment. Move slightly back."
        
        # Combine guidance parts
        if len(guidance_parts) == 1:
            return guidance_parts[0].capitalize() + "."
        else:
            return guidance_parts[0].capitalize() + ", " + ", ".join(guidance_parts[1:]) + "."
    
    def _draw_guidance_overlay(self, frame: np.ndarray, contour: Optional[np.ndarray], metrics: Optional[Dict], guidance: str) -> np.ndarray:
        """
        Draw visual guidance overlay on the frame
        
        Args:
            frame (np.ndarray): Input frame
            contour (Optional[np.ndarray]): Document contour
            metrics (Optional[Dict]): Document metrics
            guidance (str): Guidance message
            
        Returns:
            np.ndarray: Frame with overlay
        """
        output = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        # Draw center crosshair
        center_x = frame_width // 2
        center_y = frame_height // 2
        crosshair_size = 30
        cv2.line(output, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), (0, 255, 0), 2)
        cv2.line(output, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), (0, 255, 0), 2)
        
        # Draw target zone (where document should be)
        target_margin_x = int(frame_width * (1 - self.target_coverage) / 2)
        target_margin_y = int(frame_height * (1 - self.target_coverage) / 2)
        cv2.rectangle(output, 
                      (target_margin_x, target_margin_y), 
                      (frame_width - target_margin_x, frame_height - target_margin_y),
                      (0, 255, 255), 2)
        
        # Draw detected document contour
        if contour is not None:
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 3)
            
            # Draw bounding box
            if metrics:
                x, y, w, h = metrics['bounds']
                color = (0, 255, 0) if "Perfect" in guidance or "Good" in guidance else (0, 165, 255)
                cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        
        # Add guidance text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(guidance, font, font_scale, thickness)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = frame_height - 30
        
        # Draw text background
        cv2.rectangle(output, 
                      (text_x - 10, text_y - text_size[1] - 10),
                      (text_x + text_size[0] + 10, text_y + 10),
                      (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(output, guidance, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        # Add coverage info
        if metrics:
            coverage_text = f"Coverage: {metrics['coverage']*100:.1f}%"
            cv2.putText(output, coverage_text, (10, 30), font, 0.6, (255, 255, 255), 2)
        
        return output
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Union[str, Dict]]:
        """
        Process frame to provide camera aiming guidance
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (output_frame_with_overlay, result_dict)
                - output_frame: Frame with visual guidance overlay
                - result_dict: Dictionary with guidance message and metrics
        """
        try:
            # Detect document
            contour = self._detect_document_contour(frame)
            
            # Calculate metrics and guidance
            if contour is not None:
                metrics = self._calculate_document_metrics(contour, frame.shape)
                guidance = self._generate_aiming_guidance(metrics)
            else:
                guidance = "No document detected. Please point your camera at a document."
                metrics = None
            
            # Draw overlay
            output_frame = self._draw_guidance_overlay(frame, contour, metrics, guidance)
            
            # Return just the guidance message - simple and focused
            return output_frame, guidance
            
        except Exception as e:
            import traceback
            error_msg = f"Error in camera aiming: {str(e)}"
            print(f"ERROR: {error_msg}")
            print(traceback.format_exc())
            return frame, "Error processing frame"
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """
        Point cloud processing not supported for camera aiming
        
        Args:
            point_cloud_data (Dict): Point cloud data (not used)
            
        Returns:
            tuple: (None, error_message)
        """
        return None, {"error": "Camera Aiming processor does not support point cloud data"}

# Instantiate processor and expose app for uvicorn
processor = CameraAimingProcessor()
app = processor.app
