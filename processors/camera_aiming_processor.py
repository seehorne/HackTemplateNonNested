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
        
        # Apply adaptive histogram equalization to improve contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        return blurred
    
    def _simple_bright_region_detection(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Fallback: Simple detection of bright rectangular regions
        Useful for white documents on darker backgrounds
        """
        frame_height, frame_width = gray.shape[:2]
        frame_area = frame_height * frame_width
        
        # Find bright regions using adaptive thresholding (works better with varying lighting)
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 21, -5)
        
        # Also try simple threshold
        _, simple_thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        
        # Combine both
        combined = cv2.bitwise_or(adaptive_thresh, simple_thresh)
        
        # Clean up
        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours of bright regions
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Sort by area and find the largest reasonable one
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:5]:  # Check top 5 largest
            area = cv2.contourArea(contour)
            
            # Must be at least 3% of frame (very small documents)
            if area < frame_area * 0.03:
                continue
            
            # Skip if too large (> 90% likely the whole frame)
            if area > frame_area * 0.90:
                continue
            
            # Get bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (very lenient)
            aspect_ratio = float(w) / h if h > 0 else 0
            if not (0.2 <= aspect_ratio <= 5.0):
                continue
            
            # Calculate extent
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0
            
            # Accept if somewhat rectangular (extent > 0.2)
            if extent > 0.2:
                return contour
        
        return None
    
    def _detect_document_contour(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the largest rectangular contour that likely represents a document
        Uses multiple strategies for robust detection
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            Optional[np.ndarray]: Document contour or None if not found
        """
        # Preprocess
        gray = self._preprocess_image(frame)
        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_height * frame_width
        
        # Strategy 1: Threshold-based detection with multiple thresholds
        # Use Otsu's thresholding
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Also try fixed threshold for bright documents
        _, thresh_fixed = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        
        # Clean up the binary images
        kernel = np.ones((5, 5), np.uint8)
        thresh_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel, iterations=1)
        
        thresh_fixed = cv2.morphologyEx(thresh_fixed, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh_fixed = cv2.morphologyEx(thresh_fixed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours from both thresholds
        contours_otsu, _ = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_fixed, _ = cv2.findContours(thresh_fixed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Strategy 2: Edge detection with multiple parameter sets
        edges1 = cv2.Canny(gray, 30, 90)
        edges2 = cv2.Canny(gray, 50, 150)
        
        kernel_edge = np.ones((5, 5), np.uint8)
        edges1 = cv2.dilate(edges1, kernel_edge, iterations=1)
        edges1 = cv2.morphologyEx(edges1, cv2.MORPH_CLOSE, kernel_edge, iterations=2)
        
        edges2 = cv2.dilate(edges2, kernel_edge, iterations=1)
        edges2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, kernel_edge, iterations=2)
        
        # Find contours from edges
        contours_edge1, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_edge2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine contours from all strategies
        all_contours = (list(contours_otsu) + list(contours_fixed) + 
                       list(contours_edge1) + list(contours_edge2))
        
        if not all_contours:
            return None
        
        # Sort contours by area
        all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)
        
        # Find best document candidate
        best_contour = None
        best_score = 0
        
        for contour in all_contours[:20]:  # Check top 20 largest contours
            area = cv2.contourArea(contour)
            
            # Skip if too small (must be at least 5% of frame for real-world detection)
            if area < frame_area * 0.05:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if contour touches all frame edges (likely frame boundary)
            margin = 8
            touches_all = (x < margin and y < margin and 
                          (x + w) > (frame_width - margin) and 
                          (y + h) > (frame_height - margin))
            
            if touches_all and area > frame_area * 0.88:
                continue
            
            # Calculate aspect ratio
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Document-like aspect ratios (0.5 to 2.0 for flexibility)
            if not (0.5 <= aspect_ratio <= 2.0):
                continue
            
            # Approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Calculate extent (how well contour fills bounding box)
            rect_area = w * h
            extent = float(area) / rect_area if rect_area > 0 else 0
            
            # Skip if very irregular
            if extent < 0.4:
                continue
            
            # Score the contour
            score = 0
            
            # Rectangularity score
            if len(approx) == 4:
                score += 4
            elif 4 <= len(approx) <= 6:
                score += 3
            elif len(approx) <= 8:
                score += 2
            else:
                score += 1
            
            # Extent score
            if extent > 0.85:
                score += 3
            elif extent > 0.70:
                score += 2
            else:
                score += 1
            
            # Aspect ratio score (prefer typical document ratios)
            if 0.65 <= aspect_ratio <= 1.55:
                score += 3
            elif 0.5 <= aspect_ratio <= 2.0:
                score += 1
            
            # Size score (prefer medium to large documents, but accept small ones too)
            size_ratio = area / frame_area
            if 0.3 <= size_ratio <= 0.75:
                score += 3
            elif 0.15 <= size_ratio <= 0.85:
                score += 2
            elif 0.08 <= size_ratio < 0.15:
                score += 2  # Still give decent score to smaller documents
            elif size_ratio >= 0.08:
                score += 1
            
            # Bonus: If document appears to be white/bright (typical for paper)
            # Check mean intensity in the contour region
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_intensity = cv2.mean(gray, mask=mask)[0]
            if mean_intensity > 180:  # Bright document
                score += 2
            elif mean_intensity > 150:
                score += 1
            
            # Update best if this is better
            if score > best_score:
                best_score = score
                best_contour = contour
        
        # Very lenient threshold for real-world documents
        # Accept documents with score of 3 or more (was 4)
        # This allows small, imperfectly detected documents to still be recognized
        if best_score >= 3:
            return best_contour
        
        # If no good candidate but we have something reasonably sized, use it as fallback
        # This handles cases where detection isn't perfect but there's clearly something there
        if best_contour is not None and best_score >= 2:
            area = cv2.contourArea(best_contour)
            if area > frame_area * 0.10:  # At least 10% of frame
                return best_contour
        
        # Final fallback: Use simple bright region detection
        # This is specifically for white/light documents that the edge detection may have missed
        try:
            fallback_contour = self._simple_bright_region_detection(gray)
            if fallback_contour is not None:
                return fallback_contour
        except Exception as e:
            print(f"Fallback detection error: {e}")
        
        # Absolute last resort: if we found ANY contour, use the largest one
        if all_contours:
            largest = max(all_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > frame_area * 0.05:
                return largest
        
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
        
        # Combine guidance parts with coverage hint for context
        coverage_pct = int(coverage * 100)
        guidance_str = guidance_parts[0].capitalize()
        if len(guidance_parts) > 1:
            guidance_str += ", " + ", ".join(guidance_parts[1:])
        guidance_str += f". (Coverage: {coverage_pct}%)"
        
        return guidance_str
    
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
                
                # Add helpful context: show if document is detected and basic framing status
                result = f"{guidance}"
            else:
                guidance = "No document detected. Please point your camera at a document."
                result = guidance
                metrics = None
            
            # Draw overlay
            output_frame = self._draw_guidance_overlay(frame, contour, metrics, guidance)
            
            # Return guidance with minimal but helpful context
            return output_frame, result
            
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
