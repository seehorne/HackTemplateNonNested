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
        # Lowered to 10% to make directional guidance more sensitive
        self.center_tolerance = 0.10  # 10% of frame dimensions
        
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
    
    def _is_rectangular(self, contour: np.ndarray) -> float:
        """
        Calculate how rectangular a contour is (0-1 score)
        Higher score means more rectangular
        
        Returns:
            float: Rectangularity score (0-1)
        """
        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        
        # Calculate extent (how well contour fills bounding box)
        area = cv2.contourArea(contour)
        extent = area / rect_area if rect_area > 0 else 0
        
        # Calculate convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Score based on multiple factors
        score = 0.0
        
        # 1. Number of corners (4 is ideal for rectangle)
        if len(approx) == 4:
            score += 0.5  # Perfect rectangle
        elif 4 <= len(approx) <= 6:
            score += 0.3  # Close to rectangle
        elif len(approx) <= 8:
            score += 0.1  # Somewhat rectangular
        
        # 2. Extent (should fill bounding box well)
        score += extent * 0.3
        
        # 3. Solidity (should be convex, not irregular)
        score += solidity * 0.2
        
        return min(score, 1.0)
    
    def _detect_document_contour(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect document by prioritizing RECTANGULARITY over color/brightness
        A document is defined by: straight edges, 4 corners, high extent
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            Optional[np.ndarray]: Document contour or None if not found
        """
        # Preprocess
        gray = self._preprocess_image(frame)
        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_height * frame_width
        
        # Use edge detection - this finds EDGES regardless of color
        # Documents have strong straight edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Evaluate each contour based on RECTANGULARITY, not color
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Must be at least 3% of frame
            if area < frame_area * 0.03:
                continue
            
            # Skip if unreasonably large (> 95% means it's probably the frame edge)
            if area > frame_area * 0.95:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            
            # Calculate how rectangular it is
            rectangularity = self._is_rectangular(contour)
            
            # Calculate aspect ratio
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Documents have typical aspect ratios (0.5 to 2.0, covering portrait and landscape)
            if not (0.4 <= aspect_ratio <= 2.5):
                continue
            
            # Calculate extent
            extent = area / rect_area if rect_area > 0 else 0
            
            # Documents should fill their bounding box very well (stricter than before)
            # Textured surfaces often have irregular contours with low extent
            if extent < 0.70:
                continue
            
            # Score based PRIMARILY on rectangularity
            score = rectangularity * 10.0  # Max 10 points for perfect rectangle
            
            # Add points for good aspect ratio (typical document proportions)
            if 0.6 <= aspect_ratio <= 1.7:
                score += 3
            elif 0.5 <= aspect_ratio <= 2.0:
                score += 2
            else:
                score += 1
            
            # Add points for high extent (fills bounding box well)
            if extent > 0.85:
                score += 3
            elif extent > 0.70:
                score += 2
            else:
                score += 1
            
            # Size matters but less than rectangularity
            size_ratio = area / frame_area
            if 0.15 <= size_ratio <= 0.70:
                score += 2
            elif 0.05 <= size_ratio <= 0.85:
                score += 1
            
            # Check color uniformity - documents should be relatively uniform
            # Textured surfaces (carpets, floors) have high variance
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_intensity = cv2.mean(gray, mask=mask)[0]
            
            # Calculate standard deviation of intensity (texture measure)
            masked_region = gray[mask > 0]
            if len(masked_region) > 0:
                intensity_std = np.std(masked_region)
                
                # Documents have low texture (std < 30), carpets/floors have high texture
                if intensity_std > 40:
                    # Very textured - probably not a document
                    continue
                elif intensity_std < 20:
                    # Low texture - likely a document
                    score += 2
                elif intensity_std < 30:
                    # Medium texture - could be document
                    score += 1
            
            # Brightness bonus (minor)
            if mean_intensity > 160:
                score += 1
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        # Stricter threshold - require good score to avoid false positives
        # Rectangularity * 10 + texture (2) + other features = up to ~20 points
        # Require at least 9 points (high rectangularity + some other good features)
        if best_score >= 9.0:
            return best_contour
        
        # Lower threshold for very high extent/rectangularity
        if best_score >= 7.0 and best_contour is not None:
            area = cv2.contourArea(best_contour)
            x, y, w, h = cv2.boundingRect(best_contour)
            extent = area / (w * h) if (w * h) > 0 else 0
            # Only accept if extent is very high (fills bounding box well)
            if extent > 0.80 and area > frame_area * 0.05:
                return best_contour
        
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
        
        # Also calculate coverage based on bounding box (more intuitive)
        bbox_area = w * h
        bbox_coverage = bbox_area / frame_area
        
        # Calculate offsets (normalized)
        offset_x = (doc_center_x - frame_center_x) / frame_width
        offset_y = (doc_center_y - frame_center_y) / frame_height
        
        # Use bounding box coverage for guidance (more intuitive than contour area)
        # Contour area can be irregular and confusing to users
        return {
            'center': (doc_center_x, doc_center_y),
            'frame_center': (frame_center_x, frame_center_y),
            'offset_x': offset_x,
            'offset_y': offset_y,
            'coverage': bbox_coverage,  # Use bbox coverage (more intuitive)
            'contour_coverage': coverage,  # Keep original for reference
            'bounds': (x, y, w, h),
            'aspect_ratio': w / h if h > 0 else 1.0
        }
    
    def _generate_aiming_guidance(self, metrics: Dict) -> str:
        """
        Generate human-readable aiming guidance optimized for blind/low vision users
        
        Key principles for accessibility:
        - ALWAYS give directional guidance (camera movement) first when document is off-center
        - Use camera-relative directions ("move camera left/right/up/down") not "move closer"
        - Provide positive feedback when well-framed to build confidence
        - Coverage is secondary to alignment for blind users
        
        Args:
            metrics (Dict): Document metrics from _calculate_document_metrics
            
        Returns:
            str: Aiming guidance message
        """
        coverage = metrics['coverage']
        offset_x = metrics['offset_x']
        offset_y = metrics['offset_y']
        
        directional_guidance = []
        distance_guidance = None
        
        # PRIORITY 1: Directional guidance (most important for blind users)
        # Use camera-relative language: "move camera left/right/up/down"
        # This helps users know which direction to adjust to find the document
        
        # Horizontal alignment (more lenient threshold for initial guidance)
        if abs(offset_x) > self.center_tolerance:
            if offset_x > 0:
                # Document is to the right, move camera left
                directional_guidance.append("move camera left")
            else:
                # Document is to the left, move camera right  
                directional_guidance.append("move camera right")
        
        # Vertical alignment
        if abs(offset_y) > self.center_tolerance:
            if offset_y > 0:
                # Document is below center, tilt camera down
                directional_guidance.append("tilt camera down")
            else:
                # Document is above center, tilt camera up
                directional_guidance.append("tilt camera up")
        
        # PRIORITY 2: Distance/coverage guidance (secondary)
        # Only give distance guidance if directional is good or coverage is very off
        if coverage < self.min_coverage:
            # Very small - might be too far or wrong object
            distance_guidance = "move much closer"
        elif coverage < self.target_coverage - 0.15:
            # Below target, needs to be closer
            distance_guidance = "move closer"
        elif coverage > self.max_coverage:
            # Too close
            distance_guidance = "move back"
        
        # CASE 1: Perfect or near-perfect framing
        if not directional_guidance and not distance_guidance:
            if abs(coverage - self.target_coverage) < 0.10:
                # Within 10% of target - perfect
                return "Perfect! Document is well-framed."
            elif coverage >= self.target_coverage * 0.75:
                # Close enough to target (within 75% = 45%+ coverage) - good
                return "Good framing! Document is centered."
            elif coverage >= self.min_coverage:
                # Above minimum threshold - centered but could be closer
                return "Centered. Move slightly closer to fill frame."
            elif coverage < self.target_coverage:
                return "Centered. Move closer to fill frame."
            else:
                return "Centered. Move slightly back."
        
        # CASE 2: Well-centered but wrong distance
        if not directional_guidance and distance_guidance:
            coverage_pct = int(coverage * 100)
            if coverage < self.target_coverage * 0.7:
                return f"Well-centered! Now {distance_guidance}. (Coverage: {coverage_pct}%)"
            else:
                return f"Well-centered. {distance_guidance.capitalize()}."
        
        # CASE 3: Needs directional adjustment (PRIMARY USE CASE for blind users)
        # Always lead with directional guidance
        guidance_str = directional_guidance[0].capitalize()
        if len(directional_guidance) > 1:
            guidance_str += " and " + directional_guidance[1]
        
        # Add distance if also needed, but directional is primary
        if distance_guidance and coverage < self.min_coverage * 1.5:
            guidance_str += f", then {distance_guidance}"
        
        # Add coverage for context (helps users track progress)
        coverage_pct = int(coverage * 100)
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
