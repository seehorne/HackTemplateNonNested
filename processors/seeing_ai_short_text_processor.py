from .base_processor import BaseProcessor
import numpy as np
import cv2
import os
from typing import Dict, Union, Tuple, List, Optional
from PIL import Image
import re

# Try to import EasyOCR with helpful error message
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError as e:
    EASYOCR_AVAILABLE = False
    EASYOCR_IMPORT_ERROR = str(e)

class SeeingAIShortTextProcessor(BaseProcessor):
    def __init__(self,
                 languages=['en'],  # Languages for OCR
                 use_gpu=False,  # Default to CPU as per requirements
                 min_text_length=3,
                 max_output_length=200,
                 track_text_positions=True):  # New parameter for tracking
        """
        Initialize SeeingAI Short Text processor using EasyOCR for simple OCR
        
        Args:
            languages (list): Languages for OCR recognition
            use_gpu (bool): Whether to use GPU acceleration (disabled by default per requirements)
            min_text_length (int): Minimum text length to include in output
            max_output_length (int): Maximum output length for short text format
            track_text_positions (bool): Whether to track text positions to stop reading when out of view
        """
        super().__init__()
        
        # Check EasyOCR availability first
        if not EASYOCR_AVAILABLE:
            error_msg = f"EasyOCR is not available. Please install it with: pip install easyocr\nOriginal error: {EASYOCR_IMPORT_ERROR}"
            print(f"ERROR: {error_msg}")
            raise ImportError(error_msg)
        
        # Force CPU usage as per requirements (no CUDA)
        self.use_gpu = False
        self.languages = languages
        self.reader = None  # Initialize lazily
        
        self.min_text_length = min_text_length
        self.max_output_length = max_output_length
        self.track_text_positions = track_text_positions
        
        # Text tracking for out-of-view detection
        self.previous_text_regions = []  # Store previous frame's text regions
        self.current_reading_text = ""   # Currently being read text
        self.image_dimensions = None     # Store current image dimensions
        
        print(f"SeeingAI Short Text processor initialized (EasyOCR will be loaded on first use)")
        if track_text_positions:
            print("Text position tracking enabled - will stop reading when text goes out of view")

    def _get_reader(self):
        """
        Lazy initialization of EasyOCR reader
        """
        if self.reader is None:
            try:
                print("Initializing EasyOCR reader...")
                self.reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
                print("EasyOCR reader initialized successfully")
            except Exception as e:
                print(f"Error initializing EasyOCR: {e}")
                raise e
        return self.reader

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize if image is too large (for performance)
        height, width = image.shape[:2]
        max_size = 1024
        if max(height, width) > max_size:
            ratio = max_size / max(height, width)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return image

    def _extract_text_with_easyocr(self, image: np.ndarray) -> Tuple[str, List[Dict]]:
        """
        Extract text using EasyOCR and return both text and position information
        
        Args:
            image (np.ndarray): Input image in RGB format
            
        Returns:
            Tuple[str, List[Dict]]: Extracted text and list of text regions with positions
        """
        try:
            # Get the reader (lazy initialization)
            reader = self._get_reader()
            
            # Use EasyOCR to detect and read text
            results = reader.readtext(image)
            
            # Extract text from results and store position information
            texts = []
            text_regions = []
            
            for (bbox, text, confidence) in results:
                # Filter out low confidence results
                if confidence > 0.3:  # Adjust threshold as needed
                    texts.append(text.strip())
                    
                    # Store region information for tracking
                    if self.track_text_positions:
                        # Convert bbox to a more manageable format
                        # bbox is a list of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        
                        region = {
                            'text': text.strip(),
                            'confidence': confidence,
                            'bbox': {
                                'left': min(x_coords),
                                'right': max(x_coords),
                                'top': min(y_coords),
                                'bottom': max(y_coords)
                            },
                            'center_x': sum(x_coords) / 4,
                            'center_y': sum(y_coords) / 4
                        }
                        text_regions.append(region)
            
            # Join all detected text with spaces
            combined_text = ' '.join(texts)
            return combined_text, text_regions
            
        except Exception as e:
            print(f"Error in EasyOCR processing: {e}")
            return "", []

    def _is_text_out_of_view(self, current_regions: List[Dict], viewing_bounds: Dict = None) -> List[str]:
        """
        Determine which text from previous frame has gone out of view
        
        Args:
            current_regions (List[Dict]): Current frame's text regions
            viewing_bounds (Dict): Optional viewing bounds with keys: left, right, top, bottom (as percentages)
            
        Returns:
            List[str]: List of text that has gone out of view
        """
        if not self.track_text_positions or not self.previous_text_regions:
            return []
        
        out_of_view_texts = []
        
        # If no viewing bounds specified, use full image
        if viewing_bounds is None:
            viewing_bounds = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
        
        # Calculate actual pixel bounds if image dimensions are available
        if self.image_dimensions:
            height, width = self.image_dimensions[:2]
            actual_bounds = {
                'left': width * (viewing_bounds['left'] / 100),
                'right': width * (1 - viewing_bounds['right'] / 100),
                'top': height * (viewing_bounds['top'] / 100),
                'bottom': height * (1 - viewing_bounds['bottom'] / 100)
            }
        else:
            # Fallback to using the bounds as-is
            actual_bounds = viewing_bounds
        
        # Check each previous text region
        for prev_region in self.previous_text_regions:
            text_still_visible = False
            
            # Look for this text in current regions
            for curr_region in current_regions:
                if self._texts_match(prev_region['text'], curr_region['text']):
                    # Check if current position is within viewing bounds
                    if self._is_region_in_bounds(curr_region, actual_bounds):
                        text_still_visible = True
                        break
            
            # If text was not found in current regions or is out of bounds, it's out of view
            if not text_still_visible:
                out_of_view_texts.append(prev_region['text'])
        
        return out_of_view_texts
    
    def _texts_match(self, text1: str, text2: str, similarity_threshold: float = 0.8) -> bool:
        """
        Check if two texts are similar enough to be considered the same
        
        Args:
            text1, text2 (str): Texts to compare
            similarity_threshold (float): Minimum similarity ratio
            
        Returns:
            bool: True if texts match
        """
        # Simple exact match first
        if text1.strip().lower() == text2.strip().lower():
            return True
        
        # For more fuzzy matching, could use difflib or other similarity algorithms
        # For now, check if one text contains the other (for partial matches)
        t1, t2 = text1.strip().lower(), text2.strip().lower()
        if len(t1) > 5 and len(t2) > 5:  # Only for longer texts
            return t1 in t2 or t2 in t1
        
        return False
    
    def _is_region_in_bounds(self, region: Dict, bounds: Dict) -> bool:
        """
        Check if a text region is within the specified bounds
        
        Args:
            region (Dict): Text region with bbox information
            bounds (Dict): Viewing bounds
            
        Returns:
            bool: True if region is within bounds
        """
        bbox = region['bbox']
        
        # Check if the text region overlaps with the viewing area
        # Text is considered in view if its center or significant portion overlaps
        center_x = region['center_x']
        center_y = region['center_y']
        
        # Check if center is within bounds
        if (bounds.get('left', 0) <= center_x <= bounds.get('right', float('inf')) and 
            bounds.get('top', 0) <= center_y <= bounds.get('bottom', float('inf'))):
            return True
        
        return False
        
    def _clean_and_format_text(self, text: str) -> str:
        """
        Clean and format text in SeeingAI style - short and concise
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned and formatted text
        """
        if not text:
            return ""
        
        # Clean up the text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespaces with single space
        text = text.strip()
        
        # Filter out very short text fragments that are likely noise
        if len(text) < self.min_text_length:
            return ""
        
        # Limit output length for "short text" format like SeeingAI
        if len(text) > self.max_output_length:
            text = text[:self.max_output_length].rsplit(' ', 1)[0] + "..."
        
        return text

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Union[str, Dict]]:
        """
        Process frame to extract short text in SeeingAI style with out-of-view detection
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (None, result_dict_or_string)
                - None: No visual output (text-only processor like SeeingAI)
                - result: Either extracted text string or dict with text and out-of-view info
        """
        try:
            # Store image dimensions for position tracking
            self.image_dimensions = frame.shape
            
            # Preprocess image
            processed_image = self._preprocess_image(frame)
            
            # Extract text with position information
            extracted_text, current_text_regions = self._extract_text_with_easyocr(processed_image)
            
            # Clean and format text
            final_text = self._clean_and_format_text(extracted_text)
            
            # Check for text that went out of view (if tracking is enabled)
            result = final_text if final_text else "No text detected"
            
            if self.track_text_positions:
                out_of_view_texts = self._is_text_out_of_view(current_text_regions)
                
                # Create enhanced result with out-of-view information
                result_dict = {
                    "text": result,
                    "out_of_view_texts": out_of_view_texts,
                    "total_regions": len(current_text_regions),
                    "tracking_enabled": True
                }
                
                # Add warning if text went out of view
                if out_of_view_texts:
                    result_dict["warning"] = f"Text went out of view: {', '.join(out_of_view_texts[:2])}" + ("..." if len(out_of_view_texts) > 2 else "")
                
                # Update previous regions for next frame
                self.previous_text_regions = current_text_regions.copy()
                
                return None, result_dict
            else:
                # Update previous regions for next frame (even if not actively tracking)
                self.previous_text_regions = current_text_regions.copy()
                return None, result
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None, f"Error: {str(e)}"

    def process_frame_with_bounds(self, frame: np.ndarray, viewing_bounds: Dict) -> Tuple[Optional[np.ndarray], Union[str, Dict]]:
        """
        Process frame with viewing bounds for out-of-view detection
        
        Args:
            frame (numpy.ndarray): Input frame to process
            viewing_bounds (Dict): Viewing bounds with keys: left, right, top, bottom (as percentages)
            
        Returns:
            tuple: (None, result_dict_or_string)
        """
        try:
            # Store image dimensions for position tracking
            self.image_dimensions = frame.shape
            
            # Preprocess image
            processed_image = self._preprocess_image(frame)
            
            # Extract text with position information
            extracted_text, current_text_regions = self._extract_text_with_easyocr(processed_image)
            
            # Clean and format text
            final_text = self._clean_and_format_text(extracted_text)
            
            # Check for text that went out of view using provided bounds
            out_of_view_texts = self._is_text_out_of_view(current_text_regions, viewing_bounds)
            
            # Filter current text to only include what's in view
            in_view_texts = []
            for region in current_text_regions:
                if self._is_region_in_bounds(region, self._convert_bounds_to_pixels(viewing_bounds)):
                    in_view_texts.append(region['text'])
            
            in_view_text = ' '.join(in_view_texts)
            formatted_in_view_text = self._clean_and_format_text(in_view_text)
            
            # Create enhanced result
            result_dict = {
                "text": formatted_in_view_text if formatted_in_view_text else "No text in view",
                "full_text": final_text if final_text else "No text detected",
                "out_of_view_texts": out_of_view_texts,
                "in_view_regions": len(in_view_texts),
                "total_regions": len(current_text_regions),
                "tracking_enabled": True,
                "viewing_bounds": viewing_bounds
            }
            
            # Add warnings
            if out_of_view_texts:
                result_dict["warning"] = f"Text moved out of view: {', '.join(out_of_view_texts[:2])}" + ("..." if len(out_of_view_texts) > 2 else "")
            
            if not in_view_texts and current_text_regions:
                result_dict["info"] = "Text detected but outside viewing area"
            
            # Update previous regions for next frame
            self.previous_text_regions = current_text_regions.copy()
            
            return None, result_dict
                
        except Exception as e:
            print(f"Error processing frame with bounds: {e}")
            return None, f"Error: {str(e)}"
    
    def _convert_bounds_to_pixels(self, viewing_bounds: Dict) -> Dict:
        """
        Convert percentage-based viewing bounds to pixel coordinates
        
        Args:
            viewing_bounds (Dict): Bounds as percentages
            
        Returns:
            Dict: Bounds as pixel coordinates
        """
        if not self.image_dimensions:
            return viewing_bounds
        
        height, width = self.image_dimensions[:2]
        
        return {
            'left': width * (viewing_bounds.get('left', 0) / 100),
            'right': width * (1 - viewing_bounds.get('right', 0) / 100),
            'top': height * (viewing_bounds.get('top', 0) / 100),
            'bottom': height * (1 - viewing_bounds.get('bottom', 0) / 100)
        }

    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """
        Point cloud processing not supported for text extraction
        """
        return point_cloud_data, {"message": "SeeingAI Short Text processor does not process point cloud data."}

# Create processor instance
processor = SeeingAIShortTextProcessor()
app = processor.app