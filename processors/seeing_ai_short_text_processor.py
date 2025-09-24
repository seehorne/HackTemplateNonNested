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
                 max_output_length=200):
        """
        Initialize SeeingAI Short Text processor using EasyOCR for simple OCR
        
        Args:
            languages (list): Languages for OCR recognition
            use_gpu (bool): Whether to use GPU acceleration (disabled by default per requirements)
            min_text_length (int): Minimum text length to include in output
            max_output_length (int): Maximum output length for short text format
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
        
        print(f"SeeingAI Short Text processor initialized (EasyOCR will be loaded on first use)")

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

    def _extract_text_with_easyocr(self, image: np.ndarray) -> str:
        """
        Extract text using EasyOCR
        
        Args:
            image (np.ndarray): Input image in RGB format
            
        Returns:
            str: Extracted text
        """
        try:
            # Get the reader (lazy initialization)
            reader = self._get_reader()
            
            # Use EasyOCR to detect and read text
            results = reader.readtext(image)
            
            # Extract text from results and combine
            texts = []
            for (bbox, text, confidence) in results:
                # Filter out low confidence results
                if confidence > 0.3:  # Adjust threshold as needed
                    texts.append(text.strip())
            
            # Join all detected text with spaces
            combined_text = ' '.join(texts)
            return combined_text
            
        except Exception as e:
            print(f"Error in EasyOCR processing: {e}")
            return ""

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
        Process frame to extract short text in SeeingAI style
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (None, extracted_text_string)
                - None: No visual output (text-only processor like SeeingAI)
                - extracted_text_string: Clean, short text extracted from image
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(frame)
            
            # Extract text using EasyOCR
            extracted_text = self._extract_text_with_easyocr(processed_image)
            
            # Clean and format text
            final_text = self._clean_and_format_text(extracted_text)
            
            # Return in SeeingAI style - just the text, no visual overlay
            if final_text:
                return None, final_text
            else:
                return None, "No text detected"
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None, f"Error: {str(e)}"

    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """
        Point cloud processing not supported for text extraction
        """
        return point_cloud_data, {"message": "SeeingAI Short Text processor does not process point cloud data."}

# Create processor instance
processor = SeeingAIShortTextProcessor()
app = processor.app