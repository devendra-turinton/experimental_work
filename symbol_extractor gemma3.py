import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import json
import os
import re
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq
from transformers import pipeline

class PIDSymbolExtractor:
    def __init__(self, model_path_or_name="google/gemma-3-8b"):
        """Initialize the PID Symbol Extractor with Gemma3 model.
        
        Args:
            model_path_or_name: Path to local Gemma3 model or HF model name
        """
        self.output_dir = "extracted_symbols"
        self.symbols_dir = os.path.join(self.output_dir, "symbols")
        self.labels_dir = os.path.join(self.output_dir, "labels")
        
        # Create output directories
        os.makedirs(self.symbols_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        # Initialize Gemma3 for vision and text tasks
        print("Loading Gemma3 model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path_or_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Create a text generation pipeline for OCR correction
        self.text_pipeline = pipeline(
            "text-generation",
            model=model_path_or_name,
            tokenizer=self.tokenizer,
            device_map="auto"
        )
        
        print("Model initialized successfully")
        
    def load_pdf(self, pdf_path):
        """Load PDF and extract pages."""
        self.doc = fitz.open(pdf_path)
        print(f"Loaded PDF with {len(self.doc)} pages")
        return len(self.doc)
    
    def process_page(self, page_num, debug=True):
        """Process a single page to extract symbols and labels."""
        print(f"Processing page {page_num+1}...")
        
        # Extract image from page
        page = self.doc[page_num]
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert to OpenCV format
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        img_orig = img_cv.copy()
        
        # Step 1: Use traditional CV for initial symbol detection
        symbols = self._detect_symbols(img_cv)
        
        # Step 2: Use Gemma3 to validate and classify detected symbols
        symbols = self._classify_symbols_with_gemma(img_orig, symbols)
        
        # Step 3: Extract text labels using improved OCR with Gemma3
        symbols_with_labels = self._extract_labels_with_gemma(img_orig, symbols)
        
        # Save results for this page
        if debug:
            self._save_debug_images(img_orig, symbols_with_labels, page_num)
        
        return symbols_with_labels
    
    def _detect_symbols(self, img):
        """Detect symbols using traditional computer vision techniques."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 5
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by size and aspect ratio
        symbols = []
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            area = w * h
            
            # Filter for likely symbols
            if 0.25 < aspect_ratio < 3.8 and area > 400:
                symbols.append({
                    'id': idx,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'confidence': 1.0  # Initial confidence
                })
        
        return symbols
    
    def _classify_symbols_with_gemma(self, img, symbols):
        """Use Gemma3 to validate and classify detected symbols."""
        enhanced_symbols = []
        
        for symbol in symbols:
            # Extract symbol region
            x, y, w, h = symbol['x'], symbol['y'], symbol['w'], symbol['h']
            symbol_img = img[y:y+h, x:x+w]
            
            # Convert to PIL for the model
            pil_img = Image.fromarray(cv2.cvtColor(symbol_img, cv2.COLOR_BGR2RGB))
            
            # Prepare prompt for Gemma3
            prompt = """
            This is an industrial P&ID symbol. Please:
            1. Classify this as one of these categories: Pump, Compressor, Valve, Vessel, 
               Heat Exchanger, Filter, Instrument, or Other
            2. Rate your confidence from 0-1
            3. Describe the specific type if possible
            
            Format your answer as:
            Category: <category>
            Confidence: <0.0-1.0>
            Type: <specific type>
            """
            
            # Process with Gemma3 vision capabilities
            # Note: Implementation depends on exact Gemma3 API
            # This is a placeholder for the actual implementation
            result = self._run_gemma_vision(pil_img, prompt)
            
            # Parse results (simplified example)
            # In real implementation, you would parse the model's text output
            category = "Valve"  # Example, replace with actual parsing
            confidence = 0.85   # Example, replace with actual parsing
            specific_type = "Gate Valve"  # Example, replace with actual parsing
            
            # Add classification to symbol data
            symbol['category'] = category
            symbol['confidence'] = confidence
            symbol['specific_type'] = specific_type
            
            enhanced_symbols.append(symbol)
        
        return enhanced_symbols
    
    def _extract_labels_with_gemma(self, img, symbols):
        """Extract and correct text labels using Gemma3-enhanced OCR."""
        symbols_with_labels = []
        
        for symbol in symbols:
            # Define search region for label (similar to original code)
            x, y, w, h = symbol['x'], symbol['y'], symbol['w'], symbol['h']
            
            # Search to the right of the symbol
            search_x = x + w
            search_y = max(0, y - int(h * 0.15))
            search_width = min(int(w * 3), img.shape[1] - search_x)
            search_height = min(int(h * 1.5), img.shape[0] - search_y)
            
            # Extract label region
            if search_width <= 0 or search_height <= 0:
                continue
                
            label_region = img[search_y:search_y+search_height, search_x:search_x+search_width]
            
            # Use traditional OCR first
            label_region_gray = cv2.cvtColor(label_region, cv2.COLOR_BGR2GRAY)
            _, label_region_thresh = cv2.threshold(
                label_region_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            # Use Tesseract OCR (you'll need pytesseract installed)
            # import pytesseract
            # ocr_text = pytesseract.image_to_string(label_region_thresh).strip()
            
            # For this example, let's assume we have OCR text
            ocr_text = "Centrifugal pump"  # Placeholder for actual OCR result
            
            # Use Gemma3 to correct OCR text based on P&ID knowledge
            corrected_text = self._correct_ocr_with_gemma(ocr_text, symbol['category'])
            
            # Add label info to symbol data
            symbol['label_text'] = corrected_text
            symbol['label_position'] = {
                'x': search_x,
                'y': search_y,
                'width': search_width,
                'height': search_height
            }
            
            symbols_with_labels.append(symbol)
        
        return symbols_with_labels
    
    def _run_gemma_vision(self, image, prompt):
        """Run Gemma3 vision model on an image with a prompt.
        This is a placeholder for the actual implementation.
        """
        # This would be implemented based on the specific Gemma3 API
        # For now, return a dummy result
        return {
            "text": "Category: Valve\nConfidence: 0.85\nType: Gate Valve"
        }
    
    def _correct_ocr_with_gemma(self, ocr_text, symbol_category):
        """Use Gemma3 to correct OCR text based on P&ID knowledge."""
        prompt = f"""
        In a P&ID diagram, I've detected a {symbol_category} symbol with the OCR text: "{ocr_text}"
        
        This might have OCR errors. Based on standard P&ID terminology, what is the most likely 
        correct label text? Only respond with the corrected text.
        """
        
        # Generate corrected text
        # This is a placeholder for the actual implementation
        # response = self.text_pipeline(prompt, max_length=50, num_return_sequences=1)
        # corrected_text = response[0]['generated_text'].strip()
        
        # For demo purposes, return a corrected example
        if symbol_category == "Pump" and "entrifugal" in ocr_text:
            corrected_text = "Centrifugal pump"
        else:
            corrected_text = ocr_text
            
        return corrected_text
    
    def _save_debug_images(self, img, symbols_with_labels, page_num):
        """Save debug images showing detected symbols and labels."""
        debug_img = img.copy()
        
        for symbol in symbols_with_labels:
            # Draw symbol box in blue
            cv2.rectangle(
                debug_img,
                (symbol['x'], symbol['y']),
                (symbol['x'] + symbol['w'], symbol['y'] + symbol['h']),
                (255, 0, 0),
                2
            )
            
            # If label exists, draw label box in green
            if 'label_position' in symbol:
                label_pos = symbol['label_position']
                cv2.rectangle(
                    debug_img,
                    (label_pos['x'], label_pos['y']),
                    (label_pos['x'] + label_pos['width'], label_pos['y'] + label_pos['height']),
                    (0, 255, 0),
                    2
                )
                
                # Add text annotation
                cv2.putText(
                    debug_img,
                    f"{symbol['category']}: {symbol['label_text']}",
                    (symbol['x'], symbol['y'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )
        
        # Save debug image
        output_path = os.path.join(self.output_dir, f"page_{page_num+1}_debug.png")
        cv2.imwrite(output_path, debug_img)
    
    def process_all_pages(self):
        """Process all pages in the PDF and save results."""
        all_results = {}
        
        for page_num in range(len(self.doc)):
            page_symbols = self.process_page(page_num)
            all_results[f"Page {page_num+1}"] = page_symbols
        
        # Save results as JSON
        with open(os.path.join(self.output_dir, "extracted_symbols.json"), "w") as f:
            json.dump(all_results, f, indent=4)
        
        print(f"✅ All symbols and labels processed")
        print(f"✅ Results saved to {os.path.join(self.output_dir, 'extracted_symbols.json')}")
        
        return all_results

# Example usage
if __name__ == "__main__":
    extractor = PIDSymbolExtractor(model_path_or_name="local/path/to/gemma3")
    extractor.load_pdf("C:/Turinton/new_clone/pid_symbol_analyzer/data/input/pid-legend.pdf")
    results = extractor.process_all_pages()