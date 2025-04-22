import fitz  # PyMuPDF
import cv2
import numpy as np
import os
import json
import glob
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import defaultdict
import asyncio
import concurrent.futures

class AdaptivePIDSymbolDetector:
    """
    An enhanced detector for P&ID symbols with adaptive thresholding
    and context-aware detection to reduce overfitting and underfitting.
    """

    def __init__(self, 
                 symbols_dir, 
                 output_dir="detection_output", 
                 max_workers=None, 
                 use_gpu=False):
        """
        Initialize the detector with paths and configuration options.
        
        Args:
            symbols_dir: Directory containing symbol template images
            output_dir: Directory to save detection results
            max_workers: Number of worker processes/threads (None = auto)
            use_gpu: Whether to use GPU acceleration if available
        """
        self.symbols_dir = symbols_dir
        self.output_dir = output_dir
        self.max_workers = max_workers or os.cpu_count()
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        # Detection parameters that will be dynamically adjusted
        self.params = {
            'base_match_threshold': 0.75,
            'validation_edge_threshold': 0.02,
            'validation_contrast_threshold': 15,
            'clustering_distance_factor': 0.4,
            'min_cluster_samples': 1,
            'scales': [0.8, 0.9, 1.0, 1.1, 1.2],
            'rotations': [0, 90, 180, 270],  # Reduced set of rotations by default
            'feature_match_ratio': 0.7,
            'min_feature_matches': 10
        }
        
        # Context-related parameters
        self.context = {
            'region_density_map': None,
            'detected_regions': [],
            'text_regions': [],
            'region_statistics': {}
        }
        
        # Load symbol templates
        self.symbols = self._load_symbols()
        self.results = {}
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize logging
        self.log_file = os.path.join(self.output_dir, "detection_log.txt")
        with open(self.log_file, 'w') as f:
            f.write(f"AdaptivePIDSymbolDetector initialized\n")
            f.write(f"Symbols directory: {symbols_dir}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write(f"Max workers: {self.max_workers}\n")
            f.write(f"Using GPU: {self.use_gpu}\n")

    def _log(self, message):
        """Log a message to the log file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        except Exception:
            # Silently fail if logging isn't possible
            pass

    async def detect_symbols_in_pdf(self, pdf_path, match_threshold=None, use_parallel=True, dpi=150):
        """
        Main method to detect symbols in a PDF document with improved performance.
        
        Args:
            pdf_path: Path to the PDF file
            match_threshold: Initial matching threshold (None = use default)
            use_parallel: Whether to use parallel processing
            dpi: Resolution for PDF rendering
            
        Returns:
            Detection results dictionary
        """
        start_time = time.time()
        
        if match_threshold is not None:
            self.params['base_match_threshold'] = match_threshold
            
        self._log(f"Starting symbol detection in {pdf_path} with threshold {self.params['base_match_threshold']}")
        
        # Initialize results dictionary
        for symbol_name in self.symbols:
            self.results[symbol_name] = {
                "present": False,
                "occurrences": 0,
                "positions": []
            }
        
        # Process the PDF
        try:
            doc = fitz.open(pdf_path)
            self._log(f"PDF loaded with {len(doc)} pages")
            
            # First pass: Quick scan to identify pages with potential symbols
            potential_pages = self._quick_scan_pdf(doc, dpi // 4, self.params['base_match_threshold'])
            self._log(f"Quick scan identified {len(potential_pages)} potential pages out of {len(doc)}")
            
            # Only analyze context for potential pages to save time
            self._analyze_document_context(doc, dpi, potential_pages)
            
            # Prepare batches of symbols for parallel processing
            symbol_batches = []
            batch_size = max(1, len(self.symbols) // (self.max_workers * 2))
            symbols_list = list(self.symbols.items())
            
            for i in range(0, len(symbols_list), batch_size):
                symbol_batches.append(dict(symbols_list[i:i+batch_size]))
            
            self._log(f"Created {len(symbol_batches)} symbol batches for parallel processing")
            
            # Process each potential page
            for page_num in potential_pages:
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
                
                # Convert pixmap to numpy array
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                
                if pix.n == 3 or pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Convert to grayscale and preprocess in parallel
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Apply text masking
                text_mask = self.context.get(f'text_mask_page_{page_num}', None)
                if text_mask is not None:
                    if text_mask.shape != gray_img.shape:
                        text_mask = cv2.resize(text_mask, (gray_img.shape[1], gray_img.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                
                # Process image in parallel
                loop = asyncio.get_running_loop()
                preprocessed_img = await loop.run_in_executor(None, self._preprocess_image, gray_img)
                
                # Apply text mask to preprocessed image
                if text_mask is not None:
                    masked_img = preprocessed_img.copy()
                    masked_img[text_mask > 0] = 255
                else:
                    masked_img = preprocessed_img
                
                # Process each batch of symbols in parallel
                page_results = {}
                
                # Initialize page results
                for symbol_name in self.symbols:
                    page_results[symbol_name] = {
                        "present": False,
                        "occurrences": 0,
                        "positions": []
                    }
                
                # Create tasks for processing symbol batches
                tasks = []
                for batch_idx, symbol_batch in enumerate(symbol_batches):
                    task_data = (batch_idx, symbol_batch, masked_img, gray_img, img, text_mask, page_num)
                    tasks.append(self._process_symbol_batch(task_data))
                
                # Run tasks concurrently and wait for all to finish
                batch_results = await asyncio.gather(*tasks)
                
                # Merge batch results
                for batch_result in batch_results:
                    for symbol_name, detections in batch_result.items():
                        if detections:
                            page_results[symbol_name]["present"] = True
                            page_results[symbol_name]["occurrences"] = len(detections)
                            page_results[symbol_name]["positions"] = detections
                
                # Update global results
                self._update_results(page_results)
                
                # Create visualization
                vis_path = os.path.join(self.output_dir, f"page_{page_num+1}_detection.png")
                await loop.run_in_executor(
                    None, 
                    self._save_visualization, 
                    img, page_results, vis_path
                )
                
                self._log(f"Completed detection for page {page_num+1}")
            
            doc.close()
            
            # Post-processing to remove false positives
            self._post_process_results()
            
        except Exception as e:
            self._log(f"Error during detection: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
        
        elapsed_time = time.time() - start_time
        self._log(f"Detection completed in {elapsed_time:.2f} seconds")
        print(f"Detection completed in {elapsed_time:.2f} seconds")
        
        return self.results

    async def _process_symbol_batch(self, batch_data):
        """Process a batch of symbols for a page asynchronously."""
        batch_idx, symbol_batch, masked_img, gray_img, original_img, text_mask, page_num = batch_data
        batch_results = {}
        
        loop = asyncio.get_running_loop()
        
        # Process each symbol in the batch
        tasks = []
        for symbol_name, symbol_data in symbol_batch.items():
            # Use the symbol's specific threshold
            threshold = symbol_data.get('match_threshold', self.params['base_match_threshold'])
            
            task_data = (
                symbol_name,
                symbol_data,
                masked_img,
                gray_img,
                original_img,
                text_mask,
                None,  # density_map - we'll skip this for performance
                page_num,
                threshold
            )
            tasks.append(loop.run_in_executor(None, self._detect_symbol, task_data))
        
        # Wait for all symbol detections to complete
        results = await asyncio.gather(*tasks)
        
        # Process results
        for symbol_name, detections in results:
            batch_results[symbol_name] = detections
        
        return batch_results

    def _quick_scan_pdf(self, doc, dpi, threshold):
        """
        Perform a quick scan of the PDF at low resolution to identify pages with potential symbols.
        
        Args:
            doc: The PDF document
            dpi: Low resolution DPI for quick scanning
            threshold: Match threshold
        
        Returns:
            Set of page numbers that likely contain symbols
        """
        potential_pages = set()
        
        # Select a subset of common symbols for scanning
        sample_size = min(5, len(self.symbols))
        symbol_names = list(self.symbols.keys())
        step = max(1, len(symbol_names) // sample_size)
        sample_symbols = {symbol_names[i]: self.symbols[symbol_names[i]] 
                        for i in range(0, len(symbol_names), step)}
        
        # Scan each page with basic templates
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            if pix.n == 3 or pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Check for potential matches with basic templates
            for symbol_name, symbol_data in sample_symbols.items():
                # Use only original scale, no rotation for quick check
                base_template = next((t["image"] for t in symbol_data["multi_scale"] 
                                if t["scale"] == 1.0 and t["rotation"] == 0), None)
                
                if base_template is None or base_template.shape[0] > gray_img.shape[0] or base_template.shape[1] > gray_img.shape[1]:
                    continue
                
                try:
                    # Use higher threshold for quick scan
                    quick_threshold = threshold + 0.05
                    result = cv2.matchTemplate(gray_img, base_template, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= quick_threshold)
                    
                    if len(locations[0]) > 0:
                        potential_pages.add(page_num)
                        break
                except cv2.error:
                    continue
        
        # If no pages found, include all pages
        if not potential_pages and len(doc) > 0:
            potential_pages = set(range(len(doc)))
        
        return potential_pages

    def _save_visualization(self, img, page_results, vis_path):
        """Save a visualization of the detection results."""
        vis_img = img.copy()
        
        for symbol_name, data in page_results.items():
            if data["present"]:
                for match in data["positions"]:
                    x, y = match["x"], match["y"]
                    width, height = match["width"], match["height"]
                    confidence = match["confidence"]
                    method = match["method"]
                    
                    # Color based on confidence and method
                    if method == "template":
                        if confidence > 0.85:
                            color = (0, 255, 0)  # Green for high confidence
                        elif confidence > 0.75:
                            color = (0, 255, 255)  # Yellow for medium confidence
                        else:
                            color = (0, 165, 255)  # Orange for lower confidence
                    else:  # feature matching
                        color = (255, 0, 0)  # Red for feature-based matches
                    
                    # Draw rectangle and label
                    cv2.rectangle(vis_img, (x, y), (x + width, y + height), color, 2)
                    label_text = f"{symbol_name[:10]}... {confidence:.2f}"
                    cv2.putText(vis_img, label_text, (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imwrite(vis_path, vis_img)

    def _detect_symbol(self, task_data):
        """
        Detect a specific symbol in a page with early rejection and optimized processing.
        
        Args:
            task_data: Tuple of detection parameters
            
        Returns:
            Tuple of (symbol_name, list of detections)
        """
        (symbol_name, symbol_data, masked_img, gray_img, original_img, 
        text_mask, density_map, page_num, threshold) = task_data
        
        detections = []
        
        # Skip if inputs are invalid
        if masked_img is None or symbol_data["gray"] is None:
            return symbol_name, detections
        
        # Quick rejection based on global image statistics
        symbol_hist = symbol_data["features"]["histogram"]
        img_hist = cv2.calcHist([gray_img], [0], None, [32], [0, 256])
        cv2.normalize(img_hist, img_hist, 0, 1, cv2.NORM_MINMAX)
        
        hist_sim = cv2.compareHist(symbol_hist, img_hist, cv2.HISTCMP_CORREL)
        if hist_sim < 0.3:  # Very different histograms
            return symbol_name, detections
        
        # Check if symbol size is reasonable for the image
        sym_h, sym_w = symbol_data["gray"].shape
        img_h, img_w = gray_img.shape
        
        if sym_h > img_h * 0.5 or sym_w > img_w * 0.5:
            # Symbol is too large relative to the image
            return symbol_name, detections
        
        # First try template matching (more reliable but slower)
        template_detections = self._detect_with_templates(
            symbol_name, symbol_data, masked_img, gray_img, 
            original_img, text_mask, page_num, threshold)
        
        # If enough template detections found, skip feature matching
        if len(template_detections) >= 3:
            return symbol_name, template_detections
        
        # If template detection fails or finds few matches and symbol is complex enough,
        # try feature matching as a fallback
        if (len(template_detections) < 3 and 
            symbol_data["keypoints"] is not None and 
            len(symbol_data["keypoints"]) >= 10 and
            symbol_data["features"]["complexity"] > 1.2):  # Only try for complex symbols
            
            feature_detections = self._detect_with_features(
                symbol_name, symbol_data, gray_img, original_img, page_num)
            
            # Add feature detections
            detections.extend(feature_detections)
        
        # Add template detections
        detections.extend(template_detections)
        
        # Remove overlapping detections (prefer template matches over feature matches)
        detections = self._filter_overlapping_detections(detections)
        
        # Apply density-based filtering if density map is available
        if density_map is not None and len(detections) > 0:
            filtered_detections = []
            
            # Resize density map if needed
            if density_map.shape != gray_img.shape:
                density_map = cv2.resize(density_map, (gray_img.shape[1], gray_img.shape[0]), 
                                    interpolation=cv2.INTER_LINEAR)
            
            for detection in detections:
                x, y, w, h = detection["x"], detection["y"], detection["width"], detection["height"]
                
                # Check if detection is in a high-density region
                if x < density_map.shape[1] and y < density_map.shape[0]:
                    region_density = np.mean(density_map[y:y+h, x:x+w])
                    
                    # Keep detection if in high density region or high confidence
                    if region_density > 0.2 or detection["confidence"] > 0.8:
                        filtered_detections.append(detection)
                else:
                    # Keep detection if outside density map bounds
                    filtered_detections.append(detection)
            
            detections = filtered_detections
        
        # Limit number of detections to avoid excessive results
        if len(detections) > 10:
            detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)[:10]
        
        return symbol_name, detections

    def _load_symbols(self):
        """
        Load all symbol templates with optimized preprocessing.
        Returns a dictionary of symbol data.
        """
        symbols = {}
        
        # Get all symbol images
        symbol_paths = glob.glob(os.path.join(self.symbols_dir, "*.png"))
        
        for path in symbol_paths:
            # Get symbol name from filename
            symbol_name = Path(path).stem
            
            # Load the symbol image
            symbol_img = cv2.imread(path)
            
            if symbol_img is not None:
                # Convert to grayscale
                symbol_gray = cv2.cvtColor(symbol_img, cv2.COLOR_BGR2GRAY)
                
                # Compute symbol characteristics for adaptive parameters
                features = self._compute_symbol_features(symbol_gray)
                
                # Pre-generate templates at different scales and rotations
                multi_scale_templates = self._generate_templates(symbol_gray)
                
                # Initialize feature detector
                orb = cv2.ORB_create(nfeatures=500)
                keypoints, descriptors = orb.detectAndCompute(symbol_gray, None)
                
                symbols[symbol_name] = {
                    "name": symbol_name, 
                    "path": path,
                    "image": symbol_img,
                    "gray": symbol_gray,
                    "multi_scale": multi_scale_templates,
                    "keypoints": keypoints,
                    "descriptors": descriptors,
                    "features": features,
                    "match_threshold": None  # Will be set dynamically
                }
        
        print(f"Loaded {len(symbols)} symbols from {self.symbols_dir}")
        self._log(f"Loaded {len(symbols)} symbols from {self.symbols_dir}")
        return symbols
    
    def _generate_text_mask(self, img):
        """
        Generate a mask of text regions with improved accuracy and less aggressiveness
        to avoid excluding small symbols.
        
        Args:
            img: Grayscale image
            
        Returns:
            Binary mask where text regions are marked as 255
        """
        # Use adaptive processing based on image resolution
        is_high_res = img.shape[0] > 2000 or img.shape[1] > 2000
        
        # Create horizontal kernel to detect text lines (adjust size based on resolution)
        h_kernel_size = 15 if is_high_res else 11
        h_kernel = np.ones((1, h_kernel_size), np.uint8)
        h_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, h_kernel)
        h_diff = cv2.subtract(img, h_opening)
        _, h_mask = cv2.threshold(h_diff, 40, 255, cv2.THRESH_BINARY)  # Higher threshold to be less aggressive
        
        # Create vertical kernel for vertical text (adjust size based on resolution)
        v_kernel_size = 15 if is_high_res else 11
        v_kernel = np.ones((v_kernel_size, 1), np.uint8)
        v_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, v_kernel)
        v_diff = cv2.subtract(img, v_opening)
        _, v_mask = cv2.threshold(v_diff, 40, 255, cv2.THRESH_BINARY)  # Higher threshold to be less aggressive
        
        # Combine horizontal and vertical text
        text_mask = cv2.bitwise_or(h_mask, v_mask)
        
        # Apply more selective dilation to avoid covering symbol areas
        text_mask = cv2.dilate(text_mask, np.ones((2, 2), np.uint8), iterations=1)
        
        # Use connected component analysis to filter out small regions that could be symbols
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(text_mask, connectivity=8)
        
        # Create refined mask
        refined_mask = np.zeros_like(text_mask)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Skip small components
            if area < 50:
                continue
                
            # Skip components with aspect ratios similar to symbols
            aspect_ratio = width / float(height) if height > 0 else 0
            if 0.5 <= aspect_ratio <= 2.0 and area < 400:
                continue
                
            # Skip components that are relatively square and small (likely symbols)
            if 0.8 <= aspect_ratio <= 1.25 and area < 600:
                continue
            
            # Keep the component if it passed all filters
            component_mask = (labels == i)
            refined_mask[component_mask] = 255
        
        # Final cleanup - remove isolated text pixels
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        
        return refined_mask

    def _preprocess_image(self, img):
        """
        Optimized image preprocessing for better symbol detection and memory efficiency.
        
        Args:
            img: Grayscale image
            
        Returns:
            Preprocessed image
        """
        # Apply preprocessing at reduced precision to save memory
        img_float = img.astype(np.float32) if img.dtype != np.float32 else img.copy()
        
        # Normalize contrast with in-place operations when possible
        # First find min/max for normalization
        min_val = np.min(img_float)
        max_val = np.max(img_float)
        
        # Only normalize if needed (if range is not already 0-255)
        if min_val > 0 or max_val < 255:
            # Manual normalization is faster for large images
            if max_val > min_val:
                img_float = 255 * (img_float - min_val) / (max_val - min_val)
            
        # Convert back to uint8 for further processing
        normalized = img_float.astype(np.uint8)
        
        # Apply median blur for noise reduction (very effective for P&ID diagrams)
        # Use smaller kernel for efficiency
        blurred = cv2.medianBlur(normalized, 3)
        
        # Use adaptive thresholding to handle varying lighting conditions
        # Adjust block size based on image resolution
        block_size = min(15, max(7, int(img.shape[1] / 300) * 2 + 1))
        if block_size % 2 == 0:
            block_size += 1  # Ensure odd block size
            
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, block_size, 2)
        
        # Morphological operations to clean up noise and connect broken lines
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Enhance small objects that might be symbols
        # This is especially useful for small valve and instrument symbols
        small_kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.dilate(cleaned, small_kernel, iterations=1)
        
        return cleaned

    def _adjust_detection_parameters(self):
        """
        Adjust detection parameters based on document context and symbols.
        Enhances adaptability to different diagram styles and qualities.
        """
        # If no statistics available, use defaults
        if not self.context['region_statistics']:
            return
        
        # Calculate average statistics across pages
        avg_text_density = np.mean([stats['text_density'] for stats in self.context['region_statistics'].values()])
        avg_edge_density = np.mean([stats['edge_density'] for stats in self.context['region_statistics'].values()])
        avg_contrast = np.mean([stats['contrast'] for stats in self.context['region_statistics'].values()])
        
        self._log(f"Document statistics: text_density={avg_text_density:.4f}, edge_density={avg_edge_density:.4f}, contrast={avg_contrast:.2f}")
        
        # Adjust match threshold based on document characteristics
        orig_threshold = self.params['base_match_threshold']
        
        # Lower threshold for dense diagrams with high edge density
        if avg_edge_density > 0.1:
            self.params['base_match_threshold'] = max(0.65, self.params['base_match_threshold'] - 0.05)
        
        # Lower threshold for low contrast documents
        if avg_contrast < 40:
            self.params['base_match_threshold'] = max(0.65, self.params['base_match_threshold'] - 0.05)
            # Lower contrast documents need more lenient validation
            self.params['validation_contrast_threshold'] = max(8, self.params['validation_contrast_threshold'] - 7)
        
        # Adjust clustering distance based on diagram density
        if avg_edge_density > 0.15:
            # Denser diagrams need tighter clustering
            self.params['clustering_distance_factor'] = max(0.2, self.params['clustering_distance_factor'] - 0.1)
        elif avg_edge_density < 0.05:
            # Sparse diagrams need wider clustering
            self.params['clustering_distance_factor'] = min(0.6, self.params['clustering_distance_factor'] + 0.1)
        
        # Adapt to text density - high text diagrams need more aggressive validation
        if avg_text_density > 0.3:
            self.params['validation_edge_threshold'] = min(0.04, self.params['validation_edge_threshold'] + 0.01)
        
        # Set individual symbol match thresholds based on their characteristics
        for symbol_name, symbol_data in self.symbols.items():
            features = symbol_data['features']
            
            # More complex symbols can use lower thresholds (more distinctive)
            complexity_factor = min(0.1, (features['complexity'] - 1.0) * 0.05)
            
            # High edge density symbols can use lower thresholds (more distinctive)
            edge_factor = min(0.1, features['edge_density'] * 0.5)
            
            # Adjust size factor more aggressively for small symbols
            # Small symbols need lower thresholds to be detected
            if features['area'] < 500:
                size_factor = min(-0.15, -0.05 - (500 - features['area']) / 5000)
            elif features['area'] < 1000:
                size_factor = -0.1
            elif features['area'] > 10000:
                size_factor = 0.05  # Larger symbols need higher thresholds
            else:
                size_factor = 0
            
            # Aspect ratio factor - very wide or tall symbols need lower thresholds
            aspect_ratio = features['aspect_ratio']
            if aspect_ratio > 3 or aspect_ratio < 0.33:
                aspect_factor = -0.05  # Lower threshold for extreme aspect ratios
            else:
                aspect_factor = 0
            
            # Combine factors
            adjusted_threshold = self.params['base_match_threshold'] - complexity_factor - edge_factor + size_factor + aspect_factor
            
            # Ensure threshold is within reasonable bounds
            symbol_data['match_threshold'] = max(0.6, min(0.9, adjusted_threshold))
        
        self._log(f"Adjusted base threshold from {orig_threshold} to {self.params['base_match_threshold']}")
        self._log("Detection parameters adjusted based on document context")

    def _compute_symbol_features(self, symbol_gray):
        """
        Compute features of a symbol to use for adaptive parameter adjustment.
        
        Args:
            symbol_gray: Grayscale image of the symbol
            
        Returns:
            Dictionary of symbol features
        """
        # Calculate edge density
        edges = cv2.Canny(symbol_gray, 100, 200)
        edge_density = np.count_nonzero(edges) / symbol_gray.size
        
        # Calculate contrast
        contrast = np.std(symbol_gray)
        
        # Calculate aspect ratio
        h, w = symbol_gray.shape
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Calculate complexity using contour analysis
        _, binary = cv2.threshold(symbol_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            complexity = perimeter**2 / (4 * np.pi * area) if area > 0 else 1.0
        else:
            complexity = 1.0
        
        # Calculate histogram to use for quick rejection
        hist = cv2.calcHist([symbol_gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        return {
            "edge_density": edge_density,
            "contrast": contrast,
            "aspect_ratio": aspect_ratio,
            "complexity": complexity,
            "histogram": hist,
            "size": (w, h),
            "area": w * h
        }
    
    def _generate_templates(self, symbol_gray):
        """
        Generate templates at different scales and rotations.
        
        Args:
            symbol_gray: Grayscale symbol image
            
        Returns:
            List of dictionaries with template variations
        """
        multi_scale_templates = []
        
        for scale in self.params['scales']:
            width = int(symbol_gray.shape[1] * scale)
            height = int(symbol_gray.shape[0] * scale)
            
            if width <= 0 or height <= 0:
                continue
                
            resized = cv2.resize(symbol_gray, (width, height), interpolation=cv2.INTER_AREA)
            
            # Generate rotated versions
            for angle in self.params['rotations']:
                if angle == 0:
                    rotated = resized
                else:
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(resized, rotation_matrix, (width, height), 
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                
                # Pre-compute edges for faster validation
                edges = cv2.Canny(rotated, 100, 200)
                
                # Pre-compute histogram for quick rejection
                hist = cv2.calcHist([rotated], [0], None, [32], [0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                
                multi_scale_templates.append({
                    "image": rotated,
                    "edges": edges,
                    "histogram": hist,
                    "scale": scale,
                    "rotation": angle
                })
        
        return multi_scale_templates
    
    def _analyze_document_context(self, doc, dpi, potential_pages=None):
        """
        First pass to analyze document context for better detection parameters.
        
        Args:
            doc: The PDF document object
            dpi: Resolution for PDF rendering
            potential_pages: Optional set of page numbers to analyze (None = analyze all)
        """
        self._log("Starting document context analysis")
        
        # Use lower resolution for faster processing
        context_dpi = dpi // 2
        
        # Process each page to identify text regions and overall structure
        page_range = potential_pages if potential_pages is not None else range(len(doc))
        
        for page_num in page_range:
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(context_dpi/72, context_dpi/72))
            
            # Convert pixmap to numpy array
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            if pix.n == 3 or pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Generate text mask
            text_mask = self._generate_text_mask(gray_img)
            
            # Store text mask for later use (resized to full resolution)
            self.context[f'text_mask_page_{page_num}'] = cv2.resize(
                text_mask, 
                (int(pix.width * dpi / context_dpi), int(pix.height * dpi / context_dpi)),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Generate density map of potential symbol regions
            density_map = self._generate_region_density_map(gray_img, text_mask)
            
            # Store density map
            self.context[f'density_map_page_{page_num}'] = density_map
            
            # Calculate page statistics
            self._calculate_page_statistics(gray_img, text_mask, page_num)
        
        # Adjust detection parameters based on document context
        self._adjust_detection_parameters()
        
        self._log("Document context analysis completed")    

    def _detect_with_templates(self, symbol_name, symbol_data, masked_img, gray_img, 
                            original_img, text_mask, page_num, threshold):
        """
        Detect a symbol using a two-stage template matching approach for better performance.
        
        Args:
            symbol_name: Name of the symbol
            symbol_data: Symbol data
            masked_img: Preprocessed image with text masked out
            gray_img: Original grayscale image
            original_img: Original color image
            text_mask: Binary mask of text regions
            page_num: Page number
            threshold: Match threshold
            
        Returns:
            List of detection dictionaries
        """
        # Quick rejection using global histogram comparison
        symbol_hist = symbol_data["features"]["histogram"]
        img_hist = cv2.calcHist([gray_img], [0], None, [32], [0, 256])
        cv2.normalize(img_hist, img_hist, 0, 1, cv2.NORM_MINMAX)
        
        hist_sim = cv2.compareHist(symbol_hist, img_hist, cv2.HISTCMP_CORREL)
        if hist_sim < 0.25:  # Very different histograms - quick reject
            return []
        
        # First stage: Quick scan at lower resolution
        # Downscale both image and template for faster initial detection
        downscale_factor = 4
        h, w = gray_img.shape
        small_gray = cv2.resize(gray_img, (w//downscale_factor, h//downscale_factor))
        
        # Get base template (original scale, no rotation)
        base_template_idx = next((i for i, t in enumerate(symbol_data["multi_scale"]) 
                            if t["scale"] == 1.0 and t["rotation"] == 0), 0)
        
        if base_template_idx >= len(symbol_data["multi_scale"]):
            base_template_idx = 0
        
        base_template = symbol_data["multi_scale"][base_template_idx]["image"]
        th, tw = base_template.shape
        small_template = cv2.resize(base_template, (tw//downscale_factor, th//downscale_factor))
        
        # Quick match at low resolution
        low_res_locations = []
        try:
            quick_result = cv2.matchTemplate(small_gray, small_template, cv2.TM_CCOEFF_NORMED)
            # Use a lower threshold for quick scan to avoid missing potential matches
            quick_locations = np.where(quick_result >= max(0.6, threshold - 0.15))
            low_res_locations = list(zip(*quick_locations[::-1]))  # Convert to (x, y)
            
            # If no matches at low res, try a few rotations before giving up
            if not low_res_locations:
                for rotation in [90, 180, 270]:
                    rot_template_idx = next((i for i, t in enumerate(symbol_data["multi_scale"]) 
                                        if t["scale"] == 1.0 and t["rotation"] == rotation), None)
                    if rot_template_idx is None:
                        continue
                        
                    rot_template = symbol_data["multi_scale"][rot_template_idx]["image"]
                    small_rot_template = cv2.resize(rot_template, 
                                                (rot_template.shape[1]//downscale_factor, 
                                                rot_template.shape[0]//downscale_factor))
                    
                    quick_result = cv2.matchTemplate(small_gray, small_rot_template, cv2.TM_CCOEFF_NORMED)
                    quick_locations = np.where(quick_result >= max(0.6, threshold - 0.15))
                    rot_locations = list(zip(*quick_locations[::-1]))
                    low_res_locations.extend(rot_locations)
                    
                    if len(low_res_locations) >= 5:  # Found enough potential matches
                        break
            
            # If still no matches, return empty result
            if not low_res_locations:
                return []
                
        except cv2.error:
            # Skip on OpenCV errors (e.g., template larger than image)
            return []
        
        # Convert low-res locations to regions of interest in the original image
        rois = []
        for lx, ly in low_res_locations:
            # Calculate ROI in full resolution with padding
            roi_x = max(0, lx * downscale_factor - tw//2)
            roi_y = max(0, ly * downscale_factor - th//2)
            roi_w = min(w - roi_x, tw * 2)
            roi_h = min(h - roi_y, th * 2)
            
            # Add ROI if it's valid
            if roi_w > 0 and roi_h > 0:
                rois.append((roi_x, roi_y, roi_w, roi_h))
        
        # Merge overlapping ROIs
        merged_rois = []
        for roi in sorted(rois, key=lambda r: r[0] * w + r[1]):
            rx, ry, rw, rh = roi
            
            # Check if this ROI overlaps with any existing merged ROI
            overlap = False
            for i, (mx, my, mw, mh) in enumerate(merged_rois):
                # Calculate overlap
                ox = max(0, min(rx + rw, mx + mw) - max(rx, mx))
                oy = max(0, min(ry + rh, my + mh) - max(ry, my))
                
                if ox * oy > 0:  # There is overlap
                    # Merge the ROIs
                    new_x = min(rx, mx)
                    new_y = min(ry, my)
                    new_w = max(rx + rw, mx + mw) - new_x
                    new_h = max(ry + rh, my + mh) - new_y
                    
                    merged_rois[i] = (new_x, new_y, new_w, new_h)
                    overlap = True
                    break
            
            if not overlap:
                merged_rois.append(roi)
        
        # Second stage: Detailed matching in ROIs
        all_matches = []
        
        # Process each ROI
        for roi_x, roi_y, roi_w, roi_h in merged_rois:
            # Extract ROI from masked and gray images
            roi_masked = masked_img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            roi_gray = gray_img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # Calculate adaptive threshold for this ROI
            local_std = np.std(roi_gray)
            local_threshold = threshold
            
            # Lower threshold for low contrast regions
            if local_std < 30:
                local_threshold = max(0.65, local_threshold - 0.05)
            
            # Try each template variant
            for template_data in symbol_data["multi_scale"]:
                template = template_data["image"]
                scale = template_data["scale"]
                rotation = template_data["rotation"]
                
                # Skip if template is larger than ROI
                if template.shape[0] > roi_h or template.shape[1] > roi_w:
                    continue
                
                try:
                    # Match on preprocessed image (binary features)
                    result1 = cv2.matchTemplate(roi_masked, template, cv2.TM_CCOEFF_NORMED)
                    
                    # Match on original grayscale (intensity features)
                    result2 = cv2.matchTemplate(roi_gray, template, cv2.TM_CCOEFF_NORMED)
                    
                    # Combine results
                    result = np.maximum(result1, result2 * 0.9)
                    
                    # Find locations above threshold
                    locations = np.where(result >= local_threshold)
                    locations = list(zip(*locations[::-1]))  # Convert to (x, y)
                    
                    # Process matches
                    for loc in locations:
                        match_score = result[loc[1], loc[0]]
                        
                        # Convert to full image coordinates
                        x = loc[0] + roi_x
                        y = loc[1] + roi_y
                        width = template.shape[1]
                        height = template.shape[0]
                        
                        # Skip if on a text region (secondary check)
                        if text_mask is not None and y+height <= text_mask.shape[0] and x+width <= text_mask.shape[1]:
                            if np.mean(text_mask[y:y+height, x:x+width]) > 50:
                                continue
                        
                        # Validate match
                        if self._validate_match(original_img, x, y, width, height):
                            all_matches.append({
                                "x": int(x),
                                "y": int(y),
                                "width": int(width),
                                "height": int(height),
                                "scale": float(scale),
                                "rotation": int(rotation),
                                "confidence": float(match_score),
                                "page": page_num,
                                "method": "template"
                            })
                except cv2.error:
                    continue
        
        # Cluster matches to remove duplicates
        return self._cluster_matches(all_matches)

    def _generate_region_density_map(self, img, text_mask):
        """
        Generate a density map of likely symbol regions.
        
        Args:
            img: Grayscale image
            text_mask: Binary mask of text regions
            
        Returns:
            Heatmap of potential symbol density
        """
        # Apply edge detection
        edges = cv2.Canny(img, 100, 200)
        
        # Remove edges in text regions
        edges[text_mask > 0] = 0
        
        # Dilate edges to connect nearby features
        dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        # Use distance transform to create a density map
        dist_transform = cv2.distanceTransform(dilated_edges, cv2.DIST_L2, 3)
        
        # Normalize density map
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        
        return dist_transform
    
    def _calculate_page_statistics(self, img, text_mask, page_num):
        """
        Calculate statistics for a page to guide parameter adjustment.
        
        Args:
            img: Grayscale image
            text_mask: Binary mask of text regions
            page_num: Page number
        """
        # Calculate text density
        text_density = np.count_nonzero(text_mask) / text_mask.size
        
        # Calculate edge density in non-text regions
        edges = cv2.Canny(img, 100, 200)
        non_text_mask = np.logical_not(text_mask.astype(bool)).astype(np.uint8) * 255
        non_text_edges = cv2.bitwise_and(edges, edges, mask=non_text_mask)
        edge_density = np.count_nonzero(non_text_edges) / max(1, np.count_nonzero(non_text_mask))
        
        # Calculate overall contrast
        contrast = np.std(img)
        
        # Store statistics
        self.context['region_statistics'][page_num] = {
            'text_density': text_density,
            'edge_density': edge_density,
            'contrast': contrast
        }
    
    def _process_page(self, page_data):
        """
        Process a single page for symbol detection.
        
        Args:
            page_data: Tuple containing (masked_img, original_img, gray_img, text_mask, page_num)
            
        Returns:
            Tuple of (page_num, page_results)
        """
        masked_img, original_img, gray_img, text_mask, page_num = page_data
        page_results = {}
        
        # Create visualization image
        vis_img = original_img.copy()
        
        # Get density map for this page
        density_map = self.context.get(f'density_map_page_{page_num}', None)
        
        # Initialize for each symbol
        for symbol_name in self.symbols:
            page_results[symbol_name] = {
                "present": False,
                "occurrences": 0,
                "positions": []
            }
        
        # Create tasks for multithreaded processing
        symbol_tasks = []
        for symbol_name, symbol_data in self.symbols.items():
            # Use the symbol's specific threshold
            threshold = symbol_data.get('match_threshold', self.params['base_match_threshold'])
            
            symbol_tasks.append((
                symbol_name,
                symbol_data,
                masked_img,
                gray_img,
                original_img,
                text_mask,
                density_map,
                page_num,
                threshold
            ))
        
        # Process symbols in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            symbol_results = list(executor.map(self._detect_symbol, symbol_tasks))
        
        # Merge results
        for symbol_name, detections in symbol_results:
            if detections:
                page_results[symbol_name]["present"] = True
                page_results[symbol_name]["occurrences"] = len(detections)
                page_results[symbol_name]["positions"] = detections
                
                # Draw on visualization image
                for match in detections:
                    x, y = match["x"], match["y"]
                    width, height = match["width"], match["height"]
                    confidence = match["confidence"]
                    method = match["method"]
                    
                    # Color based on confidence and method
                    if method == "template":
                        if confidence > 0.85:
                            color = (0, 255, 0)  # Green for high confidence
                        elif confidence > 0.75:
                            color = (0, 255, 255)  # Yellow for medium confidence
                        else:
                            color = (0, 165, 255)  # Orange for lower confidence
                    else:  # feature matching
                        color = (255, 0, 0)  # Red for feature-based matches
                    
                    # Draw rectangle and label
                    cv2.rectangle(vis_img, (x, y), (x + width, y + height), color, 2)
                    label_text = f"{symbol_name[:15]}... {confidence:.2f}"
                    cv2.putText(vis_img, label_text, (x, y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Save visualization
        vis_path = os.path.join(self.output_dir, f"page_{page_num+1}_detection.png")
        cv2.imwrite(vis_path, vis_img)
        
        return page_num, page_results
    
    def _detect_with_features(self, symbol_name, symbol_data, gray_img, original_img, page_num):
        """
        Detect a symbol using feature matching.
        
        Args:
            symbol_name: Name of the symbol
            symbol_data: Symbol data
            gray_img: Grayscale image
            original_img: Original color image
            page_num: Page number
            
        Returns:
            List of detection dictionaries
        """
        # Skip if descriptor is empty
        if symbol_data["descriptors"] is None or len(symbol_data["descriptors"]) < 5:
            return []
        
        # Initialize feature detector
        orb = cv2.ORB_create(nfeatures=2000)
        
        # Detect features in page
        kp_page, desc_page = orb.detectAndCompute(gray_img, None)
        
        # Skip if no features found
        if desc_page is None or len(desc_page) < 10:
            return []
        
        # Initialize matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Match features
        try:
            matches = bf.knnMatch(np.float32(symbol_data["descriptors"]), 
                                 np.float32(desc_page), k=2)
        except cv2.error:
            return []
        
        # Filter matches using ratio test
        good_matches = []
        for match_group in matches:
            if len(match_group) >= 2:
                m, n = match_group
                if m.distance < self.params['feature_match_ratio'] * n.distance:
                    good_matches.append(m)
        
        # Skip if not enough good matches
        if len(good_matches) < self.params['min_feature_matches']:
            return []
        
        # Extract matching keypoints
        src_pts = np.float32([symbol_data["keypoints"][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_page[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Skip if homography not found
        if H is None:
            return []
        
        # Calculate confidence (ratio of inliers)
        confidence = np.sum(mask) / len(mask)
        
        # Skip if confidence too low
        if confidence < 0.6:
            return []
        
        # Transform symbol corners to find region in page
        h, w = symbol_data["gray"].shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        
        # Calculate bounding box
        x_coords = [pt[0][0] for pt in dst]
        y_coords = [pt[0][1] for pt in dst]
        
        min_x, max_x = int(min(x_coords)), int(max(x_coords))
        min_y, max_y = int(min(y_coords)), int(max(y_coords))
        
        # Validate detection size
        width = max_x - min_x
        height = max_y - min_y
        
        # Skip if size is unreasonable
        if width <= 0 or height <= 0 or width > gray_img.shape[1]//2 or height > gray_img.shape[0]//2:
            return []
        
        # Calculate estimated rotation
        top_edge = dst[3][0] - dst[0][0]  # Vector from top-left to top-right
        angle_rad = np.arctan2(top_edge[1], top_edge[0])
        angle_deg = np.degrees(angle_rad) % 360
        
        # Return single detection
        return [{
            "x": min_x,
            "y": min_y,
            "width": width,
            "height": height,
            "confidence": float(confidence),
            "method": "feature",
            "page": page_num,
            "rotation": float(angle_deg),
            "scale": 1.0  # Actual scale is captured in the width/height
        }]
    
    def _validate_match(self, img, x, y, w, h):
        """
        Validate a template match to filter out false positives.
        
        Args:
            img: Original image
            x, y, w, h: Bounding box coordinates
            
        Returns:
            True if match is valid, False otherwise
        """
        # Check boundaries
        if img is None:
            return False
            
        h_img, w_img = img.shape[:2]
        if x < 0 or y < 0 or x + w >= w_img or y + h >= h_img:
            return False
        
        try:
            # Extract region
            region = img[y:y+h, x:x+w]
            
            # Convert to grayscale if needed
            if len(img.shape) > 2:
                region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                region_gray = region
            
            # Check 1: Edge density
            edges = cv2.Canny(region_gray, 100, 200)
            edge_count = np.count_nonzero(edges)
            edge_density = edge_count / (w * h)
            
            if edge_density < self.params['validation_edge_threshold']:
                return False
            
            # Check 2: Contrast
            std_dev = np.std(region_gray)
            if std_dev < self.params['validation_contrast_threshold']:
                return False
            
            # Check 3: Region content (not mostly white or black)
            mean_value = np.mean(region_gray)
            if mean_value < 20 or mean_value > 235:
                return False
            
            return True
        except Exception:
            # If any error occurs during validation, reject the match
            return False
    
    def _cluster_matches(self, matches):
        """
        Cluster matches to remove duplicates.
        
        Args:
            matches: List of match dictionaries
            
        Returns:
            Filtered list of matches
        """
        if not matches:
            return []
            
        if len(matches) == 1:
            return matches
        
        try:
            # Extract positions for clustering
            positions = np.array([[m["x"], m["y"]] for m in matches])
            
            # Calculate average object size
            avg_width = np.mean([m["width"] for m in matches])
            avg_height = np.mean([m["height"] for m in matches])
            
            # Adaptive clustering distance based on object size
            clustering_distance = max(20, min(avg_width, avg_height) * 
                                   self.params['clustering_distance_factor'])
            
            # Apply clustering
            clustering = DBSCAN(
                eps=clustering_distance, 
                min_samples=self.params['min_cluster_samples']
            ).fit(positions)
            
            labels = clustering.labels_
            
            # Group matches by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(labels):
                clusters[label].append(matches[i])
            
            # Keep the best match from each cluster
            result = []
            for _, cluster_matches in clusters.items():
                best_match = max(cluster_matches, key=lambda x: x["confidence"])
                result.append(best_match)
            
            return result
            
        except Exception as e:
            # Fallback to simple non-maximum suppression
            sorted_matches = sorted(matches, key=lambda x: x["confidence"], reverse=True)
            result = []
            
            for match in sorted_matches:
                # Check if this match overlaps with any existing high-confidence match
                should_add = True
                for existing in result:
                    if self._calculate_iou(match, existing) > 0.3:
                        should_add = False
                        break
                
                if should_add:
                    result.append(match)
                    
                    # Limit to prevent too many matches
                    if len(result) >= 10:
                        break
                        
            return result
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union for two bounding boxes.
        
        Args:
            box1, box2: Dictionaries with x, y, width, height
            
        Returns:
            IoU value (0-1)
        """
        # Extract coordinates
        x1, y1, w1, h1 = box1["x"], box1["y"], box1["width"], box1["height"]
        x2, y2, w2, h2 = box2["x"], box2["y"], box2["width"], box2["height"]
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / max(union_area, 1e-6)
        
        return iou
    
    def _filter_overlapping_detections(self, detections):
        """
        Filter out overlapping detections within the same method.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
            
        # Group by method
        template_detections = [d for d in detections if d["method"] == "template"]
        feature_detections = [d for d in detections if d["method"] == "feature"]
        
        # Filter each group
        filtered_template = self._cluster_matches(template_detections)
        
        # For feature detections, just keep the highest confidence one
        if feature_detections:
            best_feature = max(feature_detections, key=lambda x: x["confidence"])
            filtered_feature = [best_feature]
        else:
            filtered_feature = []
        
        # Combine results, preferring template matches over feature matches
        # when they overlap significantly
        combined = filtered_template.copy()
        
        for feature_match in filtered_feature:
            # Check if this feature match overlaps with any template match
            has_overlap = False
            for template_match in filtered_template:
                if self._calculate_iou(feature_match, template_match) > 0.3:
                    has_overlap = True
                    break
            
            if not has_overlap:
                combined.append(feature_match)
        
        return combined
    
    def _update_results(self, page_results):
        """
        Update global results with page results.
        
        Args:
            page_results: Dictionary of results for a page
        """
        for symbol_name, data in page_results.items():
            if data["present"]:
                self.results[symbol_name]["present"] = True
                self.results[symbol_name]["occurrences"] += data["occurrences"]
                
                # Copy position data
                for pos in data["positions"]:
                    self.results[symbol_name]["positions"].append(pos.copy())
    
    def _post_process_results(self):
        """
        Apply post-processing to remove inconsistencies and false positives.
        """
        self._log("Post-processing detection results")
        
        # Filter symbols by confidence and consistency
        for symbol_name, data in self.results.items():
            if not data["present"] or len(data["positions"]) <= 1:
                continue
            
            # Extract confidence scores and sizes
            confidences = [pos["confidence"] for pos in data["positions"]]
            widths = [pos["width"] for pos in data["positions"]]
            heights = [pos["height"] for pos in data["positions"]]
            
            # Calculate statistics
            mean_conf = np.mean(confidences)
            median_width = np.median(widths)
            median_height = np.median(heights)
            
            # Filter positions
            filtered_positions = []
            for pos in data["positions"]:
                # Check confidence - remove significant outliers
                if pos["confidence"] < max(0.65, mean_conf * 0.7):
                    continue
                
                # Check size consistency 
                width_ratio = pos["width"] / median_width
                height_ratio = pos["height"] / median_height
                
                # Size should be within reasonable bounds of the median
                if width_ratio < 0.5 or width_ratio > 2.0 or height_ratio < 0.5 or height_ratio > 2.0:
                    continue
                
                # Passed all filters
                filtered_positions.append(pos)
            
            # Update results
            data["positions"] = filtered_positions
            data["occurrences"] = len(filtered_positions)
            data["present"] = len(filtered_positions) > 0
        
        self._log(f"After post-processing: {sum(1 for d in self.results.values() if d['present'])} symbols detected")
    
    def generate_report(self, output_json_path):
        """
        Generate comprehensive report with detection results.
        
        Args:
            output_json_path: Path to save JSON report
            
        Returns:
            Dictionary with detection results
        """
        # Create report structure
        report = {
            "summary": {
                "total_symbols_in_library": len(self.symbols),
                "total_symbols_detected": sum(1 for data in self.results.values() if data["present"]),
                "detection_statistics": {
                    "high_confidence_detections": sum(1 for data in self.results.values() 
                                                  for pos in data["positions"] if pos.get("confidence", 0) > 0.85),
                    "medium_confidence_detections": sum(1 for data in self.results.values() 
                                                    for pos in data["positions"] if 0.7 <= pos.get("confidence", 0) <= 0.85),
                    "low_confidence_detections": sum(1 for data in self.results.values() 
                                                 for pos in data["positions"] if pos.get("confidence", 0) < 0.7),
                    "rotated_symbols_count": sum(1 for data in self.results.values() 
                                              for pos in data["positions"] if pos.get("rotation", 0) != 0)
                }
            },
            "symbols": {},
            "detection_methods": {
                "template_matching": sum(1 for data in self.results.values() 
                                      for pos in data["positions"] if pos.get("method", "") == "template"),
                "feature_matching": sum(1 for data in self.results.values() 
                                     for pos in data["positions"] if pos.get("method", "") == "feature")
            },
            "adaptive_parameters": {
                "base_match_threshold": self.params['base_match_threshold'],
                "validation_edge_threshold": self.params['validation_edge_threshold'],
                "validation_contrast_threshold": self.params['validation_contrast_threshold'],
                "clustering_distance_factor": self.params['clustering_distance_factor']
            }
        }
        
        # Group detected symbols by confidence level
        confidence_groups = {
            "high_confidence": [],
            "medium_confidence": [],
            "low_confidence": []
        }
        
        # Add detailed symbol information
        for symbol_name, data in self.results.items():
            if data["present"]:
                # Calculate average confidence
                avg_confidence = sum(pos.get("confidence", 0) for pos in data["positions"]) / len(data["positions"])
                
                # Determine confidence group
                if avg_confidence > 0.85:
                    confidence_group = "high_confidence"
                elif avg_confidence > 0.7:
                    confidence_group = "medium_confidence"
                else:
                    confidence_group = "low_confidence"
                
                confidence_groups[confidence_group].append({
                    "name": symbol_name,
                    "avg_confidence": avg_confidence,
                    "occurrences": data["occurrences"]
                })
                
                # Count by detection method
                template_count = sum(1 for pos in data["positions"] if pos.get("method", "") == "template")
                feature_count = sum(1 for pos in data["positions"] if pos.get("method", "") == "feature")
                
                # Count rotated instances
                rotated_count = sum(1 for pos in data["positions"] if pos.get("rotation", 0) != 0)
                
                report["symbols"][symbol_name] = {
                    "present": data["present"],
                    "occurrences": data["occurrences"],
                    "avg_confidence": float(avg_confidence),
                    "detection_methods": {
                        "template": template_count,
                        "feature": feature_count
                    },
                    "rotated_instances": rotated_count,
                    "positions": data["positions"]
                }
        
        # Add confidence groups to results
        report["confidence_groups"] = confidence_groups
        
        # Write to file
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4)
            
        print(f"Results saved to {output_json_path}")
        self._log(f"Results saved to {output_json_path}")
        
        # Generate summary visualizations
        self._generate_summary_visualizations(output_json_path.replace('.json', '_charts'))
        
        return report
    
    def _generate_summary_visualizations(self, output_base_path):
        """
        Generate summary visualizations of detection results.
        
        Args:
            output_base_path: Base path for output files
        """
        # Create charts directory
        charts_dir = os.path.dirname(output_base_path)
        os.makedirs(charts_dir, exist_ok=True)
        
        # 1. Top detected symbols chart
        self._generate_top_symbols_chart(f"{output_base_path}_top_symbols.png")
        
        # 2. Detection methods comparison
        self._generate_methods_chart(f"{output_base_path}_methods.png")
        
        # 3. Confidence distribution
        self._generate_confidence_chart(f"{output_base_path}_confidence.png")
    
    def _generate_top_symbols_chart(self, output_path):
        """
        Generate chart showing top detected symbols.
        
        Args:
            output_path: Path to save chart
        """
        # Get symbols with occurrences
        present_symbols = [(name, data["occurrences"]) 
                          for name, data in self.results.items() 
                          if data["present"]]
        
        # Sort by occurrence count
        present_symbols.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to top 20 for readability
        if len(present_symbols) > 20:
            present_symbols = present_symbols[:20]
            chart_title = "Top 20 Detected Symbols"
        else:
            chart_title = "Detected Symbols"
        
        # Extract names and counts
        names = [item[0] for item in present_symbols]
        counts = [item[1] for item in present_symbols]
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        bars = plt.bar(names, counts)
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02 * max(counts),
                    f'{height}', ha='center', va='bottom')
        
        plt.title(chart_title)
        plt.xlabel("Symbol Name")
        plt.ylabel("Number of Occurrences")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save chart
        plt.savefig(output_path)
        plt.close()
        
        self._log(f"Top symbols chart saved to {output_path}")
    
    def _generate_methods_chart(self, output_path):
        """
        Generate chart showing distribution of detection methods.
        
        Args:
            output_path: Path to save chart
        """
        # Count detections by method
        template_count = sum(1 for data in self.results.values() 
                          for pos in data["positions"] if pos.get("method", "") == "template")
        feature_count = sum(1 for data in self.results.values() 
                         for pos in data["positions"] if pos.get("method", "") == "feature")
        
        # Skip if no data
        if template_count == 0 and feature_count == 0:
            self._log(f"Skipping methods chart - no detection data")
            return
        
        # Create pie chart
        plt.figure(figsize=(8, 8))
        
        # Prepare data
        labels = []
        sizes = []
        colors = []
        explode = []
        
        if template_count > 0:
            labels.append('Template Matching')
            sizes.append(template_count)
            colors.append('#ff9999')
            explode.append(0.1)
            
        if feature_count > 0:
            labels.append('Feature Matching')
            sizes.append(feature_count)
            colors.append('#66b3ff')
            explode.append(0.0)
        
        # Create chart if data exists
        if sizes:
            plt.pie(sizes, explode=tuple(explode), labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
            plt.axis('equal')  # Equal aspect ratio
            plt.title('Symbol Detection Methods Distribution')
            
            # Save chart
            plt.savefig(output_path)
            plt.close()
            
            self._log(f"Methods chart saved to {output_path}")
    
    def _generate_confidence_chart(self, output_path):
        """
        Generate chart showing confidence distribution.
        
        Args:
            output_path: Path to save chart
        """
        # Extract all confidence values
        confidences = [pos["confidence"] for data in self.results.values() 
                     for pos in data["positions"]]
        
        # Skip if no data
        if not confidences:
            self._log(f"Skipping confidence chart - no detection data")
            return
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist(confidences, bins=20, alpha=0.7, color='skyblue')
        
        # Add a vertical line at the mean
        mean_conf = np.mean(confidences)
        plt.axvline(mean_conf, color='red', linestyle='dashed', linewidth=1)
        plt.text(mean_conf + 0.01, max(n) * 0.9, f'Mean: {mean_conf:.2f}', 
               color='red', ha='left', va='center')
        
        plt.title('Detection Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Number of Detections')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path)
        plt.close()
        
        self._log(f"Confidence chart saved to {output_path}")
    
    def generate_simplified_report(self, output_json_path):
        """
        Generate a simplified report with only basic detection information.
        
        Args:
            output_json_path: Path to save JSON report
            
        Returns:
            Dictionary with simplified results
        """
        # Create simplified results
        simplified_results = {}
        
        # Include only symbols that are present
        for symbol_name, data in self.results.items():
            if data["present"]:
                simplified_results[symbol_name] = {
                    "present": True,
                    "occurrences": data["occurrences"],
                    "pages": sorted(set(pos["page"] for pos in data["positions"]))
                }
        
        # Write to file
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, indent=4)
            
        print(f"Simplified results saved to {output_json_path}")
        self._log(f"Simplified results saved to {output_json_path}")
        
        return simplified_results


# Main execution
if __name__ == "__main__":
    # Define paths
    symbols_dir = "output_images/symbols"
    pnid_pdf_path = "data/input/pnid.pdf"
    output_dir = "detection_output"
    
    # Create detector instance
    detector = AdaptivePIDSymbolDetector(symbols_dir, output_dir)
    
    # Create and run the async event loop
    async def main():
        results = await detector.detect_symbols_in_pdf(
            pnid_pdf_path, 
            match_threshold=1.0,
            use_parallel=True,
            dpi=150
        )
        
        # Generate reports
        output_json_path = os.path.join(output_dir, "symbol_detection_results.json")
        detector.generate_report(output_json_path)
        
        simplified_json_path = os.path.join(output_dir, "simplified_results.json")
        detector.generate_simplified_report(simplified_json_path)
    
    # Run the async main function
    import asyncio
    asyncio.run(main())