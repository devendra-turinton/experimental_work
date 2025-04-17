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

class EnhancedPIDSymbolDetector:

    def __init__(self, symbols_dir, output_dir="detection_output", optimization_level=2, max_workers=None, use_gpu = False):
        self.symbols_dir = symbols_dir
        self.output_dir = output_dir
        self.optimization_level = optimization_level
        self.max_workers = max_workers or os.cpu_count()
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0  # Check if GPU is actually available
        self.symbols = self._load_symbols()
        self.results = {}
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def _validate_match(self, img, x, y, w, h):
        """Validate a match to filter out false positives."""
        if img is None:
            return False
            
        # Check boundaries
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
            
            # Check 1: Edge density - symbols should have a minimum amount of edges
            edges = cv2.Canny(region_gray, 100, 200)
            edge_count = np.count_nonzero(edges)
            edge_density = edge_count / (w * h)
            
            if edge_density < 0.05:  # Minimum edge density threshold
                return False
            
            # Check 2: Contrast - symbols should have good contrast
            std_dev = np.std(region_gray)
            if std_dev < 20:  # Low standard deviation means low contrast
                return False
            
            return True
        except Exception:
            # If any error occurs during validation, reject the match
            return False

    def _generate_text_exclusion_mask(self, img):
        """Generate a mask of likely text regions to exclude from detection."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Create horizontal kernel to detect text lines
        h_kernel = np.ones((1, 15), np.uint8)
        h_text = cv2.morphologyEx(gray, cv2.MORPH_OPEN, h_kernel)
        _, h_text = cv2.threshold(h_text, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Create small vertical kernel for vertical text
        v_kernel = np.ones((10, 1), np.uint8)
        v_text = cv2.morphologyEx(gray, cv2.MORPH_OPEN, v_kernel)
        _, v_text = cv2.threshold(v_text, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Combine horizontal and vertical text detection
        text_mask = cv2.bitwise_or(h_text, v_text)
        
        # Dilate to create exclusion zones around text
        text_mask = cv2.dilate(text_mask, np.ones((5, 5), np.uint8), iterations=1)
        
        return text_mask

    def _process_page_enhanced(self, page_data):
        """Process a page with enhanced detection capabilities."""
        denoised, original_img, page_num, match_threshold = page_data
        page_results = {}
        
        # Create visualization image
        vis_img = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        # Apply preprocessing
        processed_img = self._preprocess_image(denoised)

        # Also try feature matching if optimization level allows
        if self.optimization_level < 3 and getattr(self, 'use_feature_detection', True):
            gray_orig = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) if len(original_img.shape) > 2 else original_img
            self._detect_with_features(processed_img, gray_orig, page_num, page_results, vis_img)
        
        
        # Detect with templates
        symbol_tasks = []
        for symbol_name, symbol_data in self.symbols.items():
            if symbol_name not in page_results:
                page_results[symbol_name] = {
                    "present": False,
                    "occurrences": 0,
                    "positions": []
                }
            
            # Create task for processing
            symbol_tasks.append((
                symbol_name, 
                symbol_data, 
                processed_img, 
                original_img, 
                page_num, 
                match_threshold, 
                10  # max_matches
            ))
        
        # Process symbols in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            symbol_results = list(executor.map(self._process_symbol_templates, symbol_tasks))
        
        # Merge results
        for symbol_name, matches in symbol_results:
            if matches:
                page_results[symbol_name]["present"] = True
                page_results[symbol_name]["occurrences"] += len(matches)
                page_results[symbol_name]["positions"].extend(matches)
                
                # Draw on visualization image
                for match in matches:
                    x, y = match["x"], match["y"]
                    width, height = match["width"], match["height"]
                    score = match["confidence"]
                    
                    # Choose color based on confidence
                    if score > 0.9:
                        color = (0, 255, 0)  # Green for high confidence
                    elif score > 0.8:
                        color = (0, 255, 255)  # Yellow for medium confidence
                    else:
                        color = (0, 165, 255)  # Orange for lower confidence
                    
                    # Draw on visualization image
                    cv2.rectangle(vis_img, (x, y), (x + width, y + height), color, 2)
                    label_text = f"{symbol_name[:12]}... {score:.2f}"
                    cv2.putText(vis_img, label_text, (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Save visualization
        vis_path = os.path.join(self.output_dir, f"page_{page_num+1}_enhanced_detection.png")
        cv2.imwrite(vis_path, vis_img)
        
        return page_num, page_results

    def _load_symbols(self):
        """Load all symbols with expanded template variants."""
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
                
                # Define expanded rotations and scales
                if self.optimization_level <= 2:
                    # More comprehensive for better detection
                    scales = [0.8, 0.9, 1.0, 1.1, 1.2]
                    rotations = [0, 45, 90, 135, 180, 225, 270, 315]
                else:
                    # Original settings
                    scales = [0.9, 1.0, 1.1]
                    rotations = [0, 90, 180, 270]
                    
                multi_scale_templates = []
                
                for scale in scales:
                    width = int(symbol_gray.shape[1] * scale)
                    height = int(symbol_gray.shape[0] * scale)
                    if width > 0 and height > 0:
                        resized = cv2.resize(symbol_gray, (width, height), interpolation=cv2.INTER_AREA)
                        
                        # Generate rotated versions
                        for angle in rotations:
                            if angle == 0:
                                rotated = resized
                            else:
                                center = (width // 2, height // 2)
                                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                                rotated = cv2.warpAffine(resized, rotation_matrix, (width, height), 
                                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                            
                            multi_scale_templates.append({
                                "image": rotated,
                                "scale": scale,
                                "rotation": angle
                            })
                
                # Precompute features
                orb = cv2.ORB_create(nfeatures=500)
                keypoints, descriptors = orb.detectAndCompute(symbol_gray, None)
                
                symbols[symbol_name] = {
                    "name": symbol_name, 
                    "path": path,
                    "image": symbol_img,
                    "gray": symbol_gray,
                    "multi_scale": multi_scale_templates,
                    "keypoints": keypoints,
                    "descriptors": descriptors
                }
        
        print(f"Loaded {len(symbols)} symbols from {self.symbols_dir}")
        return symbols

    def _preprocess_image(self, img):
        """Apply faster preprocessing to improve symbol detection."""
        # OPTIMIZATION: Faster contrast enhancement
        enhanced = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        # OPTIMIZATION: Use median blur instead of bilateral filter (much faster)
        denoised = cv2.medianBlur(enhanced, 5)
        
        # OPTIMIZATION: For aggressive optimization, skip adaptive threshold
        if self.optimization_level >= 3:
            # Fast global thresholding
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # Still use adaptive thresholding but with larger block size for speed
            thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 15, 2)
        
        # OPTIMIZATION: Simplified morphological operations
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return cleaned

    def _filter_inconsistent_detections(self):
        """Filter out inconsistent detections in post-processing."""
        # For each symbol, check for consistency across detections
        for symbol_name, data in self.results.items():
            if not data["present"] or len(data["positions"]) <= 1:
                continue
            
            # Calculate statistics
            confidences = [pos["confidence"] for pos in data["positions"]]
            widths = [pos["width"] for pos in data["positions"]]
            heights = [pos["height"] for pos in data["positions"]]
            
            median_width = np.median(widths)
            median_height = np.median(heights)
            median_conf = np.median(confidences)
            
            # Filter positions
            filtered_positions = []
            for pos in data["positions"]:
                # Check size consistency
                w_ratio = pos["width"] / median_width
                h_ratio = pos["height"] / median_height
                
                # Too different from median size or too low confidence
                if (w_ratio < 0.5 or w_ratio > 2.0 or 
                    h_ratio < 0.5 or h_ratio > 2.0 or 
                    pos["confidence"] < median_conf * 0.8):
                    continue
                    
                filtered_positions.append(pos)
            
            # Update results
            data["positions"] = filtered_positions
            data["occurrences"] = len(filtered_positions)
            data["present"] = len(filtered_positions) > 0

    def _check_overlap(self, pos1, pos2, overlap_threshold=0.5):
        """Check if two detection positions overlap significantly."""
        # Calculate box coordinates
        x1, y1, w1, h1 = pos1["x"], pos1["y"], pos1["width"], pos1["height"]
        x2, y2, w2, h2 = pos2["x"], pos2["y"], pos2["width"], pos2["height"]
        
        # Find overlap rectangle
        overlap_x = max(0, min(x1+w1, x2+w2) - max(x1, x2))
        overlap_y = max(0, min(y1+h1, y2+h2) - max(y1, y2))
        overlap_area = overlap_x * overlap_y
        
        # Calculate areas
        area1 = w1 * h1
        area2 = w2 * h2
        
        # Check if overlap exceeds threshold
        return overlap_area > overlap_threshold * min(area1, area2)

    def _filter_false_positives(self, detected_region, original_img):
        """Filter out likely false positives based on image analysis."""
        x, y, w, h = detected_region["x"], detected_region["y"], detected_region["width"], detected_region["height"]
        
        # Extract the region
        if y+h >= original_img.shape[0] or x+w >= original_img.shape[1]:
            return False  # Outside image bounds
            
        region = original_img[y:y+h, x:x+w]
        if region.size == 0:
            return False
            
        # Convert to grayscale
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Check 1: Region density
        _, binary = cv2.threshold(gray_region, 200, 255, cv2.THRESH_BINARY_INV)
        density = np.count_nonzero(binary) / (w * h)
        if density < 0.1:
            return False  # Too sparse
            
        # Check 2: Edge density
        edges = cv2.Canny(gray_region, 100, 200)
        edge_density = np.count_nonzero(edges) / (w * h)
        if edge_density < 0.05:
            return False  # Too few edges
        
        # Passed all checks
        return True

    def _process_symbol_templates(self, task_data):
        """Process a single symbol using template matching with validation."""
        symbol_name, symbol_data, processed_img, original_img, page_num, match_threshold, max_matches = task_data
        
        # Try template matching with expanded scales and rotations
        potential_matches = []
        
        # Define expanded rotations based on optimization level
        if self.optimization_level <= 2:
            rotations = [0, 45, 90, 135, 180, 225, 270, 315]
            scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        else:
            rotations = [0, 90, 180, 270]
            scales = [0.9, 1.0, 1.1]
        
        # Ensure processed_img is grayscale
        if len(processed_img.shape) > 2:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        
        # Ensure we have a valid grayscale version of original_img
        if original_img is not None:
            if len(original_img.shape) > 2:
                original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            else:
                original_gray = original_img
        else:
            original_gray = processed_img
        
        # Track number of matches to limit processing
        total_matches = 0
        
        # Process each template variant
        for template_data in symbol_data["multi_scale"]:
            # Break if we've found enough matches
            if total_matches >= max_matches * 2:
                break
                
            template = template_data["image"]
            scale = template_data["scale"]
            rotation = template_data["rotation"]
            
            # Skip if template is larger than the image
            if template.shape[0] >= processed_img.shape[0] or template.shape[1] >= processed_img.shape[1]:
                continue
            
            # Apply template matching - faster approach with fewer checks
            try:
                # Only match against processed image for speed
                result = cv2.matchTemplate(processed_img, template, cv2.TM_CCOEFF_NORMED)
                
                # Find locations exceeding threshold
                locations = np.where(result >= match_threshold)
                locations = list(zip(*locations[::-1]))
                
                # Only process a limited number of locations to improve speed
                max_locs_per_template = 5
                if len(locations) > max_locs_per_template:
                    # Get scores for top locations
                    scores = [result[loc[1], loc[0]] for loc in locations]
                    # Get indices of top N scores
                    top_indices = np.argsort(scores)[-max_locs_per_template:]
                    locations = [locations[i] for i in top_indices]
                
                # Add matches and update counter
                for loc in locations:
                    match_score = result[loc[1], loc[0]]
                    potential_matches.append({
                        "position": loc, 
                        "score": match_score,
                        "scale": scale, 
                        "rotation": rotation,
                        "template_width": template.shape[1],
                        "template_height": template.shape[0]
                    })
                    total_matches += 1
                    
            except cv2.error:
                continue
        
        # If matches found, cluster them - simplified for speed
        results = []
        if potential_matches:
            # Extract positions for clustering
            positions = np.array([m["position"] for m in potential_matches])
            
            # Skip clustering if only one match
            if len(positions) == 1:
                match = potential_matches[0]
                x, y = match["position"]
                width = match["template_width"]
                height = match["template_height"]
                
                # Simple validation for speed
                if self._validate_match(original_img, x, y, width, height):
                    results.append({
                        "page": page_num,
                        "x": int(x),
                        "y": int(y),
                        "width": int(width),
                        "height": int(height),
                        "confidence": float(match["score"]),
                        "scale": float(match["scale"]),
                        "rotation": int(match["rotation"]),
                        "method": "template"
                    })
            else:
                try:
                    # Adaptive clustering distance
                    clustering_distance = 20  # Fixed value for speed
                    if len(potential_matches) > 0:
                        median_width = np.median([m["template_width"] for m in potential_matches])
                        median_height = np.median([m["template_height"] for m in potential_matches])
                        clustering_distance = max(20, min(median_width, median_height) * 0.4)
                    
                    # Apply clustering
                    clustering = DBSCAN(eps=clustering_distance, min_samples=1).fit(positions)
                    labels = clustering.labels_
                    
                    # Group matches by cluster
                    clusters = {}
                    for i, label in enumerate(labels):
                        if label not in clusters:
                            clusters[label] = []
                        clusters[label].append(potential_matches[i])
                    
                    # For each cluster, keep only the best match
                    for cluster_id, matches in clusters.items():
                        best_match = max(matches, key=lambda x: x["score"])
                        
                        x, y = best_match["position"]
                        width = best_match["template_width"]
                        height = best_match["template_height"]
                        
                        # Validate match
                        if self._validate_match(original_img, x, y, width, height):
                            results.append({
                                "page": page_num,
                                "x": int(x),
                                "y": int(y),
                                "width": int(width),
                                "height": int(height),
                                "confidence": float(best_match["score"]),
                                "scale": float(best_match["scale"]),
                                "rotation": int(best_match["rotation"]),
                                "method": "template"
                            })
                except Exception as e:
                    # Fast fallback if clustering fails
                    top_matches = sorted(potential_matches, key=lambda x: x["score"], reverse=True)[:3]  # Limit to 3 for speed
                    for match in top_matches:
                        x, y = match["position"]
                        width = match["template_width"]
                        height = match["template_height"]
                        
                        # Validate match
                        if self._validate_match(original_img, x, y, width, height):
                            results.append({
                                "page": page_num,
                                "x": int(x),
                                "y": int(y),
                                "width": int(width),
                                "height": int(height),
                                "confidence": float(match["score"]),
                                "scale": float(match["scale"]),
                                "rotation": int(match["rotation"]),
                                "method": "template"
                            })
        
        return symbol_name, results

    def _quick_reject(self, template_hist, region_hist, threshold=0.5):
        """Fast histogram comparison to reject unlikely matches."""
        # Compare histograms using correlation method
        similarity = cv2.compareHist(template_hist, region_hist, cv2.HISTCMP_CORREL)
        return similarity < threshold  # Return True if should reject  

    def _detect_with_features(self, processed_img, original_img, page_num, page_results, vis_img):
        """Detect symbols using feature matching (rotation and scale invariant)."""
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=10000, scaleFactor=1.2)
        
        # Detect features in the page image
        kp_page, desc_page = orb.detectAndCompute(original_img, None)
        
        # Skip if no features found in page
        if desc_page is None or len(desc_page) < 10:
            return
        
        # Create FLANN-based matcher (faster for large images)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                           table_number=6,  # 12
                           key_size=12,     # 20
                           multi_probe_level=1)  # 2
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # For each symbol in our library
        for symbol_name, symbol_data in self.symbols.items():
            # Skip if no descriptors available
            if symbol_data["descriptors"] is None or len(symbol_data["descriptors"]) < 5:
                continue
            
            # Initialize or get existing result entry
            if symbol_name not in page_results:
                page_results[symbol_name] = {
                    "present": False,
                    #"occurrences": 0,
                    "positions": []
                }
            
            # Match descriptors
            try:
                # Convert to correct format if necessary
                desc_page_float = np.float32(desc_page)
                desc_symbol_float = np.float32(symbol_data["descriptors"])
                
                # Match features
                matches = flann.knnMatch(desc_symbol_float, desc_page_float, k=2)
            except Exception as e:
                # If matching fails, skip this symbol
                continue
            
            # Apply ratio test to filter good matches
            good_matches = []
            for match_group in matches:
                if len(match_group) >= 2:
                    m, n = match_group
                    if m.distance < 0.7 * n.distance:  # Ratio test
                        good_matches.append(m)
            
            # Need minimum number of good matches
            min_match_count = 10
            if len(good_matches) >= min_match_count:
                # Extract locations of matched keypoints
                src_pts = np.float32([symbol_data["keypoints"][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_page[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                # Only proceed if homography was found
                if H is not None:
                    # Get symbol dimensions
                    h, w = symbol_data["gray"].shape
                    
                    # Define corners of symbol image
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    
                    # Transform corners to find symbol region in page
                    dst = cv2.perspectiveTransform(pts, H)
                    
                    # Calculate bounding box
                    x_coords = [pt[0][0] for pt in dst]
                    y_coords = [pt[0][1] for pt in dst]
                    
                    min_x, max_x = int(min(x_coords)), int(max(x_coords))
                    min_y, max_y = int(min(y_coords)), int(max(y_coords))
                    
                    # Calculate confidence score (ratio of inliers)
                    confidence = np.sum(mask) / len(mask)
                    
                    # Add to results if confidence is high enough
                    if confidence > 0.5:
                        # Add match to results
                        page_results[symbol_name]["present"] = True
                        #page_results[symbol_name]["occurrences"] += 1
                        
                        # Calculate estimated rotation (this is approximate)
                        # Get top edge vector from the transformed rectangle
                        top_edge = dst[3][0] - dst[0][0]  # Vector from top-left to top-right
                        angle_rad = np.arctan2(top_edge[1], top_edge[0])
                        angle_deg = np.degrees(angle_rad)
                        
                        # Add position to results
                        page_results[symbol_name]["positions"].append({
                            "page": page_num,
                            "x": min_x,
                            "y": min_y,
                            "width": max_x - min_x,
                            "height": max_y - min_y,
                            "confidence": float(confidence),
                            "method": "feature",
                            "rotation": float(angle_deg)
                        })
                        
                        # Draw contour and label on visualization image
                        color = (255, 0, 0)  # Red for feature-based matches
                        cv2.polylines(vis_img, [np.int32(dst)], True, color, 2)
                        cv2.putText(vis_img, f"F:{symbol_name[:8]}...", (min_x, min_y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
     
    def _calculate_adaptive_threshold(self, symbol_data, base_threshold=0.8):
        """Calculate adaptive match threshold based on symbol complexity."""
        symbol_img = symbol_data["gray"]
        
        # Calculate edge density
        edges = cv2.Canny(symbol_img, 100, 200)
        edge_density = np.count_nonzero(edges) / symbol_img.size
        
        # Calculate contour complexity
        _, binary = cv2.threshold(symbol_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            complexity = perimeter**2 / (4 * np.pi * area) if area > 0 else 1.0
        else:
            complexity = 1.0
        
        # Simple symbols need higher threshold, complex symbols can use lower threshold
        adjustment = min(0.1, edge_density * 0.5) + min(0.05, (complexity - 1.0) * 0.05)
        
        return max(0.7, base_threshold - adjustment)

    def detect_symbols_in_pdf(self, pdf_path, match_threshold=0.80, use_parallel=True, dpi=150, use_feature_detection=True):
        """Detect symbols with enhanced approach for better accuracy."""
        start_time = time.time()
        self.use_feature_detection = use_feature_detection

        
        # Initialize results dictionary
        for symbol_name in self.symbols:
            self.results[symbol_name] = {
                "present": False,
                "occurrences": 0,
                "positions": []
            }
        
        # Two-phase detection
        doc = fitz.open(pdf_path)
        
        # First phase: Generate text exclusion masks in parallel
        text_mask_tasks = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            mask_dpi = int(dpi/4)  # Lower resolution for mask
            pix = page.get_pixmap(matrix=fitz.Matrix(mask_dpi/72, mask_dpi/72))
            
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 3 or pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            text_mask_tasks.append((img, page_num))

        # Process text masks in parallel
        text_masks = {}
        if use_parallel and len(text_mask_tasks) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                mask_results = list(executor.map(lambda x: (x[1], self._generate_text_exclusion_mask(x[0])), text_mask_tasks))
                for page_num, mask in mask_results:
                    text_masks[page_num] = mask
        else:
            for img, page_num in text_mask_tasks:
                text_masks[page_num] = self._generate_text_exclusion_mask(img)
        
        # Second phase: Normal detection with text exclusion
        pages_to_process = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            
            # Convert pixmap to numpy array
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            if pix.n == 3 or pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale and denoise
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.medianBlur(gray_img, 3)
            
            # Scale up text mask to match current resolution
            text_mask = cv2.resize(text_masks[page_num], (denoised.shape[1], denoised.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            # Apply text mask to denoised image (set text areas to white/background)
            denoised_no_text = denoised.copy()
            denoised_no_text[text_mask > 0] = 255
            
            # Add both versions to processing queue
            pages_to_process.append((denoised_no_text, img, page_num, match_threshold))
        
        doc.close()
        
        # Process pages in parallel
        if use_parallel and len(pages_to_process) > 1:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(self._process_page_enhanced, pages_to_process))
                
            # Combine results
            for page_num, page_results in results:
                self._update_results(page_results)
        else:
            # Process sequentially
            for page_data in pages_to_process:
                page_num, page_results = self._process_page_enhanced(page_data)
                self._update_results(page_results)
        
        # Filter out inconsistent detections in post-processing
        self._filter_inconsistent_detections()
        
        elapsed_time = time.time() - start_time
        print(f"Enhanced processing completed in {elapsed_time:.2f} seconds")
        
        return self.results

    def _update_results(self, page_results):
        """Update global results with page results."""
        for symbol_name, data in page_results.items():
            if data["present"]:
                self.results[symbol_name]["present"] = True
                self.results[symbol_name]["occurrences"] += data["occurrences"]
                self.results[symbol_name]["positions"].extend(data["positions"])
    
    def generate_report(self, output_json_path):
        """Generate and save enhanced detection report with additional statistics."""
        # Create a comprehensive version of the results for JSON
        json_results = {
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
                                      for pos in data["positions"] if pos.get("method", "template") == "template"),
                "feature_matching": sum(1 for data in self.results.values() 
                                     for pos in data["positions"] if pos.get("method", "template") == "feature")
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
                    "avg_confidence": avg_confidence
                })
                
                # Count rotated instances
                rotated_count = sum(1 for pos in data["positions"] if pos.get("rotation", 0) != 0)
                
                json_results["symbols"][symbol_name] = {
                    "present": data["present"],
                    "occurrences": data["occurrences"],
                    "avg_confidence": avg_confidence,
                    "rotated_instances": rotated_count,
                    "positions": data["positions"]
                }
        
        # Add confidence groups to results
        json_results["confidence_groups"] = confidence_groups
        
        # Write to file
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=4)
            
        print(f"Results saved to {output_json_path}")
        
        # Generate summary visualizations
        self._generate_summary_charts(output_json_path.replace('.json', '_charts'))
        
        return json_results
    
    def _quick_scan_pdf(self, pdf_path, dpi, match_threshold):
        """Perform a quick scan of PDF at low resolution to identify pages with potential symbols."""
        print(f"Performing quick scan at {dpi} DPI...")
        quick_start = time.time()
        
        # Open PDF at low resolution
        doc = fitz.open(pdf_path)
        potential_pages = set()
        
        # Use a subset of symbols for quicker scanning
        subset_symbols = {}
        symbol_count = len(self.symbols)
        sample_size = min(5, symbol_count)  # Use at most 5 symbols for scanning
        
        # Take every nth symbol to get a representative sample
        step = max(1, symbol_count // sample_size)
        symbol_names = list(self.symbols.keys())
        
        for i in range(0, symbol_count, step):
            if i < symbol_count:
                name = symbol_names[i]
                subset_symbols[name] = self.symbols[name]
        
        # Scan each page with simplified detection
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            if pix.n == 3 or pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Simple preprocessing
            _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Check for potential matches with simplified template matching
            page_has_matches = False
            
            for symbol_name, symbol_data in subset_symbols.items():
                # Use only the original scale/rotation for quick check
                template = symbol_data["gray"]
                
                # Skip if template is larger than the image
                if template.shape[0] > gray_img.shape[0] or template.shape[1] > gray_img.shape[1]:
                    continue
                
                # Use a higher threshold for quick scan to reduce false positives
                quick_threshold = match_threshold + 0.05
                
                # Basic template matching
                result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= quick_threshold)
                
                if len(locations[0]) > 0:
                    # Potential match found, add page to list and move to next page
                    potential_pages.add(page_num)
                    page_has_matches = True
                    break
            
            if not page_has_matches:
                # Try with binary image if no matches found in grayscale
                for symbol_name, symbol_data in subset_symbols.items():
                    template = symbol_data["gray"]
                    _, template_binary = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    
                    if template_binary.shape[0] > thresh.shape[0] or template_binary.shape[1] > thresh.shape[1]:
                        continue
                    
                    result = cv2.matchTemplate(thresh, template_binary, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= match_threshold)
                    
                    if len(locations[0]) > 0:
                        potential_pages.add(page_num)
                        break

        total_pages = len(doc)
        doc.close()
        
        # If no pages found, include all pages (fallback)
        if not potential_pages:
            potential_pages = set(range(total_pages))
        
        quick_time = time.time() - quick_start
        print(f"Quick scan found {len(potential_pages)} potential pages in {quick_time:.2f} seconds")
        
        return potential_pages

    def _generate_summary_charts(self, output_chart_base_path):
        """Generate multiple charts showing symbol detection statistics."""
        # 1. Top detected symbols chart
        self._generate_top_symbols_chart(f"{output_chart_base_path}_top_symbols.png")
        
        # 2. Detection methods comparison
        self._generate_methods_chart(f"{output_chart_base_path}_methods.png")
        
        # 3. Rotation distribution chart
        self._generate_rotation_chart(f"{output_chart_base_path}_rotations.png")
    
    def _generate_top_symbols_chart(self, output_path):
        """Generate a bar chart showing the most frequently detected symbols."""
        # Get symbols that were found in the diagram
        present_symbols = [(name, sum(pos.get("confidence", 0) for pos in data["positions"])/len(data["positions"]) 
                      if data["positions"] else 0) 
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
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        plt.title(chart_title)
        plt.xlabel("Symbol Name")
        #plt.ylabel("Number of Occurrences")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save chart
        plt.savefig(output_path)
        plt.close()
        print(f"Top symbols chart saved to {output_path}")
    
    def _generate_methods_chart(self, output_path):
        """Generate a pie chart showing distribution of detection methods."""
        # Count detections by method
        template_count = sum(1 for data in self.results.values() 
                        for pos in data["positions"] if pos.get("method", "template") == "template")
        feature_count = sum(1 for data in self.results.values() 
                        for pos in data["positions"] if pos.get("method", "template") == "feature")
        
        # Skip chart creation if no data or only one method
        if template_count == 0 and feature_count == 0:
            print(f"Skipping methods chart - no detection data available")
            return
        
        # Create pie chart
        plt.figure(figsize=(8, 8))
        
        # Prepare data based on what we have
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
        
        # Only create pie chart if we have valid data
        if len(sizes) > 0:
            plt.pie(sizes, explode=tuple(explode), labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=90)
            plt.axis('equal')  # Equal aspect ratio
            plt.title('Symbol Detection Methods Distribution')
            
            # Save chart
            plt.savefig(output_path)
            plt.close()
            print(f"Methods distribution chart saved to {output_path}")
        else:
            print(f"Skipping methods chart - no valid data to display")

    def generate_simplified_report(self, output_json_path):
        """Generate a simplified JSON report with only symbol names, presence, and occurrences."""
        # Create simplified results dict
        simplified_results = {}
        
        # Only include symbols that are present
        for symbol_name, data in self.results.items():
            if data["present"]:
                simplified_results[symbol_name] = {
                    "present": True,
                    "occurrences": data["occurrences"]
                }
        
        # Write to file
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, indent=4)
            
        print(f"Simplified results saved to {output_json_path}")
        
        return simplified_results

    def _exclude_text_regions(self, page_img):
        """Create a mask of likely text regions to exclude from detection."""
        gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
        
        # Use OCR or text-specific detection methods
        # Create horizontal kernel to detect text lines
        kernel = np.ones((1, 15), np.uint8)
        text_mask = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Apply thresholding
        _, text_mask = cv2.threshold(text_mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Dilate to create exclusion zones around text
        text_mask = cv2.dilate(text_mask, np.ones((5, 5), np.uint8), iterations=2)
        
        return text_mask

    def _post_process_results(self):
        """Post-process detection results to remove remaining false positives."""
        # For each symbol type
        for symbol_name, data in self.results.items():
            if not data["present"]:
                continue
            
            # Get all positions
            positions = data["positions"]
            filtered_positions = []
            
            # Group by page for spatial analysis
            page_groups = {}
            for pos in positions:
                page_num = pos["page"]
                if page_num not in page_groups:
                    page_groups[page_num] = []
                page_groups[page_num].append(pos)
            
            # For each page
            for page_num, page_positions in page_groups.items():
                if len(page_positions) < 2:
                    # If only one detection, keep it
                    filtered_positions.extend(page_positions)
                    continue
                
                # 1. Filter based on confidence outliers
                confidences = [pos["confidence"] for pos in page_positions]
                mean_conf = np.mean(confidences)
                std_conf = np.std(confidences)
                
                # Keep positions with confidence not much lower than others
                conf_threshold = max(0.7, mean_conf - 2 * std_conf)
                
                # 2. Filter based on size consistency
                widths = [pos["width"] for pos in page_positions]
                heights = [pos["height"] for pos in page_positions]
                
                median_width = np.median(widths)
                median_height = np.median(heights)
                
                # Process each position
                for pos in page_positions:
                    # Check confidence
                    if pos["confidence"] < conf_threshold:
                        continue
                    
                    # Check size consistency (not too different from median)
                    width_ratio = pos["width"] / median_width
                    height_ratio = pos["height"] / median_height
                    
                    if width_ratio < 0.5 or width_ratio > 2.0 or height_ratio < 0.5 or height_ratio > 2.0:
                        continue
                    
                    # Passed all filters, keep it
                    filtered_positions.append(pos)
            
            # Update results
            data["positions"] = filtered_positions
            data["occurrences"] = len(filtered_positions)
            data["present"] = len(filtered_positions) > 0

    def _generate_rotation_chart(self, output_path):
        """Generate a bar chart showing distribution of symbol rotations."""
        # Count symbols by rotation angle
        rotation_counts = {}
        for data in self.results.values():
            for pos in data["positions"]:
                rotation = pos.get("rotation", 0)
                if rotation in rotation_counts:
                    rotation_counts[rotation] += 1
                else:
                    rotation_counts[rotation] = 1
        
        # Sort by rotation angle
        rotation_items = sorted(rotation_counts.items())
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        angles = [str(angle) for angle, _ in rotation_items]
        counts = [count for _, count in rotation_items]
        
        bars = plt.bar(angles, counts)
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        plt.title('Symbol Rotations Distribution')
        plt.xlabel('Rotation Angle (degrees)')
        plt.ylabel('Number of Detections')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path)
        plt.close()
        print(f"Rotation distribution chart saved to {output_path}")


if __name__ == "__main__":

    symbols_dir = "output_images/symbols"
    pnid_pdf_path = "C:/Turinton/new_clone/pid_symbol_analyzer/data/input/pnid.pdf"  # Your flow diagram PDF
    output_dir = "detection_output"
    output_json_path = os.path.join(output_dir, "symbol_detection_results.json")

    detector = EnhancedPIDSymbolDetector(symbols_dir, output_dir)

    results = detector.detect_symbols_in_pdf(
        pnid_pdf_path, 
        match_threshold=0.80,  
        use_parallel=True,
        dpi=150,
        use_feature_detection=False 
    )
    
    # Generate report
    detector.generate_report(output_json_path)
    simplified_json_path = os.path.join(output_dir, "simplified_results.json")
    detector.generate_simplified_report(simplified_json_path)
