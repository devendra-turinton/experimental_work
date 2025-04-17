import fitz  # PyMuPDF
import cv2
import pytesseract
import numpy as np
from PIL import Image
import json
import os
import re
import shutil

# Load PDF and convert to images
pdf_path = "C:/Turinton/new_clone/pid_symbol_analyzer/data/input/pid-legend.pdf"
doc = fitz.open(pdf_path)

# Create output directories
output_img_dir = "output_images"
labels_dir = os.path.join(output_img_dir, "labels")
symbols_dir = os.path.join(output_img_dir, "symbols")
output_json_path = "extracted_labels.json"


os.makedirs(labels_dir, exist_ok=True)
os.makedirs(symbols_dir, exist_ok=True)
os.makedirs(output_img_dir, exist_ok=True)

# Dictionary to store extracted labels
extracted_labels_dict = {}

# Function to create a valid filename from the label text
def create_valid_filename(text):
    # Replace invalid filename characters with underscores
    valid_filename = re.sub(r'[\\/*?:"<>|]', "_", text)
    # Replace spaces with underscores
    valid_filename = valid_filename.replace(" ", "_")
    # Ensure filename is not empty
    if not valid_filename:
        return "unlabeled"
    # Limit filename length
    return valid_filename[:100]

# Iterate over each page in the PDF
for page_num in range(len(doc)):
    # Extract image from the page
    page = doc[page_num]
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Convert PIL image to OpenCV format
    img_cv = np.array(img) 
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)  # Convert PIL RGB to OpenCV BGR
    img_orig = img_cv.copy()  # Keep an original copy for final visualization
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for symbol detection
    thresh_symbols = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )
    
    # ---- STEP 1: Symbol Detection Using Contours ----
    # Find all contours
    contours, _ = cv2.findContours(thresh_symbols, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    symbol_boxes = []
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h

        # Filter for symbols
        if 0.25 < aspect_ratio < 3.8 and area > 400:
            symbol_boxes.append({
                'id': idx,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })

    # Create a subdirectory for each page
    page_output_dir = os.path.join(output_img_dir, f"page_{page_num+1}")
    os.makedirs(page_output_dir, exist_ok=True)

    # Create a page entry in the dictionary
    page_symbols = []
    extracted_labels_dict[f"Page {page_num+1}"] = page_symbols

    # Debug visualization image 
    debug_image = img_orig.copy()
    
    # ---- STEP 2: Text Detection ----
    # Prepare text binary image
    _, text_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Process for each symbol
    for symbol in symbol_boxes:
        symbol_id = symbol['id']
        sym_x, sym_y, sym_w, sym_h = symbol['x'], symbol['y'], symbol['w'], symbol['h']
        
        # Draw bounding box around symbol
        cv2.rectangle(img_cv, (sym_x, sym_y), (sym_x + sym_w, sym_y + sym_h), (255, 0, 0), 2)
        cv2.rectangle(debug_image, (sym_x, sym_y), (sym_x + sym_w, sym_y + sym_h), (255, 0, 0), 2)
        
        # Define text search region based on symbol position
        # Look to the right of the symbol with finely tuned parameters
        search_x = sym_x + sym_w
        search_y = max(0, sym_y - int(sym_h * 0.15))  # Start slightly above symbol
        
        # Adjust width based on symbol width but with tighter constraints
        # Smaller symbols need proportionally wider search areas
        if sym_w < 40:
            width_multiplier = 4.0  # More space for small symbols
        elif sym_w < 80:
            width_multiplier = 3.0  # Medium space for medium symbols
        else:
            width_multiplier = 2.5  # Less space for large symbols
            
        search_width = min(int(sym_w * width_multiplier), min(200, img_orig.shape[1] - search_x))
        
        # Adjust height based on vertical position in the document (tighter in dense areas)
        # This helps avoid overlapping with adjacent symbols' labels
        row_height_factor = 1.4  # Base factor
        if sym_y < img_orig.shape[0] * 0.3:  # Top third of document
            row_height_factor = 1.5
        elif sym_y > img_orig.shape[0] * 0.7:  # Bottom third of document
            row_height_factor = 1.3
            
        search_height = min(int(sym_h * row_height_factor), img_orig.shape[0] - search_y)
        
        # Draw search area on debug image (optional)
        cv2.rectangle(debug_image, 
                     (search_x, search_y), 
                     (search_x + search_width, search_y + search_height), 
                     (255, 255, 0), 1)  # Yellow rectangle for search area
        
        if search_width <= 0 or search_y + search_height > img_orig.shape[0]:
            continue  # Skip if search region is invalid
            
        # Create a mask for the text search region
        mask = np.zeros_like(text_binary)
        mask[search_y:search_y+search_height, search_x:search_x+search_width] = 1
        search_region_binary = cv2.bitwise_and(text_binary, text_binary, mask=mask)
        
        # Find text contours in the search region
        text_contours, _ = cv2.findContours(search_region_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and merge text contours
        text_boxes = []
        for text_cnt in text_contours:
            tx, ty, tw, th = cv2.boundingRect(text_cnt)
            text_area = tw * th
            text_aspect = tw / float(th) if th > 0 else 0
            
            # More aggressive filtering to avoid noise
            # Adjust based on what you see in the debug image
            if text_area > 15 and 0.2 < text_aspect < 20 and th > 2:
                # Only include contours within reasonable distance from symbol
                if tx < search_x + search_width * 0.9:  # Stay within 90% of search width
                    text_boxes.append((tx, ty, tw, th))
        
        # If text contours found, create optimized bounding box
        if text_boxes:
            # Group text boxes by vertical position to handle multi-line text
            # Sort by y-coordinate
            text_boxes.sort(key=lambda box: box[1])
            
            # Group boxes that are on the same line (close vertical position)
            line_groups = []
            current_group = [text_boxes[0]] if text_boxes else []
            
            for i in range(1, len(text_boxes)):
                current_box = text_boxes[i]
                prev_box = text_boxes[i-1]
                
                # If this box is on the same vertical level as previous (with tolerance)
                vertical_distance = abs((current_box[1] + current_box[3]//2) - 
                                       (prev_box[1] + prev_box[3]//2))
                
                if vertical_distance < max(prev_box[3] * 0.7, 10):  # Same line
                    current_group.append(current_box)
                else:
                    line_groups.append(current_group)
                    current_group = [current_box]
            
            if current_group:
                line_groups.append(current_group)
            
            # Calculate bounding box for each line group
            line_boxes = []
            for group in line_groups:
                if group:
                    group_min_x = min([b[0] for b in group])
                    group_min_y = min([b[1] for b in group])
                    group_max_x = max([b[0] + b[2] for b in group])
                    group_max_y = max([b[1] + b[3] for b in group])
                    line_boxes.append((group_min_x, group_min_y, 
                                      group_max_x - group_min_x, 
                                      group_max_y - group_min_y))
            
            # Now find the overall bounding box that contains all lines
            if line_boxes:
                min_x = min([box[0] for box in line_boxes])
                min_y = min([box[1] for box in line_boxes])
                max_x = max([box[0] + box[2] for box in line_boxes])
                max_y = max([box[1] + box[3] for box in line_boxes])
                
                # Create a bounding box with appropriate padding
                padding_x = 3  # Horizontal padding
                padding_y = 2  # Vertical padding
                
                label_x = max(search_x, min_x - padding_x)
                label_y = max(search_y, min_y - padding_y)
                label_w = min(search_width, max_x - label_x + padding_x*2)
                label_h = min(search_height, max_y - label_y + padding_y*2)
                
                # Ensure minimum width and height for very small text
                label_w = max(label_w, 15)
                label_h = max(label_h, 10)
                
                # Draw the label bounding box on the debug image
                cv2.rectangle(debug_image, 
                             (label_x, label_y), 
                             (label_x + label_w, label_y + label_h), 
                             (0, 255, 0), 2)
                
                # Extract the label region for OCR
                label_img = img_orig[label_y:label_y+label_h, label_x:label_x+label_w]
                
                # Save the symbol image with temporary name
                symbol_crop = img_orig[sym_y:sym_y+sym_h, sym_x:sym_x+sym_w]
                temp_symbol_filename = os.path.join(page_output_dir, f"symbol_{symbol_id+1}.png")
                cv2.imwrite(temp_symbol_filename, symbol_crop)
                
                # Perform OCR on the label
                # Resize for better OCR
                label_img_resized = cv2.resize(label_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                label_gray = cv2.cvtColor(label_img_resized, cv2.COLOR_BGR2GRAY)
                
                # Try multiple OCR configurations
                ocr_results = []
                
                # Direct OCR
                text1 = pytesseract.image_to_string(label_gray, config='--psm 6 --oem 3').strip()
                if text1:
                    ocr_results.append(text1)
                    
                # Threshold and OCR
                _, thresh = cv2.threshold(label_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                text2 = pytesseract.image_to_string(thresh, config='--psm 6 --oem 3').strip()
                if text2:
                    ocr_results.append(text2)
                
                # Choose best OCR result (longest non-empty result)
                ocr_results = [text for text in ocr_results if text.strip()]
                final_text = max(ocr_results, key=len) if ocr_results else "No text detected"
                
                # Clean up the text
                final_text = re.sub(r'[^\w\s.,\-()]', '', final_text)
                final_text = re.sub(r'\s+', ' ', final_text).strip()
                final_text = re.sub(r'www\.edrawsoft\.com', '', final_text)
                
                # Save the label image
                label_filename = os.path.join(page_output_dir, f"label_{symbol_id+1}.png")
                cv2.imwrite(label_filename, label_img)
                
                # Rename symbol image to use label text as filename
                if final_text and final_text != "No text detected":
                    valid_filename = create_valid_filename(final_text)
                    
                    # Handle potential duplicates
                    symbol_output_filename = os.path.join(symbols_dir, f"{valid_filename}.png")
                    label_output_filename = os.path.join(labels_dir, f"{valid_filename}.png")
                    
                    # If a file with this name already exists, add a suffix number
                    counter = 1
                    base_filename = valid_filename
                    while os.path.exists(symbol_output_filename):
                        counter += 1
                        valid_filename = f"{base_filename}_{counter}"
                        symbol_output_filename = os.path.join(symbols_dir, f"{valid_filename}.png")
                        label_output_filename = os.path.join(labels_dir, f"{valid_filename}.png")
                    
                    # Copy files to their respective directories
                    shutil.copy2(temp_symbol_filename, symbol_output_filename)
                    shutil.copy2(label_filename, label_output_filename)
                    
                    # Update the image paths in our results
                    symbol_path = f"symbols/{valid_filename}.png"
                    label_path = f"labels/{valid_filename}.png"
                else:
                    # For unlabeled symbols, use the symbol ID
                    unlabeled_name = f"unlabeled_{symbol_id+1}"
                    symbol_output_filename = os.path.join(symbols_dir, f"{unlabeled_name}.png")
                    
                    # Copy the file
                    shutil.copy2(temp_symbol_filename, symbol_output_filename)
                    
                    # Update the image path
                    symbol_path = f"symbols/{unlabeled_name}.png"
                    label_path = None

                
                # Add text label to debug image
                cv2.putText(debug_image, 
                           final_text[:15] + "..." if len(final_text) > 15 else final_text,
                           (label_x, label_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Add to results
                page_symbols.append({
                    "symbol_id": symbol_id + 1,
                    "symbol_image": symbol_path,
                    "label_image": label_path,
                    "symbol_position": {"x": sym_x, "y": sym_y, "width": sym_w, "height": sym_h},
                    "label_text": final_text,
                    "label_position": {"x": label_x, "y": label_y, "width": label_w, "height": label_h}
                })
            else:
                # No label found
                symbol_crop = img_orig[sym_y:sym_y+sym_h, sym_x:sym_x+sym_w]
                symbol_filename = os.path.join(page_output_dir, f"symbol_{symbol_id+1}.png")
                cv2.imwrite(symbol_filename, symbol_crop)
                
                page_symbols.append({
                    "symbol_id": symbol_id + 1,
                    "symbol_image": f"symbols/unlabeled_{symbol_id+1}.png",
                    "label_image": None,
                    "symbol_position": {"x": sym_x, "y": sym_y, "width": sym_w, "height": sym_h},
                    "label_text": "No label detected",
                    "label_position": None
                })
        else:
            # No text contours found
            symbol_crop = img_orig[sym_y:sym_y+sym_h, sym_x:sym_x+sym_w]
            symbol_filename = os.path.join(page_output_dir, f"symbol_{symbol_id+1}.png")
            cv2.imwrite(symbol_filename, symbol_crop)
            
            # Save to symbols directory with unlabeled prefix
            unlabeled_name = f"unlabeled_{symbol_id+1}"
            symbol_output_filename = os.path.join(symbols_dir, f"{unlabeled_name}.png")
            shutil.copy2(symbol_filename, symbol_output_filename)
            
            page_symbols.append({
                "symbol_id": symbol_id + 1,
                "symbol_image": f"symbols/{unlabeled_name}.png",
                "label_image": None,
                "symbol_position": {"x": sym_x, "y": sym_y, "width": sym_w, "height": sym_h},
                "label_text": "No label detected",
                "label_position": None
            })
            
    print(f"✅ Processed symbols and labels for page {page_num+1}")
    
    # Save the visualization images
    output_page_path = os.path.join(output_img_dir, f"page_{page_num+1}_associations.png")
    cv2.imwrite(output_page_path, img_cv)
    
    debug_page_path = os.path.join(output_img_dir, f"page_{page_num+1}_debug.png")
    cv2.imwrite(debug_page_path, debug_image)

# Save extracted labels as JSON
with open(output_json_path, "w", encoding="utf-8") as json_file:
    json.dump(extracted_labels_dict, json_file, indent=4, ensure_ascii=False)

print(f"✅ All symbols and labels processed")
print(f"✅ Results saved to {output_json_path}")
print(f"✅ Debug images saved to {output_img_dir}")