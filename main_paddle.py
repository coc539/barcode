import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from paddleocr import PaddleOCR
from pyzbar import pyzbar
import pandas as pd
from PIL import Image, ImageTk
import datetime
import time
from pathlib import Path
import os
from ttkthemes import ThemedTk
import re
import logging
import sys
import queue
import threading

class UnifiedScannerApp:
    def __init__(self, root):
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
            handlers=[
                logging.FileHandler('unified_scanner_debug.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.root = root
        self.root.title("Unified Scanner")
        self.root.set_theme("arc")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Initialize variables
        self.is_scanning = False
        self.camera_active = False
        self.scanned_items = {}
        self.last_scanned_time = {}
        self.debounce_time = 3
        self.session_start_time = None
        
        # Create base directory for scans
        self.base_dir = "scan_results"
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Session tracking and data storage
        self.current_session_data = []
        self.results_lock = threading.Lock()
        
        # Enhanced frame quality parameters
        self.min_contrast_threshold = 40
        self.min_sharpness_threshold = 60
        self.blur_threshold = 90
        self.brightness_threshold = (50, 200)  # Min and max acceptable brightness
        
        # Initialize PaddleOCR with enhanced configuration
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True, 
                lang='en', 
                show_log=False,
                det_limit_side_len=960,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5
            )
            logging.info("PaddleOCR initialized successfully")
        except Exception as e:
            logging.error(f"OCR Initialization Error: {str(e)}")
            messagebox.showerror("OCR Initialization Error", f"Failed to initialize PaddleOCR: {str(e)}")
            raise
        
        self.setup_ui()
        
        # Setup structured text parsing method
        self.parse_structured_text = self.create_parse_structured_text_method()
        
        # Bind application close event to save session
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.current_session_data = []
        self.current_row_data = {}
    
    def create_parse_structured_text_method(self):
        def parse_structured_text(text):
            """
            Enhanced text parsing with multiple strategies and more comprehensive patterns
            """
            # Remove any extra whitespace and normalize
            text = re.sub(r'\s+', ' ', text.strip())

            # Comprehensive regular expression patterns for different fields with multiple variations
            patterns = {
                'GRN_NUMBER': [
                    r'GRN\s*[:#]?\s*(\w+)',
                    r'GOODS\s*RECEIPT\s*NOTE\s*[:#]?\s*(\w+)',
                    r'G\.?R\.?N\.?\s*[:#]?\s*(\w+)'
                ],
                'ITEMID': [
                    r'ITEM\(1P\)\s*[:#]?\s*([\w-]+)',  #TDK   
                    r'ITEM\s*(?:ID|CODE)\s*[:#]?\s*(\w+)',
                    r'MATERIAL\s*CODE\s*[:#]?\s*(\w+)' ,
                    r'33TPUID\s*\s*(\w+)' #nexperia
                    
                                 
                ],
                'QUANTITY': [
                    r'QTY\(Q\)\s*[:#]?\s*(\d+)', #TDK
                    r'QUANTITY\s*[:#]?\s*(\d+(?:\.\d+)?)',
                    r'NO\.\s*OF\s*PIECES\s*[:#]?\s*(\d+(?:\.\d+)?)',
                    r'\(Q\)QTY\s*[:#]?\s*(\d+)' #nexperia
                ], 
                'FLM': [
                    r'FLM\s*[:#]?\s*(\w+)',
                    r'FIRST\s*LEVEL\s*MATERIAL\s*[:#]?\s*(\w+)',
                    r'PRIMARY\s*MATERIAL\s*[:#]?\s*(\w+)'
                ],
                'MANF_NAME': [
                    r'VDR\(V\)\s*[:]?\s*(\w+)', #TDK
                    r'MANF(?:ACTURER)?\s*[:#]?\s*([^0-9\n]+)',
                    r'MANUFACTURER\s*[:#]?\s*([^0-9\n]+)',
                    r'MADE\s*BY\s*[:#]?\s*([^0-9\n]+)',
                    r'NEXPERIA\s*' #nexperia
                
                ],
                'ORDER_CODE': [
                    r'ORDER\s*[:#]?\s*(\w+)',
                    r'PO\s*[:#]?\s*(\w+)',
                    r'PURCHASE\s*ORDER\s*[:#]?\s*(\w+)'         
                ],
                'BATCH_NUMBER': [
                    r'BATCH\s*[:#]?\s*(\w+)',
                    r'PRODUCTION\s*BATCH\s*[:#]?\s*(\w+)'
                ],
                'DATECODE': [
                    r'DATE\s*CODE\(T\)\s*[:#]?\s*(\d+)', #TDK
                    r'DATE\s*(?:CODE)?\s*[:#]?\s*(\d{2,4}[-/]\d{2,4}[-/]\d{2,4})',
                    r'MFG\.\s*DATE\s*[:#]?\s*(\d{2,4}[-/]\d{2,4}[-/]\d{2,4})',
                    r'PRODUCTION\s*DATE\s*[:#]?\s*(\d{2,4}[-/]\d{2,4}[-/]\d{2,4})',
                    r'\(9D\)DATE\s*[:#]?\s*(\d+)' #nexperia

                ],
                'LOT': [
                    r'LOT\s*NO\(1T\)\s*[:#]?\s*([\w+]+\w+)',
                    r'LOT\s*NO\(1T\)\s*:\s*([\w-]+\+\w+)', #TDK
                    r'LOT\s*[:#]?\s*(\w+)',
                    r'LOT\s*NUMBER\s*[:#]?\s*(\w+)',
                    r'BATCH\s*[:#]?\s*(\w+)' ,
                    r'\(1T\)LOT\s*[:#]?\s*([\w-]+)'  #nexperia

                ],
                'MARKING': [  
                    r'MARKING\s*[:#]?\s*([^0-9\n]+)',
                    r'PART\s*MARKING\s*[:#]?\s*([^0-9\n]+)'
                ],
                'ROHSCOMPLIANCE': [
                    r'ROHS\s*[:#]?\s*(\w+)',
                    r'RoHS\s*COMPLIANT\s*[:#]?\s*(\w+)',
                    r'RoHS\s*COMPLIANT'
                ]
            }

            parsed_data = {}
            for key, pattern_list in patterns.items():
                for pattern in pattern_list:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        # Clean up the matched value
                        value = match.group(1).strip()

                        # Additional validation and cleaning for specific fields
                        if key == 'QUANTITY':
                            # Ensure numerical value
                            try:
                                value = str(float(value))
                            except ValueError:
                                continue
                            
                        parsed_data[key] = value
                        break  # Stop if a match is found for this key
                    
            return parsed_data

        return parse_structured_text
        
    def assess_frame_quality(self, frame):
        """
        Enhanced frame quality assessment
        Returns a quality score between 0 and 100
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Assess contrast using standard deviation
        contrast = gray.std()
        
        # Assess sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Assess blur using standard deviation of Laplacian
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Assess brightness
        mean_brightness = np.mean(gray)
        
        # Calculate individual component scores
        contrast_score = min(max(contrast / self.min_contrast_threshold * 100, 0), 100)
        sharpness_score = min(max(laplacian_var / self.min_sharpness_threshold * 100, 0), 100)
        blur_score = max(100 - (blur_value / self.blur_threshold * 100), 0)
        
        # Brightness score (penalize frames that are too dark or too bright)
        brightness_score = 100 - abs(mean_brightness - np.mean(self.brightness_threshold)) / np.ptp(self.brightness_threshold) * 100
        
        # Combine scores with custom weighting
        quality_score = (
            0.3 * contrast_score + 
            0.3 * sharpness_score + 
            0.2 * blur_score +
            0.2 * brightness_score
        )
        
        return quality_score
    
    
    def setup_ui(self):
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Control Panel
        control_panel = ttk.Frame(main_container)
        control_panel.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title_frame = ttk.Frame(control_panel)
        title_frame.pack(side=tk.LEFT)
        
        title_label = ttk.Label(title_frame, text="Unified Scanner", font=("Helvetica", 16, "bold"))
        title_label.pack(anchor=tk.W)
        
        self.save_status_label = ttk.Label(title_frame, text="Session: Not Started", font=("Helvetica", 8))
        self.save_status_label.pack(anchor=tk.W)
        
        # Control Buttons
        self.scan_button = ttk.Button(control_panel, text="Start Scanning", command=self.toggle_scanning)
        self.scan_button.pack(side=tk.RIGHT, padx=5)
        
        # Main Content Area
        split_container = ttk.Frame(main_container)
        split_container.pack(fill=tk.BOTH, expand=True)
        
        # Camera Feed
        camera_frame = ttk.LabelFrame(split_container, text="Camera Feed", padding="5")
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.video_frame = ttk.Frame(camera_frame)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        self.status_label = ttk.Label(camera_frame, text="Camera: Stopped", font=("Helvetica", 10))
        self.status_label.pack(pady=5)
        
        # Results Area
        items_frame = ttk.LabelFrame(split_container, text="Scanned Items", padding="5")
        items_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Treeview for results
        self.tree = ttk.Treeview(items_frame, columns=("Row", "Type", "Data", "Timestamp"), show="headings")
        self.tree.heading("Row", text="Row")
        self.tree.heading("Type", text="Type")
        self.tree.heading("Data", text="Data")
        self.tree.heading("Timestamp", text="Timestamp")
        
        self.tree.column("Row", width=50, anchor='center')
        self.tree.column("Type", width=100)
        self.tree.column("Data", width=300)
        self.tree.column("Timestamp", width=150)
        
        scrollbar = ttk.Scrollbar(items_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_session_directory(self):
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        date_dir = os.path.join(self.base_dir, current_date)
        os.makedirs(date_dir, exist_ok=True)
        
        self.session_start_time = datetime.datetime.now()
        session_name = self.session_start_time.strftime("%H-%M-%S")
        session_dir = os.path.join(date_dir, session_name)
        os.makedirs(session_dir)
        
        return session_dir

    def toggle_scanning(self):
        if not self.is_scanning:
            self.start_scanning()
        else:
            self.stop_scanning()

    def start_scanning(self):
        # If there's a previous row with data, save it before starting new scan
        if hasattr(self, 'current_row_data') and (self.current_row_data.get('Scanned_Barcodes') or self.current_row_data.get('Scanned_Text')):
            # Add the current row to the session data if it has content
            self.current_session_data.append(self.current_row_data)

        # Reset current row data when starting a new scan
        self.current_row_data = {
            'Row': len(self.current_session_data) + 1,  # Increment row number
            'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Scanned_Barcodes': [],
            'Scanned_Text': []
        }
        
        

        # Clear previous items in the treeview
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.session_start_time = datetime.datetime.now()
        self.is_scanning = True
        self.camera_active = True
        self.scan_button.configure(text="Stop Scanning")
        self.status_label.configure(text="Camera: Active")

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        self.update_frame()


    def stop_scanning(self):
        if self.is_scanning:
            # Save the current row before stopping
            if self.current_row_data.get('Scanned_Barcodes') or self.current_row_data.get('Scanned_Text'):
                self.current_session_data.append(self.current_row_data)

            # Save the entire session
            self.save_session()

        self.is_scanning = False
        self.camera_active = False
        self.scan_button.configure(text="Start Scanning")
        self.status_label.configure(text="Camera: Stopped")
        if hasattr(self, 'cap'):
            self.cap.release()

    def update_frame(self):     
        if self.camera_active:
            ret, frame = self.cap.read()
            if ret:
                processed_frame = self.process_frame(frame)
                
                # Convert to PhotoImage for display
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image=image)
                self.video_label.configure(image=photo)
                self.video_label.image = photo
            
            self.root.after(10, self.update_frame)


    def detect_and_extract_text(self, frame):
        """
        Comprehensive text detection with multiple fallback strategies
        """
        try:
            # Try preprocessed frame first
            preprocessed = self.preprocess_frame(frame)
            results = self.ocr.ocr(preprocessed, cls=True)
            
            # Ensure results have the expected structure
            if results and isinstance(results, list):
                # Flatten results if necessary
                if len(results) > 0 and results[0] is not None:
                    # Process results
                    all_texts = []
                    all_bboxes = []
                    all_confidences = []
                    
                    # Safely iterate through results
                    for detection in results[0]:
                        try:
                            # Convert bbox to a string representation that's easier to parse
                            bbox = detection[0]
                            bbox_str = ' '.join([f'{int(coord)}' for sublist in bbox for coord in sublist])
                            text = detection[1][0]
                            confidence = detection[1][1]
                            
                            all_texts.append(text)
                            all_bboxes.append(bbox_str)
                            all_confidences.append(str(confidence))
                        except (IndexError, TypeError) as e:
                            logging.error(f"Error processing OCR result line: {e}")
                    
                    # Consolidate into single entries
                    if all_texts:
                        consolidated_result = {
                            'texts': ' '.join(all_texts),
                            'bounding_boxes': ' | '.join(all_bboxes),
                            'confidences': ' '.join(map(str, all_confidences))
                        }
                        
                        logging.info(f"Detected {len(results[0])} text regions")
                        return consolidated_result
            
            logging.warning("No text detected in frame")
            return None
        
        except Exception as e:
            logging.error(f"Text detection error: {e}")
            return None


    def preprocess_frame(self, frame):
        """
        Advanced frame preprocessing techniques
        """
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Apply multiple preprocessing techniques0
            # 1. Adaptive Thresholding
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_MEAN_C, 
                cv2.THRESH_BINARY, 11, 6
            )
            
            # 2. Noise Reduction
            denoised = cv2.fastNlMeansDenoising(adaptive_thresh, None, 10, 7, 21)
            
            # 3. Contrast Enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            logging.debug("Frame preprocessing completed successfully")
            return enhanced
        
        except Exception as e:
            logging.error(f"Frame preprocessing error: {e}")
            return frame
        
    def process_frame(self, frame):
        # Assess frame quality first
        quality_score = self.assess_frame_quality(frame)
        
        # Only process if frame quality is good (above 70%)
        if quality_score < 70:
            return frame  # Return original frame without processing
        
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Flag to track if any meaningful data was found
        data_found = False
        
        # Process barcodes
        barcodes = pyzbar.decode(frame)
        for barcode in barcodes:
            self.process_barcode(barcode)
            # Draw rectangle around barcode
            (x, y, w, h) = barcode.rect
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, "Barcode", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),  2)
            data_found = True
        
        # Enhanced text detection
        detected_texts = self.detect_and_extract_text(frame)
        
        if detected_texts:
            # Process detected texts
            if 'texts' in detected_texts:
                self.process_text(detected_texts['texts'])
                
                # Draw bounding boxes if available
                try:
                    bboxes_str = detected_texts.get('bounding_boxes', '')
                    bboxes_list = []
                    
                    # Split and process bounding boxes
                    for bbox_str in bboxes_str.split(' | '):
                        bbox_coords = list(map(int, bbox_str.split()))
                        bbox = np.array(bbox_coords).reshape(-1, 2)
                        bboxes_list.append(bbox)
                    
                    # Draw bounding boxes on the frame
                    for bbox in bboxes_list:
                        cv2.polylines(display_frame, [bbox], True, (255, 0, 0), 2)
                        
                    data_found = True
                except Exception as e:
                    logging.error(f"Error drawing text bounding boxes: {e}")
        
        # Add quality score and data detection to frame
        cv2.putText(display_frame, 
                    f"Quality: {quality_score:.2f}%", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0) if data_found else (0, 0, 255), 
                    2)
        
        return display_frame


    def process_barcode(self, barcode):
        barcode_data = barcode.data.decode("utf-8")
        
        # Only add if not already in current row's barcodes
        if barcode_data not in self.current_row_data['Scanned_Barcodes']:
            self.current_row_data['Scanned_Barcodes'].append(barcode_data)
            
            # Update display (but don't create new treeview entry)
            self.tree.delete(*self.tree.get_children())  # Clear existing entries
            self.tree.insert("", 0, values=(
                self.current_row_data['Row'], 
                f"Barcode ({barcode.type})", 
                ", ".join(self.current_row_data['Scanned_Barcodes']),
                self.current_row_data['Timestamp']
            ))

    def process_text(self, text):
        """
        Enhanced text processing with additional logging and structured extraction
        """
        # Detect and parse structured text 
        try:
            # Parse structured text
            parsed_data = self.parse_structured_text(text)
            
            # Advanced text processing similar to previous method
            if parsed_data:
                # Store text as full object, not just display
                if parsed_data not in [item.get('text_data', {}) for item in self.current_row_data.get('Scanned_Text', [])]:
                    # Add text to current row's scanned items if not already present
                    if 'Scanned_Text' not in self.current_row_data:
                        self.current_row_data['Scanned_Text'] = []
                    
                    text_entry = {
                        'text_data': parsed_data,
                        'raw_text': text
                    }
                    
                    self.current_row_data['Scanned_Text'].append(text_entry)
                    
                    # Log the detected text
                    logging.info(f"Processed Text: {text}")
                    logging.info(f"Parsed Data: {parsed_data}")
                    
                    # Update display
                    self.tree.delete(*self.tree.get_children())  # Clear existing entries
                    self.tree.insert("", 0, values=(
                        self.current_row_data.get('Row', 1), 
                        "Structured Text", 
                        f"GRN: {parsed_data.get('GRN_NUMBER', 'N/A')}", 
                        f"Item: {parsed_data.get('ITEMID', 'N/A')}", 
                        self.current_row_data.get('Timestamp', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    ))
        except Exception as e:
            logging.error(f"Text processing error: {e}")

                
    def save_session(self):
        try:
            # Prepare a list to store rows for saving
            rows_to_save = []

            # Iterate through sessions data
            for row_data in self.current_session_data:
                # Base row with timestamp and barcodes
                base_row = {
                    'Timestamp': row_data.get('Timestamp', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    'Barcodes': ', '.join(row_data.get('Scanned_Barcodes', []))
                }

                # If no data, skip this row
                if not base_row['Barcodes'] and not row_data.get('Scanned_Text'):
                    continue

                # Handle multiple text entries
                if row_data.get('Scanned_Text'):
                    for text_entry in row_data['Scanned_Text']:
                        # Create a complete row by merging base data and parsed text data
                        row = base_row.copy()

                        # Add structured text details
                        text_data = text_entry.get('text_data', {})
                        for key, value in text_data.items():
                            row[key] = value

                        rows_to_save.append(row)
                else:
                    rows_to_save.append(base_row)

            # If no data to save, return
            if not rows_to_save:
                return

            # Create DataFrame
            df = pd.DataFrame(rows_to_save)

            # Ensure consistent column order with most important fields first
            desired_columns_order = [
                'Timestamp', 'Barcodes', 
                'GRN_NUMBER', 'ITEMID', 'QUANTITY', 
                'MANF_NAME', 'LOT_NUMBER', 'DATECODE', 
                'ORDER_CODE', 'FLM', 'MARKING', 'ROHSCOMPLIANCE'
            ]

            # Add missing columns with empty values
            for col in desired_columns_order:
                if col not in df.columns:
                    df[col] = ''

            # Reorder columns
            df = df[desired_columns_order]

            # Create session directory
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            date_dir = os.path.join(self.base_dir, current_date)
            os.makedirs(date_dir, exist_ok=True)

            # Generate unique filename with timestamp
            timestamp = datetime.datetime.now().strftime("%H-%M-%S")
            excel_filename = f"scan_data_{timestamp}.xlsx"
            excel_path = os.path.join(date_dir, excel_filename)

            # Save DataFrame to Excel
            df.to_excel(excel_path, index=False)

            self.save_status_label.config(text=f"Saved: {excel_filename}")

            messagebox.showinfo("Session Saved", 
                                f"Data saved to {excel_path}\n"
                                f"Total entries: {len(rows_to_save)}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving session data: {str(e)}")
            
    def on_closing(self):
        # Save current session before closing
        self.save_session()
        # Close the application
        self.root.destroy()

def main():
    root = ThemedTk(theme="arc")
    app = UnifiedScannerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

