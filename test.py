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
        self.root.title("OCR Scanner")
        self.root.set_theme("arc")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Initialize queues for thread communication
        self.frame_queue = queue.Queue(maxsize=2)  # Limit queue size to prevent memory buildup
        self.result_queue = queue.Queue()
        
        # Initialize variables
        self.is_scanning = False
        self.camera_active = False
        self.processing_thread = None
        self.camera_thread = None
        self.scanned_items = {}
        self.last_scanned_time = {}
        self.debounce_time = 3
        self.session_start_time = None
        
        # Enhanced frame quality parameters
        self.min_contrast_threshold = 40
        self.min_sharpness_threshold = 60
        self.blur_threshold = 90
        self.brightness_threshold = (50, 200)  # Min and max acceptable brightness
        
        # Create base directory for scans
        self.base_dir = "scan_results"
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Session tracking and data storage
        self.current_session_data = []
        self.results_lock = threading.Lock()
        
        # Initialize OCR in a separate thread
        self.ocr = None
        self.init_ocr_thread = threading.Thread(target=self.initialize_ocr)
        self.init_ocr_thread.daemon = True
        self.init_ocr_thread.start()
        
        self.current_excel_file = None
        self.excel_data = []
        self.create_new_excel_file()
        
        self.available_cameras = {}
        self.selected_camera = tk.StringVar()
        self.get_available_cameras()
        
        
        self.setup_ui()
        
        # Setup structured text parsing method
        self.parse_structured_text = self.create_parse_structured_text_method()
        
        # Bind application close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.current_session_data = []
        self.current_row_data = {}
        
        
        logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler('unified_scanner_debug.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    def get_available_cameras(self):
        """Detect available cameras and identify Razer webcam"""
        self.available_cameras.clear()
        
        # Try cameras from index 0 to 10
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera name/description
                backend = cap.getBackendName()
                try:
                    # Try to get camera name - this might not work on all systems
                    camera_name = cap.getBackendName() + " " + str(i)
                    
                    # Check if it's a Razer camera
                    if "Razer" in camera_name:
                        # Put Razer camera first in the dict
                        self.available_cameras = {camera_name: i, **self.available_cameras}
                    else:
                        self.available_cameras[camera_name] = i
                except:
                    camera_name = f"Camera {i}"
                    self.available_cameras[camera_name] = i
                
                cap.release()
        
        # If no cameras found, show error
        if not self.available_cameras:
            messagebox.showerror("Error", "No cameras detected!")
            return
        
        # Set default camera to Razer if available, otherwise first available camera
        default_camera = next(iter(self.available_cameras.keys()))
        self.selected_camera.set(default_camera)

    def refresh_cameras(self):
        """Refresh the list of available cameras"""
        current_camera = self.selected_camera.get()
        self.get_available_cameras()
        
        # Update dropdown values
        self.camera_dropdown['values'] = list(self.available_cameras.keys())
        
        # Try to keep the same camera selected if it's still available
        if current_camera in self.available_cameras:
            self.selected_camera.set(current_camera)
        else:
            self.selected_camera.set(next(iter(self.available_cameras.keys())))
            
    
    def initialize_ocr(self):
        """Initialize PaddleOCR in a separate thread"""
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
            self.root.after(0, lambda: messagebox.showerror("OCR Initialization Error", f"Failed to initialize PaddleOCR: {str(e)}"))

    def camera_capture_loop(self):
        """Dedicated thread for camera capture"""
        while self.camera_active:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    try:
                        # Only put new frame if queue is not full
                        if not self.frame_queue.full():
                            self.frame_queue.put(frame)
                    except queue.Full:
                        continue
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage

    def process_frames_loop(self):
        """Dedicated thread for frame processing"""
        while self.camera_active:
            try:
                frame = self.frame_queue.get(timeout=1)
                processed_frame = self.process_frame(frame)
                self.result_queue.put(processed_frame)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Frame processing error: {e}")

    def update_frame(self):
        """Update UI with processed frames"""
        if self.camera_active:
            try: 
                # Get processed frame from result queue
                processed_frame = self.result_queue.get_nowait()
                 
                # Convert to PhotoImage for display
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image=image)
                self.video_label.configure(image=photo)
                self.video_label.image = photo
                
            except queue.Empty:
                pass  # Skip frame if none available
            
            self.root.after(10, self.update_frame)
    
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
                    # r'GRN\s*[:#]?\s*(\w+)',
                    # r'GOODS\s*RECEIPT\s*NOTE\s*[:#]?\s*(\w+)',
                    # r'G\.?R\.?N\.?\s*[:#]?\s*(\w+)'
                ],
                'ITEMID': [
                    r'ITEM\(1P\)\s*[:#]?\s*([\w-]+)',  #TDK   
                    r'ITEM\s*(?:ID|CODE)\s*[:#]?\s*(\w+)'
                    # r'MATERIAL\s*CODE\s*[:#]?\s*(\w+)' ,
                    # r'\(33T\)PUID\s*\s*\s*(\w+)' #nexperia
                ],
                'QUANTITY': [
                    r'QTY\s*\(Q\)\s*[:]\s*(\d+)', #TDK
                    # r'\(Q\)QTY\s*\s*\s*\s*\s*(\d+)', #nexperia
                    r'QUANTITY\s*[:#]?\s*(\d+(?:\.\d+)?)'
                    # r'NO\.\s*OF\s*PIECES\s*[:#]?\s*(\d+(?:\.\d+)?)'
                ],
                'FLM': [
                    # r'FLM\s*[:#]?\s*(\w+)',
                    # r'FIRST\s*LEVEL\s*MATERIAL\s*[:#]?\s*(\w+)',
                    # r'PRIMARY\s*MATERIAL\s*[:#]?\s*(\w+)'
                ],
                'MANF_NAME': [
                    r'VDR\(V\)\s*[:]?\s*(\w+)', #TDK
                    # r'VDR\(V\)\s*:\s*(\w+)',
                    # r'MANF(?:ACTURER)?\s*[:#]?\s*([^0-9\n]+)',
                    # r'MANUFACTURER\s*[:#]?\s*([^0-9\n]+)',
                    # r'MADE\s*BY\s*[:#]?\s*([^0-9\n]+)',
                    # r'NEXPERIA\s*' #nexperia
                ],
                'ORDER_CODE': [
                    # r'ORDER\s*[:#]?\s*(\w+)',
                    # r'PO\s*[:#]?\s*(\w+)',
                    # r'PURCHASE\s*ORDER\s*[:#]?\s*(\w+)'         
                ],
                'BATCH_NUMBER': [
                    # r'BATCH\s*[:#]?\s*(\w+)',
                    # r'PRODUCTION\s*BATCH\s*[:#]?\s*(\w+)'
                ],
                'DATECODE': [
                    r'DATE\s*CODE\(T\)\s*[:#]?\s*(\d+)', #TDK
                    # r'\(9D\)DATE\s*\s*\s*(\d+)', #nexperia
                    # r'DATE\s*(?:CODE)?\s*[:#]?\s*(\d{2,4}[-/]\d{2,4}[-/]\d{2,4})',
                    # r'MFG\.\s*DATE\s*[:#]?\s*(\d{2,4}[-/]\d{2,4}[-/]\d{2,4})',
                    # r'PRODUCTION\s*DATE\s*[:#]?\s*(\d{2,4}[-/]\d{2,4}[-/]\d{2,4})'
                ],
                'LOT': [
                    # r'\(1T\)\s*([A-Za-z0-9-]+)',           # Basic (1T) format
                    r'LOT\s*NO\s*\(1T\):\s*([\w-]+(?:\+\w+)?)',
                    r'LOT\s*NO\(1T\)\s*:\s*([\w-]+(?:\+\w+)?)',# TDK format with optional +
                    # r'LOT:\s*NO\s*\(1T\):([\w-]+(?:\+\w+)?)',
                    r'LOT\s*[:#]?\s*([\w-]+(?:\+\w+)?)',   # General LOT format
                    r'LOT\s*NUMBER\s*[:#]?\s*([\w-]+)'     # Full LOT NUMBER format
                    # r'BATCH\s*[:#]?\s*([\w-]+)',           # BATCH format
                    # r'\(1T\)LOT\s*[:#]?\s*([\w-]+)'        # Nexperia format
                ],
                'MARKING': [  
                    # r'MARKING\s*[:#]?\s*([^0-9\n]+)',
                    # r'PART\s*MARKING\s*[:#]?\s*([^0-9\n]+)'
                ],
                'ROHSCOMPLIANCE': [
                    # r'ROHS\s*[:#]?\s*(\w+)',
                    # r'RoHS\s*COMPLIANT\s*[:#]?\s*(\w+)',
                    # r'RoHS\s*COMPLIANT'
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
        
        title_label = ttk.Label(title_frame, text="OCR Scanner", font=("Helvetica", 16, "bold"))
        title_label.pack(anchor=tk.W)
        
        self.save_status_label = ttk.Label(title_frame, text="Session: Not Started", font=("Helvetica", 8))
        self.save_status_label.pack(anchor=tk.W)
        
        # Camera Selection
        camera_frame = ttk.Frame(control_panel)
        camera_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(camera_frame, text="Select Camera:").pack(side=tk.LEFT, padx=5)
        
        self.camera_dropdown = ttk.Combobox(
            camera_frame, 
            textvariable=self.selected_camera,
            values=list(self.available_cameras.keys()),
            state="readonly",
            width=30
        )
        self.camera_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Refresh Camera Button
        refresh_btn = ttk.Button(
            camera_frame, 
            text="↻", 
            width=3,
            command=self.refresh_cameras
        )
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # Control Buttons Frame
        buttons_frame = ttk.Frame(control_panel)
        buttons_frame.pack(side=tk.RIGHT)
        
        # Start/Stop Button
        self.scan_button = ttk.Button(buttons_frame, text="Start Camera", command=self.toggle_scanning)
        self.scan_button.pack(side=tk.LEFT, padx=5)
        
        # Capture Button
        self.capture_button = ttk.Button(buttons_frame, text="Capture", command=self.capture_data, state='disabled')
        self.capture_button.pack(side=tk.LEFT, padx=5)
        
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
        """Modified start_scanning with proper thread initialization checks"""
        self.is_scanning = True
        self.camera_active = True

        # Initialize new row data
        self.current_row_data = {
            'Row': len(self.excel_data) + 1,
            'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Scanned_Barcodes': [],
            'Scanned_Text': {}
        }

        # Clear display tree
        for item in self.tree.get_children():
            self.tree.delete(item)

        try:
            # Get selected camera index
            camera_name = self.selected_camera.get()
            camera_index = self.available_cameras[camera_name]

            # Initialize camera
            self.cap = cv2.VideoCapture(camera_index)

            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera {camera_name}")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Safely check and create threads
            # First, initialize thread attributes if they don't exist
            if not hasattr(self, 'camera_thread'):
                self.camera_thread = None
            if not hasattr(self, 'processing_thread'):
                self.processing_thread = None

            # Now check if threads exist and are alive
            if self.camera_thread is None or not self.camera_thread.is_alive():
                self.camera_thread = threading.Thread(target=self.camera_capture_loop)
                self.camera_thread.daemon = True
                self.camera_thread.start()

            if self.processing_thread is None or not self.processing_thread.is_alive():
                self.processing_thread = threading.Thread(target=self.process_frames_loop)
                self.processing_thread.daemon = True
                self.processing_thread.start()

            # Update UI
            self.scan_button.configure(text="Stop Camera")
            self.capture_button.configure(state='normal')
            self.status_label.configure(text=f"Camera Active: {camera_name}")

            # Start frame updates
            self.update_frame()

        except Exception as e:
            logging.error(f"Camera initialization error: {str(e)}")
            messagebox.showerror("Camera Error", f"Failed to start camera: {str(e)}")
            self.stop_scanning()

    def stop_scanning(self):
        """Stop the scanning process and clean up resources"""
        if self.is_scanning:
            self.camera_active = False

            # Wait for threads to finish
            if self.camera_thread:
                self.camera_thread.join(timeout=1.0)
            if self.processing_thread:
                self.processing_thread.join(timeout=1.0)

            # Clear queues
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
                
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    break
                
            self.is_scanning = False
            self.scan_button.configure(text="Start Camera")
            self.capture_button.configure(state='disabled')
            self.status_label.configure(text="Camera: Stopped")

            if hasattr(self, 'cap'):
                self.cap.release()

            # Only show summary if there's actual data
            if self.excel_data and os.path.exists(self.current_excel_file):
                try:
                    df = pd.read_excel(self.current_excel_file)
                    if not df.empty:  # Only show summary if there's data
                        required_fields = {'ITEMID', 'QUANTITY', 'MANF_NAME', 'DATECODE', 'LOT'}
                        complete_rows = df[list(required_fields)].notna().all(axis=1).sum()
                        partial_rows = len(df) - complete_rows

                        # Show summary of completed file
                        messagebox.showinfo("Session Summary", 
                                        f"Session saved to {os.path.basename(self.current_excel_file)}\n"
                                        f"Complete entries: {complete_rows}\n"
                                        f"Partial entries: {partial_rows}")
                except Exception as e:
                    logging.error(f"Error reading Excel file: {str(e)}")

            # Reset current row data regardless of whether we saved
            self.current_row_data = {
                'Row': 1,
                'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Scanned_Barcodes': [],
                'Scanned_Text': {}
            }

            # Clear display tree
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Update status label
            if self.current_excel_file and os.path.exists(self.current_excel_file):
                self.save_status_label.config(text=f"Session: {os.path.basename(self.current_excel_file)}")
            else:
                self.save_status_label.config(text="Session: Not Started")

    def capture_data(self):
        """Modified capture_data with safer thread handling"""
        try:
            # Only proceed if we have data to save
            if self.current_row_data.get('Scanned_Barcodes') or self.current_row_data.get('Scanned_Text'):
                # Store the current camera state
                was_scanning = self.is_scanning

                # Temporarily pause scanning while we save
                self.is_scanning = False

                # Save current data to Excel
                self.append_row_to_excel()

                # Initialize new row data for next capture
                self.current_row_data = {
                    'Row': len(self.excel_data) + 1,
                    'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Scanned_Barcodes': [],
                    'Scanned_Text': {}
                }

                # Clear display tree for next capture
                for item in self.tree.get_children():
                    self.tree.delete(item)

                # Restore scanning state
                self.is_scanning = was_scanning

                # Keep camera and processing active
                if was_scanning:
                    if not self.camera_active:
                        self.camera_active = True

                        # Safely restart threads if needed
                        if self.camera_thread is None or not self.camera_thread.is_alive():
                            self.camera_thread = threading.Thread(target=self.camera_capture_loop)
                            self.camera_thread.daemon = True
                            self.camera_thread.start()

                        if self.processing_thread is None or not self.processing_thread.is_alive():
                            self.processing_thread = threading.Thread(target=self.process_frames_loop)
                            self.processing_thread.daemon = True
                            self.processing_thread.start()

                # Show success message
                self.status_label.configure(text="Data captured successfully - Continue scanning")

            else:
                messagebox.showwarning("No Data", "No data detected to capture")

        except Exception as e:
            logging.error(f"Error capturing data: {str(e)}")
            messagebox.showerror("Capture Error", f"Error capturing data: {str(e)}")

            # Try to recover camera operation
            try:
                if self.is_scanning:
                    self.camera_active = True
                    self.update_frame()
            except Exception as recovery_error:
                logging.error(f"Failed to recover after capture error: {str(recovery_error)}")


    def update_frame(self):
        """Modified update_frame to be more robust"""
        if self.camera_active and hasattr(self, 'cap') and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret:
                    processed_frame = self.process_frame(frame)

                    # Convert to PhotoImage for display
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    photo = ImageTk.PhotoImage(image=image)
                    self.video_label.configure(image=photo)
                    self.video_label.image = photo

                    # Schedule next update only if camera is still active
                    if self.camera_active:
                        self.root.after(10, self.update_frame)
                else:
                    logging.warning("Failed to read frame")
                    if self.camera_active:
                        self.root.after(100, self.update_frame)  # Retry after longer delay
            except Exception as e:
                logging.error(f"Error in update_frame: {str(e)}")
                if self.camera_active:
                    self.root.after(100, self.update_frame)  # Retry after longer delay

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
        Enhanced text processing with progressive data collection
        """
        try:
            # Parse structured text
            parsed_data = self.parse_structured_text(text)
            
            if parsed_data:
                # Initialize Scanned_Text if not present
                if 'Scanned_Text' not in self.current_row_data:
                    self.current_row_data['Scanned_Text'] = {}
                
                # Update the existing data with new values
                for key, value in parsed_data.items():
                    self.current_row_data['Scanned_Text'][key] = value
                
                # Update display with all collected data so far
                self.update_display()
                
                logging.info(f"Updated Text Data: {self.current_row_data['Scanned_Text']}")
                
        except Exception as e:
            logging.error(f"Text processing error: {e}")


    def update_display(self):
        """
        Update the treeview display with current data in vertical format
        """
        self.tree.delete(*self.tree.get_children())
    
        # Create a formatted string of all collected data
        collected_data = []
        if 'Scanned_Text' in self.current_row_data:
            for key, value in self.current_row_data['Scanned_Text'].items():
                collected_data.append(f"{key}:\n{value}")
    
        # Add barcode data if present
        if self.current_row_data.get('Scanned_Barcodes'):
            collected_data.append(f"Barcodes:\n{chr(10).join(self.current_row_data['Scanned_Barcodes'])}")
    
        # Join all data with newlines      
        display_text = '\n\n'.join(collected_data)
    
        self.tree.insert("", 0, values=(
            self.current_row_data.get('Row', 1),
            "Collected Data",
            display_text,
            self.current_row_data.get('Timestamp', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ))
    
        # Adjust row height to accommodate multiple lines
        item = self.tree.get_children()[0]
        text_height = len(display_text.split('\n'))
        self.tree.item(item, tags=('multiline',))
                    
                
    def create_new_excel_file(self):
        """Initialize Excel file path without creating the file"""
        try:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            date_dir = os.path.join(self.base_dir, current_date)
            os.makedirs(date_dir, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%H-%M-%S")
            excel_filename = f"scan_data_{timestamp}.xlsx"
            self.current_excel_file = os.path.join(date_dir, excel_filename)
            self.excel_data = []  # Reset excel data
            logging.info(f"Excel file path initialized: {excel_filename}")

        except Exception as e:
            logging.error(f"Error initializing Excel file path: {str(e)}")
            messagebox.showerror("File Initialization Error", f"Error initializing Excel file path: {str(e)}")

    def append_row_to_excel(self):
        """Append current row to Excel file, creating the file if it doesn't exist"""
        try:
            # Prepare row data
            base_row = {
                'Timestamp': self.current_row_data['Timestamp'],
                'Barcodes': ', '.join(self.current_row_data.get('Scanned_Barcodes', []))
            }
            
            # Add scanned text data
            if self.current_row_data.get('Scanned_Text'):
                for key, value in self.current_row_data['Scanned_Text'].items():
                    base_row[key] = value
            
            # Append to excel_data list
            self.excel_data.append(base_row)
            
            # Create DataFrame with required columns
            columns = [
                'Timestamp', 'Barcodes', 
                'GRN_NUMBER', 'ITEMID', 'QUANTITY', 
                'MANF_NAME', 'LOT', 'DATECODE',
                'ORDER_CODE', 'FLM', 'MARKING', 'ROHSCOMPLIANCE'
            ]
            
            if os.path.exists(self.current_excel_file):
                # If file exists, read it
                df = pd.read_excel(self.current_excel_file)
            else:
                # If file doesn't exist, create new DataFrame
                df = pd.DataFrame(columns=columns)
            
            # Create new row DataFrame
            new_row_df = pd.DataFrame([base_row])
            
            # Concatenate and save
            df = pd.concat([df, new_row_df], ignore_index=True)
            df.to_excel(self.current_excel_file, index=False)
            
            # Update status
            self.save_status_label.config(text=f"Row added to: {os.path.basename(self.current_excel_file)}")
            
        except Exception as e:
            logging.error(f"Error appending row: {str(e)}")
            messagebox.showerror("Save Error", f"Error appending row: {str(e)}")


    def save_session(self):
        try:
            # Prepare a list to store rows for saving
            rows_to_save = []
    
            # Required fields we're collecting
            required_fields = {'ITEMID', 'QUANTITY', 'MANF_NAME', 'DATECODE', 'LOT'}  # Added LOT
    
            # Iterate through sessions data
            for row_data in self.current_session_data:
                # Log the raw data for debugging
                logging.debug(f"Processing row data: {row_data}")
                
                # Base row with timestamp and barcodes
                base_row = {
                    'Timestamp': row_data.get('Timestamp', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    'Barcodes': ', '.join(row_data.get('Scanned_Barcodes', []))
                }
    
                # Get all collected text data
                if row_data.get('Scanned_Text'):
                    # Log the text data for debugging
                    logging.debug(f"Text data for row: {row_data['Scanned_Text']}")
                    
                    # Add all collected data to the row
                    for key, value in row_data['Scanned_Text'].items():
                        base_row[key] = value
                        logging.debug(f"Added {key}: {value} to base_row")
    
                rows_to_save.append(base_row)
    
            # If no data to save, return
            if not rows_to_save:
                messagebox.showinfo("No Data", "No records found")
                return
    
            # Create DataFrame
            df = pd.DataFrame(rows_to_save)
    
            # Ensure consistent column order with most important fields first
            desired_columns_order = [
                'Timestamp', 'Barcodes', 
                'GRN_NUMBER', 'ITEMID', 'QUANTITY', 
                'MANF_NAME', 'LOT', 'DATECODE',  # Changed LOT_NUMBER to LOT
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
    
            # Count how many rows have all required fields
            complete_rows = df[list(required_fields)].notna().all(axis=1).sum()
            partial_rows = len(df) - complete_rows
    
            messagebox.showinfo("Session Saved", 
                              f"Data saved to {excel_path}\n"
                              f"Complete entries: {complete_rows}\n"
                              f"Partial entries: {partial_rows}")
    
        except Exception as e:
            logging.error(f"Error saving session: {str(e)}")
            messagebox.showerror("Save Error", f"Error saving session data: {str(e)}")

            
    def on_closing(self):
        """Handle application closing without creating new file"""
        try:
            # If scanning is still active, stop it cleanly
            if self.is_scanning:
                self.camera_active = False
    
                # Stop the camera and clean up threads
                if self.camera_thread:
                    self.camera_thread.join(timeout=1.0)
                if self.processing_thread:
                    self.processing_thread.join(timeout=1.0)
    
                if hasattr(self, 'cap'):
                    self.cap.release()
    
            self.root.destroy()
    
        except Exception as e:
            logging.error(f"Error during closing: {str(e)}")
            messagebox.showerror("Close Error", f"Error during closing: {str(e)}")



def main():
    root = ThemedTk(theme="arc")
    app = UnifiedScannerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
