import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import pytesseract
from pyzbar import pyzbar
import pandas as pd
from PIL import Image, ImageTk
import datetime
import time
import numpy as np
from pathlib import Path
import os
from ttkthemes import ThemedTk
import re

class UnifiedScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Unified Scanner")
        self.root.set_theme("arc")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Initialize variables
        self.is_scanning = False
        self.camera_active = False
        self.scanned_items = {}  # Combined dictionary for both barcodes and text
        self.last_scanned_time = {}
        self.debounce_time = 3
        self.session_start_time = None
        
        # Create base directory for scans
        self.base_dir = "scan_results"
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Session tracking and data storage
        self.current_session_data = []
        self.min_contrast_threshold = 30
        self.min_sharpness_threshold = 50
        self.blur_threshold = 100

        self.setup_ui()
        
        # Setup structured text parsing method
        self.parse_structured_text = self.create_parse_structured_text_method()
        
        # Bind application close event to save session
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.current_session_data = []
        self.current_row_data = {} 
        
        # Add a method to parse structured text
    def create_parse_structured_text_method(self):
        def parse_structured_text(text):
            """
            Parse text with predefined structure
            Returns a dictionary with parsed fields
            """
            # Remove any extra whitespace and normalize
            text = re.sub(r'\s+', ' ', text.strip())
            
            # Regular expression patterns for different fields
            patterns = {
                'GRN_NUMBER': r'GRN\s*[:#]?\s*(\w+)',
                'ITEMID': r'ITEMID\s*[:#]?\s*(\w+)',
                'QUANTITY': r'QTY\s*[:#]?\s*(\d+)',
                'FLM': r'FLM\s*[:#]?\s*(\w+)',
                'MANF_NAME': r'MANF\s*[:#]?\s*([^O]+)',
                'ORDER_CODE': r'ORDER\s*[:#]?\s*(\w+)',
                'BATCH_NUMBER': r'BATCH\s*[:#]?\s*(\w+)',
                'DATECODE': r'DATE\s*[:#]?\s*(\d{2,4}[-/]\d{2,4}[-/]\d{2,4})',
                'LOT': r'LOT\s*[:#]?\s*(\w+)',
                'MARKING': r'MARKING\s*[:#]?\s*([^R]+)',
                'ROHSCOMPLIANCE': r'ROHS\s*[:#]?\s*(\w+)'
            }
            
            parsed_data = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    parsed_data[key] = match.group(1).strip()
                else:
                    parsed_data[key] = ''
            
            # If no structured data found, return None
            if not any(parsed_data.values()):
                return None
            
            return parsed_data
        
        return parse_structured_text

        
    def assess_frame_quality(self, frame):
        """
        Assess the quality of a frame based on multiple metrics
        Returns a quality score between 0 and 100
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Assess contrast
        contrast = gray.std()
        
        # Assess sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Assess blur using standard deviation of Laplacian
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate quality components
        contrast_score = min(contrast / self.min_contrast_threshold * 100, 100)
        sharpness_score = min(laplacian_var / self.min_sharpness_threshold * 100, 100)
        blur_score = max(100 - (blur_value / self.blur_threshold * 100), 0)
        
        # Combine scores (weighted average)
        quality_score = (
            0.4 * contrast_score + 
            0.4 * sharpness_score + 
            0.2 * blur_score
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
            cv2.putText(display_frame, "Barcode", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            data_found = True
        
        # Process text
        # Convert to grayscale for OCR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                roi = thresh[y:y+h, x:x+w]
                text = pytesseract.image_to_string(roi).strip()
                
                if text and len(text) > 3:  # Filter out very short text
                    self.process_text(text)
                    # Draw rectangle around text
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(display_frame, "Text", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    data_found = True
        
        # Optionally, add quality score to the frame for debugging
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
        # Parse structured text
        parsed_data = self.parse_structured_text(text)

        if parsed_data:
            # Store text as full object, not just display
            if parsed_data not in [item.get('text_data', {}) for item in self.current_row_data.get('Scanned_Text', [])]:
                # Add text to current row's scanned items if not already present
                if 'Scanned_Text' not in self.current_row_data:
                    self.current_row_data['Scanned_Text'] = []
                
                self.current_row_data['Scanned_Text'].append({
                    'text_data': parsed_data,
                    'raw_text': text
                })
                
                # Update display
                self.tree.delete(*self.tree.get_children())  # Clear existing entries
                self.tree.insert("", 0, values=(
                    self.current_row_data.get('Row', 1), 
                    "Structured Text", 
                    f"GRN: {parsed_data.get('GRN_NUMBER', 'N/A')}, "
                    f"Item: {parsed_data.get('ITEMID', 'N/A')}", 
                    self.current_row_data.get('Timestamp', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                ))
                
    def save_session(self):
        try:
            # Prepare a list to store rows for saving
            rows_to_save = []

            # Iterate through sessions data
            for row_data in self.current_session_data:
                base_row = {
                    'Timestamp': row_data.get('Timestamp', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    'Barcodes': ', '.join(row_data.get('Scanned_Barcodes', []))
                }

                # If no data, skip this row
                if not base_row['Barcodes'] and not row_data.get('Scanned_Text'):
                    continue

                # Handle multiple text entries
                if row_data.get('Scanned_Text'):
                    for i, text_entry in enumerate(row_data['Scanned_Text'], 1):
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

            # Update UI status
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
    # Set up Tesseract path (modify this according to your installation)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
    # For Linux/Mac, you might not need this line if Tesseract is in PATH
    
    root = ThemedTk(theme="arc")
    app = UnifiedScannerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()