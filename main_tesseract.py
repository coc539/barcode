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
        
        self.setup_ui()

        # Extend scanned_items to support more detailed text parsing
        self.scanned_items = {}
        
        # Add a method to parse structured text
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
        
        self.parse_structured_text = parse_structured_text
        
        
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
        self.tree = ttk.Treeview(items_frame, columns=("Type", "Data", "Timestamp"), show="headings")
        self.tree.heading("Type", text="Type")
        self.tree.heading("Data", text="Data")
        self.tree.heading("Timestamp", text="Timestamp")
        
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
        self.session_dir = self.create_session_directory()
        self.scanned_items = {}
        self.last_scanned_time = {}
        
        # Clear previous items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        self.is_scanning = True
        self.camera_active = True
        self.scan_button.configure(text="Stop Scanning")
        self.status_label.configure(text="Camera: Active")
        session_time = self.session_start_time.strftime("%H:%M:%S")
        self.save_status_label.configure(text=f"Session started: {session_time}")
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        self.update_frame()

    def stop_scanning(self):
        if self.is_scanning:
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
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Process barcodes
        barcodes = pyzbar.decode(frame)
        for barcode in barcodes:
            self.process_barcode(barcode)
            # Draw rectangle around barcode
            (x, y, w, h) = barcode.rect
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, "Barcode", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
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
        
        return display_frame

    def process_barcode(self, barcode):
        barcode_data = barcode.data.decode("utf-8")
        current_time = time.time()
        
        if barcode_data not in self.scanned_items or \
                (current_time - self.last_scanned_time.get(barcode_data, 0)) > self.debounce_time:
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.scanned_items[barcode_data] = {
                "Type": f"Barcode ({barcode.type})",
                "Timestamp": timestamp
            }
            self.last_scanned_time[barcode_data] = current_time
            
            self.tree.insert("", 0, values=(f"Barcode ({barcode.type})", barcode_data, timestamp))

    def process_text(self, text):
           current_time = time.time()

           # Parse structured text
           parsed_data = self.parse_structured_text(text)

           if parsed_data:
               # Create a unique key, prioritizing GRN or Batch Number
               unique_key = parsed_data.get('GRN_NUMBER') or parsed_data.get('BATCH_NUMBER') or text

               if unique_key not in self.scanned_items or \
                       (current_time - self.last_scanned_time.get(unique_key, 0)) > self.debounce_time:

                   timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                   # Store the parsed data
                   self.scanned_items[unique_key] = {
                       "Type": "Structured Text",
                       "Timestamp": timestamp,
                       "ParsedData": parsed_data
                   }
                   self.last_scanned_time[unique_key] = current_time

                   # Insert into treeview with more detailed information
                   display_text = f"GRN: {parsed_data.get('GRN_NUMBER', 'N/A')} | Item: {parsed_data.get('ITEMID', 'N/A')}"
                   self.tree.insert("", 0, values=("Structured Text", display_text, timestamp))

    def save_session(self):
            try:
                if not self.scanned_items:
                    messagebox.showinfo("Session End", "No items were scanned in this session")
                    return

                # Prepare data with separate columns for structured text
                data = []
                for item_data, details in self.scanned_items.items():
                    if details.get("Type") == "Structured Text":
                        parsed_data = details.get("ParsedData", {})
                        row = {
                            "Timestamp": details["Timestamp"],
                            "GRN_NUMBER": parsed_data.get('GRN_NUMBER', ''),
                            "ITEMID": parsed_data.get('ITEMID', ''),
                            "QUANTITY": parsed_data.get('QUANTITY', ''),
                            "FLM": parsed_data.get('FLM', ''),
                            "MANF_NAME": parsed_data.get('MANF_NAME', ''),
                            "ORDER_CODE": parsed_data.get('ORDER_CODE', ''),
                            "BATCH_NUMBER": parsed_data.get('BATCH_NUMBER', ''),
                            "DATECODE": parsed_data.get('DATECODE', ''),
                            "LOT": parsed_data.get('LOT', ''),
                            "MARKING": parsed_data.get('MARKING', ''),
                            "ROHSCOMPLIANCE": parsed_data.get('ROHSCOMPLIANCE', ''),
                        }
                        data.append(row)
                    elif details.get("Type", "").startswith("Barcode"):
                        row = {
                            "Timestamp": details["Timestamp"],
                            "Barcode": item_data,
                        }
                        data.append(row)

                # Continue with the rest of the save_session method as before
                if not data:
                    messagebox.showinfo("Session End", "No items were scanned in this session")
                    return

                df = pd.DataFrame(data)

                # Sort by timestamp
                df = df.sort_values("Timestamp", ascending=False)

                # Calculate statistics for session info
                structured_text_count = len([row for row in data if 'GRN_NUMBER' in row])
                barcode_count = len([row for row in data if 'Barcode' in row])

                # Save session details
                session_info = {
                    "Session Start": self.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Session End": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Total Structured Text": structured_text_count,
                    "Total Barcodes": barcode_count,
                    "Total Scans": len(data)
                }
                info_df = pd.DataFrame([session_info])

                # Save to Excel with multiple sheets
                excel_path = os.path.join(self.session_dir, "scan_data.xlsx")
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Scanned Items', index=False)
                    info_df.to_excel(writer, sheet_name='Session Info', index=False)

                    # Auto-adjust column widths
                    for sheet in writer.sheets.values():
                        for column in sheet.columns:
                            max_length = 0
                            column = [cell for cell in column]
                            for cell in column:
                                try:
                                    if len(str(cell.value)) > max_length:
                                        max_length = len(cell.value)
                                except:
                                    pass
                            adjusted_width = (max_length + 2)
                            sheet.column_dimensions[column[0].column_letter].width = adjusted_width

                messagebox.showinfo("Session End",
                                  f"Session data saved successfully\n"
                                  f"Location: {excel_path}\n"
                                  f"Total scans: {len(data)}\n"
                                  f"Structured Text: {structured_text_count}\n"
                                  f"Barcodes: {barcode_count}")

            except Exception as e:
                messagebox.showerror("Save Error", f"Error saving session data: {str(e)}")

def main():
    # Set up Tesseract path (modify this according to your installation)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
    # For Linux/Mac, you might not need this line if Tesseract is in PATH
    
    root = ThemedTk(theme="arc")
    app = UnifiedScannerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()