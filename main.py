import streamlit as st
from paddleocr import PaddleOCR
from pdf2image import convert_from_bytes
import pdfplumber
import docx2txt
import json
import numpy as np
from PIL import Image
import cv2
import io
import re
from typing import List, Dict, Tuple, Generator
import logging
import os
from datetime import datetime

# Configure logging
logging.getLogger('').setLevel(logging.ERROR)

class DocumentProcessor:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        self.chunk_size = 500
        self.segment_height = 800
        self.segment_width = 800
        self.segment_overlap = 40
        self.output_dir = "processed_documents"
        self.ensure_output_directory()

    def ensure_output_directory(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_processed_text(self, text: str, filename: str) -> str:
        """Save processed text to file with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_{timestamp}.txt"
        output_path = os.path.join(self.output_dir, output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            st.success(f"Processed text saved to: {output_path}")
            return output_path
        except Exception as e:
            st.error(f"Error saving text file: {str(e)}")
            return None

    def process_image_segment(self, segment_data: Tuple[Image.Image, Tuple[int, int]]) -> List[dict]:
        """Process a single image segment."""
        segment, (x_offset, y_offset) = segment_data
        segment_np = np.array(segment)
        
        try:
            result = self.ocr.ocr(segment_np, cls=True)
            if not result or not result[0]:
                return []
            
            processed_results = []
            for line in result[0]:
                if isinstance(line, (list, tuple)) and len(line) == 2:
                    coords, (text, conf) = line
                    if text and text.strip():
                        adjusted_coords = [[x + x_offset, y + y_offset] for x, y in coords]
                        processed_results.append({
                            'coords': adjusted_coords,
                            'text': text.strip(),
                            'confidence': conf
                        })
            
            return processed_results
        except Exception as e:
            st.warning(f"Segment processing error: {str(e)}")
            return []

    def extract_text_from_image(self, image: Image.Image, filename: str) -> str:
        """Extract text from image and save to file."""
        try:
            all_results = []
            segment_count = 0
            
            # Calculate total segments for progress bar
            width, height = image.size
            total_segments = ((width // (self.segment_width - self.segment_overlap)) + 1) * \
                           ((height // (self.segment_height - self.segment_overlap)) + 1)
            
            progress_bar = st.progress(0)
            
            for y in range(0, height, self.segment_height - self.segment_overlap):
                for x in range(0, width, self.segment_width - self.segment_overlap):
                    segment_count += 1
                    progress_bar.progress(segment_count / total_segments)
                    
                    right = min(x + self.segment_width, width)
                    bottom = min(y + self.segment_height, height)
                    segment = image.crop((x, y, right, bottom))
                    
                    results = self.process_image_segment((segment, (x, y)))
                    if results:
                        all_results.extend(results)
                        
                        # Save intermediate results
                        intermediate_text = self._generate_text_from_results(all_results)
                        self.save_processed_text(intermediate_text, f"intermediate_{filename}")
            
            final_text = self._generate_text_from_results(all_results)
            return final_text
            
        except Exception as e:
            st.error(f"Image extraction error: {str(e)}")
            return ""

    def _generate_text_from_results(self, results: List[dict]) -> str:
        """Generate text from OCR results."""
        if not results:
            return ""
            
        sorted_results = sorted(results, key=lambda x: (x['coords'][0][1] if x['coords'] else 0))
        text_parts = []
        prev_y = None
        
        for result in sorted_results:
            curr_y = result['coords'][0][1] if result['coords'] else 0
            
            if prev_y is not None and curr_y - prev_y > 50:
                text_parts.append('\n')
            
            text_parts.append(result['text'])
            prev_y = curr_y
        
        return ' '.join(text_parts)

    def extract_text_from_pdf(self, pdf_file, filename: str) -> str:
        """Extract text from PDF and save chunks."""
        try:
            # Try extracting text directly first
            with pdfplumber.open(pdf_file) as pdf:
                text_parts = []
                total_pages = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    st.write(f"Processing page {page_num} of {total_pages}...")
                    extracted = page.extract_text()
                    if extracted and len(extracted.strip()) > 0:
                        text_parts.append(extracted)
                        
                        # Save intermediate results
                        intermediate_text = '\n'.join(text_parts)
                        self.save_processed_text(intermediate_text, f"intermediate_{filename}")
                
                if text_parts:
                    return '\n'.join(text_parts)

            # Fall back to OCR if needed
            pdf_file.seek(0)
            st.write("Converting PDF to images...")
            images = convert_from_bytes(pdf_file.read(), dpi=300)
            
            text_parts = []
            for page_num, image in enumerate(images, 1):
                st.write(f"OCR Processing page {page_num} of {len(images)}...")
                text_part = self.extract_text_from_image(image, f"page_{page_num}_{filename}")
                if text_part:
                    text_parts.append(text_part)
                    
                    # Save intermediate results after each page
                    intermediate_text = '\n\n'.join(text_parts)
                    self.save_processed_text(intermediate_text, f"intermediate_{filename}")
            
            return '\n\n'.join(text_parts)

        except Exception as e:
            st.error(f"PDF processing error: {str(e)}")
            return ""

def main():
    st.title("📄 Advanced Real Estate Document Processor")
    st.write("Upload a document to extract key real estate details.")

    processor = DocumentProcessor()
    
    uploaded_file = st.file_uploader(
        "Upload Document (PDF, DOCX, JPEG, PNG)", 
        type=["pdf", "docx", "jpg", "jpeg", "png"]
    )

    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        with st.spinner("Processing document..."):
            try:
                if file_extension == "pdf":
                    extracted_text = processor.extract_text_from_pdf(uploaded_file, uploaded_file.name)
                elif file_extension == "docx":
                    extracted_text = docx2txt.process(uploaded_file)
                    processor.save_processed_text(extracted_text, uploaded_file.name)
                else:  # Image files
                    image = Image.open(uploaded_file)
                    extracted_text = processor.extract_text_from_image(image, uploaded_file.name)

                if not extracted_text.strip():
                    st.error("No text could be extracted from the document.")
                    return

                # -----------------------------------------
                # Extract key real estate details using regex
                # -----------------------------------------
                text = extracted_text  # Or rename for clarity

                extracted_info = {
                    "date_of_sale": next(iter(re.findall(r"Date of sale\s*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)), "Not Found"),
                    "seller_name": next(iter(re.findall(r"Seller\s+(?:is\s+)?([^,\n]+)", text, re.IGNORECASE)), "Not Found"),
                    "buyer_name": next(iter(re.findall(r"Buyer\s+(?:is\s+)?([^,\n]+)", text, re.IGNORECASE)), "Not Found"),
                    "sales_price": next(iter(re.findall(r"\$[\d,]+(?:\.?\d{2})?", text)), "Not Found"),
                    "registry_number": next(iter(re.findall(r"Registry Number\s*([\d,]+)", text, re.IGNORECASE)), "Not Found"),
                    "registry_section": next(iter(re.findall(r"(\d+)(?:st|nd|rd|th)\s+section\s+of\s+([^,\n]+)", text, re.IGNORECASE)), ("Not Found", "Not Found")),
                    "cadaster": next(iter(re.findall(r"Cadaster\s*([\d-]+)", text, re.IGNORECASE)), "Not Found"),
                    "unit": next(iter(re.findall(r"Unit\s*(\d+)[^\d]*([^,\n]+)", text, re.IGNORECASE)), ("Not Found", "Not Found")),
                    "bedrooms": next(iter(re.findall(r"(\d+)\s*bedroom", text, re.IGNORECASE)), "Not Found"),
                    "bathrooms": next(iter(re.findall(r"(\d+)\s*bathroom", text, re.IGNORECASE)), "Not Found"),
                    "area_sf": next(iter(re.findall(r"([\d,]+\.?\d*)\s*(?:square feet|[Ss]\.?[Ff]\.?)", text)), "Not Found"),
                    "area_sm": next(iter(re.findall(r"([\d,]+\.?\d*)\s*(?:square meters|[Ss][Mm])", text)), "Not Found"),
                    "parking_spaces": re.findall(r"(?:parking|park)\s*(?:space|spot)?\s*(?:numbers?)?\s*((?:\d+(?:[,\s]+(?:and\s+)?\d+)*)|(?:[A-Z]\d+))", text, re.IGNORECASE)
                }

                # -----------------------------------------
                # Save final extracted text
                # -----------------------------------------
                final_text_path = processor.save_processed_text(extracted_text, uploaded_file.name)
                
                if final_text_path:
                    st.success(f"Full text saved to: {final_text_path}")
                    
                    # Show text preview in expander
                    with st.expander("Show Extracted Text"):
                        st.text_area(
                            "Raw text:",
                            extracted_text[:10000] + "..." if len(extracted_text) > 10000 else extracted_text, 
                            height=200
                        )

                    # Add download button for text file
                    with open(final_text_path, 'r', encoding='utf-8') as f:
                        st.download_button(
                            label="Download Extracted Text",
                            data=f.read(),
                            file_name=os.path.basename(final_text_path),
                            mime="text/plain"
                        )

                    # Display the extracted info
                    st.subheader("Extracted Real Estate Details")
                    st.json(extracted_info)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
