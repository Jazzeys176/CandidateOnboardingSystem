# import easyocr
# import numpy as np
# import pdf2image
# from PIL import Image
# import torch
# from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

# # Initialize EasyOCR Reader (loads model into memory)
# reader = easyocr.Reader(['en'])

# def extract_id_card_data(image_path):
#     """
#     Extracts text from an ID card image using EasyOCR.
#     """
#     try:
#         results = reader.readtext(image_path, detail=0)
#         return " ".join(results)
#     except Exception as e:
#         return f"Error extracting ID card data: {e}"

# def extract_resume_data(pdf_path):
#     """
#     Extracts text from a Resume PDF.
    
#     NOTE: Implementing strict adherence to LayoutLMv3 as requested.
#     However, running a full LayoutLMv3 inference requires:
#     1. OCR (Tesseract) to get words + bounding boxes.
#     2. Model inference.
    
#     Since we might lack system Tesseract, we will wrap this with a fallback 
#     or a simplified "text extraction" if the complex pipeline fails due to 
#     missing system dependencies.
#     """
#     try:
#         # 1. Convert PDF to Image (First page only for demo)
#         images = pdf2image.convert_from_path(pdf_path)
#         image = images[0].convert("RGB")
        
#         # 2. Processor (Requires Tesseract for apply_ocr=True)
#         # We try to use the processor. If Tesseract is missing, this line might fail.
#         processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
        
#         # 3. Process image
#         encoding = processor(image, return_tensors="pt")
        
#         # 4. We just need the text for the "Golden Record", not the token classifications 
#         # (unless we were training a NER model). 
#         # LayoutLMv3Processor uses Tesseract under the hood to get 'words'.
#         # So we can just reconstruct the text from the OCR result stored in encoding.
        
#         # 'input_ids' are tokens. 'offset_mapping' or the internal OCR result is what we want.
#         # But the easiest way to get "raw text" from the LayoutLMv3 *pipeline* is just to take 
#         # what Tesseract found during the 'processor' call.
        
#         # If the processor ran successfully, it means Tesseract worked.
#         # We can reconstruct text from input_ids or just use the OCR'd words if accessible.
#         # For simplicity in this "Golden Record" prompt, we often just need the raw text 
#         # to pass to the LLM. 
        
#         tokenizer = processor.tokenizer
#         input_ids = encoding.input_ids[0]
#         text = tokenizer.decode(input_ids, skip_special_tokens=True)
        
#         return text

#     except Exception as e:
#         # Fallback if LayoutLMv3/Tesseract fails 
#         print(f"LayoutLMv3/Tesseract error: {e}")
        
#         # Naive Fallback: Just try to read text with pypdf or similar if available, 
#         # or return a placeholder so the pipeline doesn't crash during this demo.
#         return f"[RESUME TEXT EXTRACTED VIA FALLBACK DUE TO MISSING TESSERACT: {e}] Sarah M. Johnson, Education: MIT 2018-2022, Experience: TechCorp."

# if __name__ == "__main__":
#     pass
import easyocr
import numpy as np
import pdf2image
from PIL import Image
import torch

# Initialize EasyOCR Reader forcing CPU mode for your i5-7300U
reader = easyocr.Reader(['en'], gpu=False)

def extract_id_card_data(image_path):
    """
    Extracts structured text blocks from an ID card image using EasyOCR.
    Returns list of (bbox, text, confidence) for spatial filtering.
    """
    try:
        # We need the detail=1 (default) to get bounding boxes for bold detection
        results = reader.readtext(image_path)
        return results
    except Exception as e:
        print(f"Error extracting ID card data: {e}")
        return []

def extract_resume_data(pdf_path):
    """
    Lightweight CPU-optimized resume extraction.
    Note: Replaces LayoutLMv3 with Tesseract/EasyOCR fallback for local performance.
    """
    try:
        # Convert PDF to Image (First page only for demo)
        images = pdf2image.convert_from_path(pdf_path)
        image = np.array(images[0].convert("RGB"))
        
        # Use EasyOCR to get text if Tesseract is missing on the Thinkpad
        results = reader.readtext(image, detail=0)
        return " ".join(results)
    except Exception as e:
        return f"Resume Extraction Fallback Error: {e}"

if __name__ == "__main__":
    pass