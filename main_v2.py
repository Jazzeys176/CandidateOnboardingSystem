import os
import sys
import json
from dotenv import load_dotenv
from src.ocr_engine import extract_id_card_data, extract_resume_data
# We now use the updated extractor which handles ID data merging
from src.extractor import extract_id_fields, extract_candidate_data

def main_v2():
    load_dotenv()
    
    # Ensure API Key is set
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY is not set in environment or .env file.")
        sys.exit(1)
        
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # User requested to provide data
    resume_path = os.path.join(base_dir, "inputs", "Resume.pdf")
    id_card_path = os.path.join(base_dir, "inputs", "pan_father.jpeg")

    # Check file existence
    if not os.path.exists(resume_path):
        print(f"Warning: {resume_path} not found.")
    if not os.path.exists(id_card_path):
        print(f"Warning: {id_card_path} not found.")

    resume_text = ""
    if os.path.exists(resume_path):
        print(f"Processing Resume: {resume_path}")
        resume_text = extract_resume_data(resume_path)
        print(f"Resume Text Preview: {resume_text[:100]}...")

    id_data = None
    if os.path.exists(id_card_path):
        print(f"\nProcessing ID Card: {id_card_path}")
        # Now returns list of (bbox, text, prob)
        ocr_results = extract_id_card_data(id_card_path) 
        
        # Apply regex logic
        print("\nApplying Regex Extraction to ID Card Data...")
        id_data = extract_id_fields(ocr_results)
    
    # Generate Golden Record
    print("\nBuilding Golden Record (Resume + ID)...")
    golden_record = extract_candidate_data(resume_text, transcript_text="", id_data=id_data)
    
    print("\n--- Golden Record (KB) ---\n")
    print(golden_record)

    # Save to file for inspection
    output_path = os.path.join(base_dir, "golden_record.json")
    with open(output_path, "w") as f:
        f.write(golden_record)
    print(f"\nSaved Golden Record to {output_path}")

if __name__ == "__main__":
    main_v2()
