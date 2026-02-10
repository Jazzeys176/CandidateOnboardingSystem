# import os
# import json
# from groq import Groq
# from dotenv import load_dotenv

# load_dotenv()

# def extract_candidate_data(resume_text, transcript_text):
#     api_key = os.getenv("GROQ_API_KEY")
#     if not api_key:
#         raise ValueError("GROQ_API_KEY environment variable not set")
    
#     client = Groq(api_key=api_key)

#     prompt = f"""
#     Act as a Senior Data Extraction Engineer specializing in HR tech. Your task is to process a candidate's resume and HR interview notes into a structured JSON knowledge base.

#     Instructions:

#         Extraction Only: Extract data only if it is explicitly stated in the provided text. If a field (e.g., GPA) is missing, return null. NEVER invent data.

#         Structure: Output must be a single, valid JSON object following this schema:

#             personal_details: {{name, email, phone, location}}

#             education: list of {{degree, institution, graduation_year, gpa}}

#             employment: list of {{company, title, start_date, end_date}}

#         Conflict Resolution: If the Resume and HR Notes contradict, prioritize the Resume for dates and the HR Notes for 'current' status.

#         Constraint: Do not include any conversational text. Output only the JSON.

#     Input Context:
#     [Resume Text]
#     {resume_text}

#     [HR Call Transcript]
#     {transcript_text}
#     """

#     completion = client.chat.completions.create(
#         model="llama-3.3-70b-versatile",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant that outputs only JSON."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0,
#         stream=False,
#         response_format={"type": "json_object"}
#     )

#     return completion.choices[0].message.content

# if __name__ == "__main__":
#     # Test execution
#     pass
import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def extract_id_fields(ocr_results):
    """
    Logic-based filter to isolate Aadhar and Pincode from OCR blocks.
    """
    # Pattern: 12 digits, can have spaces (Aadhar)
    aadhar_pattern = r'[2-9][0-9]{3}\s[0-9]{4}\s[0-9]{4}'
    # Pattern: PAN (5 letters, 4 numbers, 1 letter)
    pan_pattern = r'[A-Z]{5}[0-9]{4}[A-Z]{1}'
    # Pattern: Pincode
    pincode_pattern = r'\b[1-9][0-9]{5}\b'
    
    extracted = {"id_type": None, "id_number": None, "pincode": None}
    max_id_height = 0  # To track the largest (boldest) ID number
    
    print("\n--- Starting OCR Data Extraction Log (Size/Bold Logic) ---")
    for i, (bbox, text, prob) in enumerate(ocr_results):
        # Clean text: strip spaces and convert to uppercase for PAN matching
        clean_text_raw = text.strip()
        clean_text_upper = clean_text_raw.upper().replace(" ", "")
        
        # Calculate Height
        height = abs(bbox[2][1] - bbox[1][1])
        
        # Log every block processed
        print(f"Block {i}: '{clean_text_raw}' (H: {height:.2f}, Conf: {prob:.2f})")
        
        # 1. Check for Aadhar (12 digits, spaces allowed)
        if re.search(aadhar_pattern, clean_text_raw):
            print(f"   >>> POTENTIAL AADHAR: {clean_text_raw} | Height: {height:.2f}")
            if height > max_id_height:
                extracted["id_type"] = "Aadhar"
                extracted["id_number"] = clean_text_raw
                max_id_height = height
                print(f"       >>> UPDATING CANDIDATE (New Max Height: {max_id_height:.2f})")
        
        # 2. Check for PAN (10 chars, no spaces)
        elif re.match(pan_pattern, clean_text_upper): # Use match or search, clean_text_upper has no spaces
             # If using search on raw text, regex needs to allow spaces? usually PAN is one block.
             # Better to check clean_text_upper for the pattern [A-Z]{5}[0-9]{4}[A-Z]
             
             # Re-checking regex on clean_text_upper
             if re.search(pan_pattern, clean_text_upper):
                print(f"   >>> POTENTIAL PAN: {clean_text_upper} | Height: {height:.2f}")
                if height > max_id_height:
                    extracted["id_type"] = "PAN"
                    extracted["id_number"] = clean_text_upper
                    max_id_height = height
                    print(f"       >>> UPDATING CANDIDATE (New Max Height: {max_id_height:.2f})")

        # 3. Check for Pincode
        if re.search(pincode_pattern, clean_text_raw) and len(clean_text_raw) == 6:
            extracted["pincode"] = clean_text_raw
            print(f"   >>> MATCH FOUND: Pincode -> {extracted['pincode']}")
            
    print("--- Final Extracted ID Data ---")
    print(json.dumps(extracted, indent=4))
    print("----------------------------------------\n")
    return extracted

def extract_candidate_data(resume_text, transcript_text, id_data=None, form_data=None):
    """
    Combines OCR extracted ID data, Resume, Transcript, and Form Data into a detailed Golden Record.
    """
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)

    prompt = f"""
    Act as a Senior AI Solutions Architect. Create a consolidated JSON 'Golden Record' for a candidate.
    
    Data Sources:
    1. [Resume] (Source: OCR)
    {resume_text}
    
    2. [ID Card Data] (Source: Spatial OCR)
    {json.dumps(id_data) if id_data else 'None'}
    
    3. [Onboarding Form Data] (Source: User Input)
    {json.dumps(form_data) if form_data else 'None'}
    
    4. [HR Transcript] (Source: Uploaded File)
    {transcript_text}

    Conflict Resolution Rules:
    - Identity (Name, ID Numbers): Prioritize [ID Card Data] (especially Aadhar/PAN).
    - Current Status (Availability, Current Job): Prioritize [HR Transcript].
    - Employment/Education History: Prioritize [Resume].
    - Contact Info: Prioritize [Onboarding Form Data] if available, else Resume.

    Output Schema:
    The output must be a single valid JSON object with the following structure:
    {{
      "personal_details": {{
        "name": "...",
        "email": "...",
        "phone": "...",
        "id_type": "...",
        "id_number": "...",
        "address": "...",
        "pincode": "..."
      }},
      "education": [
        {{ "institution": "...", "degree": "...", "year": "...", "score": "..." }}
      ],
      "employment": [
        {{ "company": "...", "role": "...", "start_date": "...", "end_date": "..." }}
      ],
      "metadata": {{
        "sources_verified": ["Resume", "ID_Card", "Transcript", "Form"]
      }}
    }}
    
    Constraint: Output ONLY valid JSON.
    """

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    return completion.choices[0].message.content