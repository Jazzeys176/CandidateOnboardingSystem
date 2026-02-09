import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def build_golden_record(resume_text, id_text):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    client = Groq(api_key=api_key)

    prompt = f"""
    Act as a Multimodal Data Engineer. Your goal is to build a candidate Knowledge Base (KB). Inputs: 1. Raw text from LayoutLMv3 (Resume) 2. Extracted strings from EasyOCR (ID Card). Task: Merge these into a single JSON 'Golden Record'. Instructions:

        Normalize dates to YYYY-MM-DD.

        Extract National ID specifically from the EasyOCR data.

        If Education dates in Resume conflict with HR notes, prioritize the Resume.

        Output Schema: {{personal_details: {{}}, education: [], employment: [], identity_verification: {{id_type, id_number}}}}.

        No conversational filler. Output ONLY the JSON.
    
    [Resume (LayoutLMv3 Text)]
    {resume_text}

    [ID Card (EasyOCR Text)]
    {id_text}
    """

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs only JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        stream=False,
        response_format={"type": "json_object"}
    )

    return completion.choices[0].message.content

if __name__ == "__main__":
    pass
