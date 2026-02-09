import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def generate_report(validation_json):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    client = Groq(api_key=api_key)

    prompt = f"""
    Act as a Technical Writer for HR Compliance. Take the following 'Validation Results' and generate a professional summary report.

    Report Requirements:

        Executive Summary: Start with a 'Risk Level' (LOW, MEDIUM, HIGH) based on the error percentage.

        Discrepancy Table: List ONLY fields marked 'INCORRECT' or 'AMBIGUOUS'. Include: 'Field Name', 'Candidate Claim', 'Verified Truth', and 'Explanation'.

        Source Attribution: Every explanation MUST state exactly where the verified data came from (e.g., 'Verified via Resume, Section: Education').

        Tone: Neutral, objective, and professional. Use Markdown for structure.

    Validation Results: 
    {validation_json}
    """

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs Markdown reports."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        stream=False,
    )

    return completion.choices[0].message.content

if __name__ == "__main__":
    pass
