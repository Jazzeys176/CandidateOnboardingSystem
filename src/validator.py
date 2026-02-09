import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def validate_candidate_data(kb_json, form_json):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    client = Groq(api_key=api_key)

    prompt = f"""
    Act as a Fact-Checking Assistant. You are provided with a 'Knowledge Base' (the truth) and an 'Onboarding Form' (the candidate's claim).

    Validation Rules:

        Classification:

            CORRECT: Exact or highly similar match (e.g., 'SDE' vs 'Software Engineer').

            AMBIGUOUS: Partial match or missing data in KB to confirm.

            INCORRECT: Direct contradiction (e.g., Claiming 3.9 GPA when KB says 3.8).

        Reasoning: For every field, provide a 'reason' citing the specific source from the KB.

        Thinking Process: Think step-by-step: compare the string, check for semantic synonyms, and then assign the label.

        Constraint: If you are unsure, you MUST label it 'AMBIGUOUS'. Do not guess.

    Data: 
    KB: 
    {kb_json}
    
    Form: 
    {form_json}
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
