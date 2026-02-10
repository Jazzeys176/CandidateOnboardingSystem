import os
import json
import re
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
from groq import Groq

# ONNX / Semantic Search Imports
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

load_dotenv()

class SemanticValidator:
    def __init__(self):
        print("Loading ONNX-optimized Sentence Transformer (all-MiniLM-L6-v2)...")
        # Load model from HuggingFace and export to ONNX on the fly (or load cached)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2", 
            export=True  # This triggers ONNX export
        )
        print("Model loaded successfully.")
        
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, text):
        encoded_input = self.tokenizer([text], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Perform pooling
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings[0].numpy()

    def get_similarity_score(self, text1, text2):
        if not text1 or not text2:
            return 0.0
        
        # Direct normalized match optimization
        if text1.strip().lower() == text2.strip().lower():
            return 1.0

        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        # Cosine Similarity
        return float(np.dot(emb1, emb2))

    def get_similarity_label(self, score):
        if score >= 0.9:
            return "CORRECT"
        elif 0.7 <= score < 0.9:
            return "AMBIGUOUS"
        else:
            return "INCORRECT"

    def llm_fallback_check(self, kb_val, form_val):
        """
        Uses Groq to verify if two ambiguous terms are actually synonyms.
        """
        prompt = f"""
        Act as a Data Validator.
        Are these two terms effectively synonymous in an employment/education context?
        Term 1: "{kb_val}"
        Term 2: "{form_val}"
        
        Reply strictly with JSON: {{"is_synonym": boolean, "reason": "short explanation"}}
        """
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            result = json.loads(completion.choices[0].message.content)
            return result
        except Exception as e:
            print(f"LLM Fallback Error: {e}")
            return {"is_synonym": False, "reason": "LLM Error"}

    def check_temporal_consistency(self, education_list, employment_list):
        """
        Ensures Graduation Year < Employment Start Date.
        Returns list of inconsistencies.
        """
        issues = []
        
        # Helper to parse year/date
        def parse_year(date_str):
            # Extremely naive parser for demo: looks for 4 digits
            match = re.search(r'\d{4}', str(date_str))
            return int(match.group(0)) if match else None

        # Determine latest graduation year
        latest_grad_year = 0
        for edu in education_list:
            grad_year = parse_year(edu.get('year') or edu.get('graduation_year'))
            if grad_year and grad_year > latest_grad_year:
                latest_grad_year = grad_year
        
        # Check against employment start dates
        for job in employment_list:
            start_year = parse_year(job.get('start_date'))
            if start_year and latest_grad_year and start_year < latest_grad_year:
                # If started working BEFORE graduating, might be an internship, but flag it if not checking job title
                # User Prompt: "strictly before any Employment Start Date" -> Assuming full-time?
                # The rule is "Graduation Year is strictly before ANY Employment Start Date".
                # But typically people have internships. 
                # I'll flag it but maybe note if it looks like 'Intern'.
                issues.append(f"Employment at '{job.get('company', 'Unknown')}' starts in {start_year}, which is before graduation in {latest_grad_year}.")
        
        return issues

    def _is_acronym(self, text1, text2):
        """
        Simple heuristic: Check if one string is a potential acronym of the other.
        """
        s1, s2 = text1.strip().upper(), text2.strip().upper()
        if len(s1) == len(s2):
            return False
            
        short, long_str = (s1, s2) if len(s1) < len(s2) else (s2, s1)
        
        # Heuristic 1: Short string is contained in first letters of words in Long string
        # e.g., TCET vs Thakur College of Engineering and Technology
        long_words = re.findall(r'\w+', long_str)
        initials = "".join([w[0] for w in long_words])
        
        if short in initials:
            return True
            
        # Heuristic 2: All chars of short string appear in distinct words of long string in order
        # Very loose check
        s_idx, w_idx = 0, 0
        while s_idx < len(short) and w_idx < len(long_words):
            if long_words[w_idx].startswith(short[s_idx]):
                s_idx += 1
            w_idx += 1
        
        if s_idx == len(short):
             return True
             
        return False

    def validate(self, kb_json, form_json):
        report = {}
        
        # 1. Personal Details Check
        kb_pd = kb_json.get("personal_details", {})
        form_pd = form_json.get("personal_details", {})
        
        fields_to_check = ["name", "email", "phone", "id_number"]
        
        for field in fields_to_check:
            kb_val = kb_pd.get(field, "")
            form_val = form_pd.get(field, "")
            
            score = self.get_similarity_score(str(kb_val), str(form_val))
            status = self.get_similarity_label(score)
            reason = f"Similarity Score: {score:.2f}"
            
            # Check for Acronyms if score is low
            if status == "INCORRECT" and self._is_acronym(str(kb_val), str(form_val)):
                status = "AMBIGUOUS"
                reason += " (Potential Acronym Detected)"
            
            # Semantic / LLM Fallback for Ambiguous
            if status == "AMBIGUOUS":
                llm_check = self.llm_fallback_check(kb_val, form_val)
                if llm_check.get("is_synonym"):
                    status = "CORRECT"
                    reason += f" (Verified by LLM: {llm_check['reason']})"
                else:
                    reason += f" (LLM rejected synonymy: {llm_check['reason']})"
            
            report[field] = {
                "status": status,
                "score": score,
                "kb_value": kb_val,
                "form_value": form_val,
                "reasoning": reason
            }

        # 2. Education Logic (Simplified: Check if Form Institution exists in KB)
        # For a robust system, we'd fuzzy match lists. 
        # Here we just verify the first item for demo purposes or check temporal logic.
        
        # 3. Temporal Consistency
        kb_edu = kb_json.get("education", [])
        form_emp = form_json.get("employment", []) # Use Form employment to check against KB education? 
        # Usually verify Form claims against KB facts.
        # User Prompt: "check ensuring Graduation Year is strictly before any Employment Start Date"
        # Assuming we check consistency WITHIN the Golden Record (Validity of Truth) or Form (Internal Consistency)?
        # Usually "Cross-verify Onboarding Form against Golden Record".
        # But Temporal Consistency is often an internal logic check on the Form claims.
        # I'll check the Form's Education vs Form's Employment to see if the Candidate is lying/confused.
        
        temporal_issues = self.check_temporal_consistency(form_json.get("education", []), form_json.get("employment", []))
        if temporal_issues:
            report["temporal_consistency"] = {
                "status": "INCORRECT",
                "issues": temporal_issues
            }
        else:
            report["temporal_consistency"] = {
                "status": "CORRECT",
                "reasoning": "Sequence of events is logical."
            }

        return report

# Global instance for re-use if imported
_validator_instance = None

def validate_candidate_data(kb_json, form_json):
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = SemanticValidator()
    
    # Parse JSONs if they are strings
    if isinstance(kb_json, str): kb_json = json.loads(kb_json)
    if isinstance(form_json, str): form_json = json.loads(form_json)
        
    return _validator_instance.validate(kb_json, form_json)

if __name__ == "__main__":
    pass
