import os
import sys
from dotenv import load_dotenv
from src.extractor import extract_candidate_data

def main():
    load_dotenv()
    
    # Ensure API Key is set
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY is not set in environment or .env file.")
        sys.exit(1)
        
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resume_path = os.path.join(base_dir, "inputs", "resume.txt")
    transcript_path = os.path.join(base_dir, "inputs", "transcript.txt")

    try:
        with open(resume_path, "r") as f:
            resume_text = f.read()
            
        with open(transcript_path, "r") as f:
            transcript_text = f.read()
            
        print("Extracting data...")
        json_output = extract_candidate_data(resume_text, transcript_text)
        
        if json_output:
            print("\n--- Extracted KB JSON ---\n")
            print(json_output)
            
            # Phase 2: Validation
            from src.validator import validate_candidate_data
            
            onboarding_form_path = os.path.join(base_dir, "inputs", "onboarding_form.json")
            with open(onboarding_form_path, "r") as f:
                form_json = f.read()

            print("\nValidating data...")
            validation_report = validate_candidate_data(json_output, form_json)
            # print("\n--- Validation Report ---\n")
            # print(validation_report)
            
            # Phase 3: Reporting
            from src.reporter import generate_report
            print("\nGenerating HR Report...")
            hr_report = generate_report(validation_report)
            
            print("\n==================================================")
            print("AUTOMATED CANDIDATE ONBOARDING VALIDATION REPORT")
            print("==================================================\n")
            print(hr_report)
            
            # Save report
            output_path = os.path.join(base_dir, "validation_report.md")
            with open(output_path, "w") as f:
                f.write(hr_report)
            print(f"\nReport saved to: {output_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
