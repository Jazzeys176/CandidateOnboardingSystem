import streamlit as st
import os
import json
import shutil
from dotenv import load_dotenv

# Import Backend Logic
from src.ocr_engine import extract_id_card_data, extract_resume_data
from src.extractor import extract_id_fields, extract_candidate_data
from src.validator import validate_candidate_data
from src.reporter import generate_report, generate_pdf_report

# Load Env
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found! Please set it in .env file.")
    st.stop()

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
os.makedirs(INPUTS_DIR, exist_ok=True)

st.set_page_config(page_title="Candidate Onboarding System", layout="wide", page_icon="📝")

# Sidebar Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("Go to", ["Dashboard", "Onboarding Form", "HR Call Transcripts", "Final Output"])

def save_uploaded_file(uploaded_file, filename):
    try:
        path = os.path.join(INPUTS_DIR, filename)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def count_files(pattern):
    count = 0
    for file in os.listdir(INPUTS_DIR):
        if pattern in file.lower():
            count += 1
    return count

# --- Menu 1: Dashboard ---
if menu == "Dashboard":
    st.title("📊 Application Dashboard")
    st.markdown("Overview of Candidate Data Processing.")
    
    col1, col2, col3 = st.columns(3)
    
    # Simple Metrics based on file existence triggers
    resumes_count = count_files("resume")
    forms_count = count_files("form")
    ids_count = count_files("card")
    
    col1.metric("Total Resumes", f"{resumes_count}")
    col2.metric("Onboarding Forms", f"{forms_count}")
    col3.metric("ID Cards Verified", f"{ids_count}")
    
    st.markdown("---")
    st.info("System is optimized for Lenovo ThinkPad T470 (CPU-Only).")

# --- Menu 2: Onboarding Form ---
elif menu == "Onboarding Form":
    st.title("📝 Candidate Onboarding Form")
    
    with st.form("onboarding_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name *", placeholder="First Last")
            email = st.text_input("Email ID *", placeholder="name@example.com")
            college = st.text_input("College Name *")
            id_type = st.selectbox("ID Type", ["Aadhar", "PAN"])
            dob = st.date_input("Date of Birth *")
        
        with col2:
            phone = st.text_input("Phone Number *")
            degree = st.text_input("Degree (with Branch) *", placeholder="B.Tech CS")
            grad_year = st.number_input("Graduation Year *", min_value=1900, max_value=2100, step=1, value=2024)
        
        st.markdown("### Document Uploads")
        resume_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])
        id_file = st.file_uploader("Upload Verification Card (Image)", type=["jpg", "jpeg", "png"])
        
        submitted = st.form_submit_button("Submit Data")
        
        if submitted:
            if not (name and email and college and phone and degree):
                st.error("Please fill all mandatory fields.")
            else:
                # Save Form Data
                form_data = {
                    "personal_details": {
                        "name": name,
                        "email": email,
                        "phone": phone,
                        "dob": str(dob),
                        "id_type_claimed": id_type
                    },
                    "education": [{
                        "institution": college,
                        "degree": degree,
                        "graduation_year": int(grad_year)
                    }],
                    "employment": [] # Placeholder for manual entry scaling
                }
                
                with open(os.path.join(INPUTS_DIR, "onboarding_form.json"), "w") as f:
                    json.dump(form_data, f, indent=4)
                
                # Save Files
                if resume_file:
                    save_uploaded_file(resume_file, "Resume.pdf")
                if id_file:
                    save_uploaded_file(id_file, "id_card.jpeg") # Standardize name for simplicity
                    
                st.success("Form Data and Documents Saved Successfully!")
                st.session_state['form_submitted'] = True

# --- Menu 3: HR Call Transcripts ---
elif menu == "HR Call Transcripts":
    st.title("📞 HR Transcript Upload")
    
    resume_path = os.path.join(INPUTS_DIR, "Resume.pdf")
    id_path = os.path.join(INPUTS_DIR, "id_card.jpeg")
    
    if os.path.exists(resume_path) and os.path.exists(id_path):
        st.success("✅ Resume and ID Card found. You can proceed.")
        
        transcript_file = st.file_uploader("Upload HR Call Transcript (TXT)", type=["txt"])
        if transcript_file:
            path = save_uploaded_file(transcript_file, "transcript.txt")
            if path:
                st.success("Transcript Saved Successfully.")
                # Show preview
                st.text_area("Transcript Preview", transcript_file.getvalue().decode("utf-8"), height=150)
                
    else:
        st.warning("Please upload Resume and ID Card in the 'Onboarding Form' menu first.")

# --- Menu 4: Final Output ---
elif menu == "Final Output":
    st.title("✅ Validation Processing")
    
    if st.button("Process Everything"):
        results_container = st.container()
        
        with results_container:
            status = st.status("Processing Candidate Data...", expanded=True)
            
            # Paths
            resume_path = os.path.join(INPUTS_DIR, "Resume.pdf")
            id_path = os.path.join(INPUTS_DIR, "id_card.jpeg")
            form_path = os.path.join(INPUTS_DIR, "onboarding_form.json")
            transcript_path = os.path.join(INPUTS_DIR, "transcript.txt")
            
            # 1. OCR Step
            status.write("Running OCR on Documents...")
            resume_text = ""
            if os.path.exists(resume_path):
                resume_text = extract_resume_data(resume_path)
            
            id_data = None
            if os.path.exists(id_path):
                ocr_results = extract_id_card_data(id_path)
                status.write("Applying Regulatory Regex on ID...")
                id_data = extract_id_fields(ocr_results)
            
            # 2. Reading Inputs
            transcript_text = ""
            if os.path.exists(transcript_path):
                with open(transcript_path, "r") as f:
                    transcript_text = f.read()
            
            form_data = None
            if os.path.exists(form_path):
                with open(form_path, "r") as f:
                    form_data = json.load(f)

            # 3. KB Extraction
            status.write("Synthesizing Multimodal Knowledge Base...")
            golden_record_str = extract_candidate_data(resume_text, transcript_text, id_data, form_data)
            try:
                golden_record = json.loads(golden_record_str)
            except json.JSONDecodeError:
                status.update(label="Error Parsing JSON", state="error")
                st.error("Failed to generate valid JSON from LLM.")
                st.stop()
            
            # 4. Validation
            status.write("Calculating Semantic Similarity (ONNX)...")
            validation_report = validate_candidate_data(golden_record, form_data if form_data else {})
            
            # 5. Reporting
            # 5. Reporting
            status.write("Generating Recommendation Report...")
            # Legacy string report for Display
            hr_report_summary = generate_report(json.dumps(validation_report))
            
            # PDF Report for Download
            pdf_path = os.path.join(INPUTS_DIR, "validation_report.pdf")
            pdf_status = generate_pdf_report(validation_report, pdf_path)
            
            status.update(label="Processing Complete!", state="complete", expanded=False)
            
            # Display Results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Golden Record (Extracted Truth)")
                st.json(golden_record)
            
            with col2:
                st.subheader("Validation Report (Semantic Check)")
                st.json(validation_report)
            
            st.markdown("---")
            st.subheader("📝 Compliance Recommendation")
            st.info(hr_report_summary)
            
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📄 Download Official PDF Report",
                        data=pdf_file,
                        file_name="validation_report.pdf",
                        mime="application/pdf"
                    )
            else:
                st.error(f"Failed to generate PDF: {pdf_status}")
