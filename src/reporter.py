import os
import json
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def evaluate_risk(validation_report):
    """
    Determines Risk Level based on discrepancy counts.
    """
    incorrect_count = 0
    ambiguous_count = 0
    
    # Validation report is a dict of fields -> {status, ...}
    for field, data in validation_report.items():
        if field == "temporal_consistency":
            if data.get("status") == "INCORRECT":
                incorrect_count += 1
            continue
            
        status = data.get("status")
        if status == "INCORRECT":
            incorrect_count += 1
        elif status == "AMBIGUOUS":
            ambiguous_count += 1
            
    if incorrect_count > 2:
        return "HIGH"
    elif incorrect_count >= 1 or ambiguous_count > 3:
        return "MEDIUM"
    else:
        return "LOW"

def generate_executive_summary(validation_report, risk_level):
    """
    Uses Groq to generate a professional HR summary.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY not found."
        
    client = Groq(api_key=api_key)
    
    # Filter only relevant issues for the prompt
    issues = {k: v for k, v in validation_report.items() if v.get("status") in ["INCORRECT", "AMBIGUOUS"]}
    
    prompt = f"""
    Act as an HR Compliance Auditor.
    Risk Level: {risk_level}
    
    Validation Issues Detected:
    {json.dumps(issues, indent=2)}
    
    Task: Write a concise (max 100 words) executive summary advising the HR manager on whether to proceed with this candidate. 
    Focus on data integrity. If Risk is HIGH, advise caution. If LOW, recommend proceeding.
    Tone: Professional, Objective.
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {e}"

def generate_pdf_report(validation_report, output_path):
    """
    Generates a PDF report using ReportLab.
    """
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # 1. Title
    story.append(Paragraph("Automated Candidate Onboarding Validation Report", styles['Title']))
    story.append(Spacer(1, 12))
    
    # 2. Risk Level
    risk = evaluate_risk(validation_report)
    risk_color = "green" if risk == "LOW" else "orange" if risk == "MEDIUM" else "red"
    risk_style = ParagraphStyle('Risk', parent=styles['Heading2'], textColor=risk_color)
    story.append(Paragraph(f"RISK ASSESSMENT: {risk}", risk_style))
    story.append(Spacer(1, 12))
    
    # 3. Executive Summary
    summary = generate_executive_summary(validation_report, risk)
    story.append(Paragraph("Executive Summary", styles['Heading3']))
    story.append(Paragraph(summary, styles['BodyText']))
    story.append(Spacer(1, 24))
    
    # 4. Discrepancy Table
    story.append(Paragraph("Discrepancy Details (Incorrect / Ambiguous Fields)", styles['Heading3']))
    
    table_data = [["Field", "Candidate Claim", "Verified Truth", "Status", "Reasoning"]]
    
    has_issues = False
    for field, data in validation_report.items():
        status = data.get("status")
        if status in ["INCORRECT", "AMBIGUOUS"]:
            has_issues = True
            
            # Formatting for table (handle long text)
            form_val = str(data.get("form_value", "N/A"))
            kb_val = str(data.get("kb_value", "N/A"))
            reason = str(data.get("reasoning", ""))
            
            # Wrap text manually or let Paragraph handle it if we used Flowables in cells (advanced)
            # For simplicity, just truncate if super long or rely on basic wrapping
            table_data.append([
                Paragraph(field, styles['BodyText']),
                Paragraph(form_val, styles['BodyText']),
                Paragraph(kb_val, styles['BodyText']),
                Paragraph(status, styles['BodyText']),
                Paragraph(reason, styles['BodyText'])
            ])
            
    if has_issues:
        t = Table(table_data, colWidths=[70, 100, 100, 70, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No discrepancies found. All data verified successfully.", styles['BodyText']))
        
    # Build
    try:
        doc.build(story)
        return f"Report generated successfully at {output_path}"
    except Exception as e:
        return f"Error generating PDF: {e}"

# Backwards compatibility for main.py (Legacy)
def generate_report(validation_json_str):
    try:
        report_data = json.loads(validation_json_str) 
        # Just return the summary logic for legacy string output
        risk = evaluate_risk(report_data)
        return generate_executive_summary(report_data, risk)
    except:
        return "Error parsing validation JSON."
