import json
import os
import pandas as pd
from fpdf import FPDF

# Paths for the evaluation results
EVAL_RESULTS_PATH = os.path.join("models", "evaluations", "results.json")
REPORTS_PATH = os.path.join("reports", "model_report.pdf")

# Load evaluation results
def load_evaluation_results(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

# Generate a PDF report
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Model Evaluation Report', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_chapter(self, title, body):
        self.add_page()
        self.chapter_title(title)
        self.chapter_body(body)

# Generate report content
def generate_report_content(results):
    lines = []
    lines.append("Model Evaluation Summary:")
    for metric, value in results.items():
        lines.append(f"{metric}: {value}")
    return "\n".join(lines)

# Main function to generate report
def generate_pdf_report(eval_results_path, output_report_path):
    results = load_evaluation_results(eval_results_path)
    
    # Generate report content
    content = generate_report_content(results)
    
    # Create PDF report
    pdf = PDFReport()
    pdf.set_title('Model Evaluation Report')
    pdf.add_chapter("Evaluation Metrics", content)
    
    # Save the report
    pdf.output(output_report_path)

if __name__ == "__main__":
    generate_pdf_report(EVAL_RESULTS_PATH, REPORTS_PATH)