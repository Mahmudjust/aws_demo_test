import json
import os
import boto3
import tempfile
import google.generativeai as genai
from PyPDF2 import PdfReader

# Your 3 things – change only these
BUCKET = "pdf-demo-rag"          # ← change
PDF_NAME = "document.pdf"             # ← change
GEMINI_KEY = "" # ← change

genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def extract_text_from_pdf_s3():
    s3 = boto3.client('s3')
    tmp = tempfile.NamedTemporaryFile(delete=False)
    s3.download_file(BUCKET, PDF_NAME, tmp.name)
    reader = PdfReader(tmp.name)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text[:15000]  # first ~15k chars is enough for demo

# Load once at cold start
DOC_TEXT = extract_text_from_pdf_s3()

def lambda_handler(event, context):
    try:
        query = json.loads(event['body'])['query']
        prompt = f"Document content:\n{DOC_TEXT}\n\nQuestion: {query}\nAnswer in 2-3 sentences:"
        response = model.generate_content(prompt)
        answer = response.text
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"answer": answer})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
