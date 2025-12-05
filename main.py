"""
Optimized main.py for a lightweight Lambda RAG function.

Environment variables:
- GOOGLE_API_KEY   (optional) : your Gemini / Google GenAI key (if you will call Gemini server-side)
- PDF_URL          (optional) : public HTTP(S) link to the PDF to index (one-time)
- PDF_S3_BUCKET    (optional) : S3 bucket where PDF is stored (if using S3)
- PDF_S3_KEY       (optional) : S3 object key for the PDF
- CACHE_PREFIX     (optional) : prefix name used when caching processed chunks to /tmp (default "raged")
- MAX_CHUNKS       (optional) : number of top chunks to include in prompt (default 3)

Notes:
- /tmp on Lambda is limited (512 MB). This code keeps only text chunks on disk.
- To reduce cold-start latency, pre-generate deployment.zip including only requirements in requirements.txt.
"""

import os
import io
import json
import math
import tempfile
import logging
import requests
from typing import List, Tuple

from pypdf import PdfReader

# If you want to call Gemini directly from this Lambda, you can adapt the call below.
# This file does NOT require any Gemini SDK to run the PDF processing and prompt assembly.
# If you have a specific client library, replace the `call_gemini()` body with library calls.

LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

def download_pdf_from_s3(bucket: str, key: str) -> bytes:
    # boto3 is included in Lambda runtime — import only when used to reduce cold start overhead
    import boto3
    LOG.info("Downloading PDF from S3: s3://%s/%s", "pdf-demo-rag", "Thinking, Fast and Slow -- Kahneman, Daniel.pdf")
    s3 = boto3.client("s3")
    bio = io.BytesIO()
    s3.download_fileobj(bucket, key, bio)
    return bio.getvalue()


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    LOG.info("Extracting text from PDF (size %d bytes)", len(pdf_bytes))
    bio = io.BytesIO(pdf_bytes)
    reader = PdfReader(bio)
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception as e:
            LOG.warning("Error extracting page %d: %s", i, e)
            t = ""
        if t:
            texts.append(t)
    combined = "\n\n".join(texts)
    LOG.info("Extracted text length: %d chars", len(combined))
    return combined


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into chunks of roughly max_chars, with overlap (characters)."""
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + max_chars
        if end >= L:
            chunks.append(text[start:L].strip())
            break
        # avoid splitting inside a word — try to move to nearest newline or space backwards
        slice_ = text[start:end]
        # try split on last newline in slice:
        nl = slice_.rfind("\n")
        sp = slice_.rfind(" ")
        cut = nl if nl > 0 else sp
        if cut > max_chars // 2:
            end = start + cut
        chunks.append(text[start:end].strip())
        start = max(start + max_chars - overlap, end)
    LOG.info("Created %d chunks", len(chunks))
    return chunks


def score_chunks_by_query(chunks: List[str], query: str) -> List[Tuple[int, int]]:
    """Simple score: count number of query tokens found in each chunk. Returns list of (score, index)."""
    q = query.lower()
    q_words = [w for w in q.split() if len(w) > 2]
    result = []
    for idx, c in enumerate(chunks):
        lc = c.lower()
        score = 0
        for w in q_words:
            if w in lc:
                score += 1
        result.append((score, idx))
    return result


def select_top_k_chunks(chunks: List[str], query: str, k: int = 3) -> List[str]:
    if not chunks:
        return []
    scored = score_chunks_by_query(chunks, query)
    # sort by score desc, index asc
    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = []
    for s, idx in scored[:k]:
        if s <= 0:
            # if no chunk matches by keywords, fallback to simple sampling
            break
        selected.append(chunks[idx])
    if not selected:
        # fallback: pick first k chunks (document likely not matching query keywords)
        selected = chunks[:k]
    return selected


def build_prompt(context_chunks: List[str], question: str) -> str:
    """
    Builds a compact prompt for the LLM. Keep prompt small — include only top chunks.
    """
    ctx = "\n\n---\n\n".join([c.strip() for c in context_chunks if c.strip()])
    prompt = (
        "You are an assistant that answers user questions using ONLY the provided CONTEXT. "
        "If the answer is not contained in the context, say you don't know. Be concise.\n\n"
        "CONTEXT:\n"
        f"{ctx}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "Answer:"
    )
    return prompt


def call_gemini_via_http(prompt: str, google_api_key: str, model: str = "gemini-1.5-flash") -> dict:
    """
    Example HTTP POST to Google Generative API.
    NOTE: This is only an example. You should adapt to the exact client you use.
    If you have an official client library installed (google-generativeai), you can replace this
    function with the client's chat/completion call.
    """
    # Minimal example using the 'chat' style endpoint. Adjust payload per your provider's API.
    url = f"https://generativelanguage.googleapis.com/v1beta2/models/{model}:generateMessage"
    headers = {"Authorization": f"Bearer {google_api_key}", "Content-Type": "application/json"}
    body = {
        "message": {
            "content": [{"type": "text", "text": prompt}]
        },
        "temperature": 0.0,
        "maxOutputTokens": 512
    }
    # If your API version / endpoint is different, adapt this call.
    LOG.info("Calling Gemini-like HTTP endpoint (may need adaptation for your setup).")
    r = requests.post(url, headers=headers, json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def lambda_handler(event, context):
    """
    Expected event JSON:
    {
        "query": "What is this document about?"
    }

    Optionally, you can include per-request PDF source:
    {
        "query": "...",
        "pdf_url": "...",
        "pdf_s3_bucket": "...",
        "pdf_s3_key": "..."
    }
    """
    try:
        body = event.get("body")
        if isinstance(body, str):
            try:
                data = json.loads(body)
            except Exception:
                # maybe body is raw text
                data = {"query": body}
        elif isinstance(body, dict):
            data = body
        else:
            data = event

        query = (data.get("query") or data.get("q") or "").strip()
        if not query:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing 'query' field."})}

        # determine PDF source
        pdf_url = data.get("pdf_url") or os.environ.get("PDF_URL")
        s3_bucket = data.get("pdf_s3_bucket") or os.environ.get("PDF_S3_BUCKET")
        s3_key = data.get("pdf_s3_key") or os.environ.get("PDF_S3_KEY")

        cache_prefix = os.environ.get("CACHE_PREFIX", "raged")
        max_chunks = int(os.environ.get("MAX_CHUNKS", "3"))

        # local cache file path (under /tmp)
        safe_name = cache_prefix.replace("/", "_")
        cache_path = f"/tmp/{safe_name}_chunks.json"

        if os.path.exists(cache_path):
            LOG.info("Loading chunks from /tmp cache")
            with open(cache_path, "r", encoding="utf-8") as fh:
                chunks = json.load(fh)
        else:
            # fetch PDF
            if pdf_url:
                pdf_bytes = download_pdf_from_url(pdf_url)
            elif s3_bucket and s3_key:
                pdf_bytes = download_pdf_from_s3(s3_bucket, s3_key)
            else:
                return {"statusCode": 400, "body": json.dumps({"error": "No PDF source found. Set PDF_URL or PDF_S3_BUCKET & PDF_S3_KEY or provide per-request pdf_url/pdf_s3_*."})}

            text = extract_text_from_pdf_bytes(pdf_bytes)
            chunks = chunk_text(text, max_chars=1000, overlap=200)
            # cache chunks to /tmp to speed up subsequent invocations (cold starts still apply)
            try:
                with open(cache_path, "w", encoding="utf-8") as fh:
                    json.dump(chunks, fh)
            except Exception as e:
                LOG.warning("Could not write cache: %s", e)

        top_chunks = select_top_k_chunks(chunks, query, k=max_chunks)
        prompt = build_prompt(top_chunks, query)

        # If you want the function to call Gemini directly, ensure GOOGLE_API_KEY env var is set.
        google_api_key = os.environ.get("GOOGLE_API_KEY") or data.get("google_api_key")
        if google_api_key:
            # Call Gemini-like endpoint (example). Replace with your actual client usage if needed.
            try:
                resp = call_gemini_via_http(prompt, google_api_key)
                # Attempt to extract text from response gracefully:
                answer = None
                # Try a few shapes of server responses:
                if isinstance(resp, dict):
                    # This is provider-specific — adapt to the structure returned by your model endpoint.
                    # Example: look for 'candidates' or 'output' fields.
                    if "candidates" in resp and len(resp["candidates"]) > 0:
                        answer = resp["candidates"][0].get("content", {}).get("text") or str(resp["candidates"][0])
                    elif "output" in resp and isinstance(resp["output"], dict):
                        # some APIs put reply in output[0]
                        output = resp["output"]
                        # scan for text fields
                        for v in output.values():
                            if isinstance(v, str) and v.strip():
                                answer = v
                                break
                    else:
                        # fallback stringify
                        answer = json.dumps(resp)[:2000]
                else:
                    answer = str(resp)
            except Exception as e:
                LOG.exception("Error when calling LLM: %s", e)
                answer = None

            if not answer:
                # return prompt for debugging + a message
                return {"statusCode": 200, "body": json.dumps({
                    "warning": "LLM call returned no parsable text. See 'prompt' and raw_response for debugging.",
                    "prompt": prompt[:4000],
                    "raw_response": resp
                })}
            else:
                return {"statusCode": 200, "body": json.dumps({"answer": answer})}
        else:
            # If no API key present, return the prepared prompt + context so you can call your LLM externally
            return {"statusCode": 200, "body": json.dumps({
                "note": "No GOOGLE_API_KEY found. Returning prepared prompt. Send this prompt to Gemini or your LLM.",
                "prompt": prompt,
                "context_chunks_count": len(top_chunks)
            })}
    except Exception as e:
        LOG.exception("Unhandled error: %s", e)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
