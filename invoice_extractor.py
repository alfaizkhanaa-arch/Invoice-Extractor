
# SECTION 1 — Import Required Libraries

import os
import json
import pdfplumber
from groq import Groq           
from dotenv import load_dotenv  



# SECTION 2 — Load Environment Variables

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))



# SECTION 3 — Step 1 & 2: Extract Text from PDF

def extract_text_from_pdf(pdf_path: str) -> str:
   

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    raw_pages = []

    with pdfplumber.open(pdf_path) as pdf:
        print(f"  → PDF loaded | Pages: {len(pdf.pages)}")

        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                raw_pages.append(page_text)
                print(f"  → Page {page_number}: {len(page_text)} chars")
            else:
                print(f"  → Page {page_number}: No text (image-based?)")

    return "\n".join(raw_pages)



# SECTION 4 — Step 3: Clean the Text

def clean_extracted_text(raw_text: str) -> str:
   

    lines = raw_text.split("\n")
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    cleaned_text = "\n".join(cleaned_lines)

    print(f"  → Cleaned: {len(raw_text)} → {len(cleaned_text)} chars")

    return cleaned_text


# SECTION 5 — Step 4: Build the Prompt

def build_extraction_prompt() -> str:


    return """
You are an expert invoice data extraction assistant.

Read the invoice text and return ONLY a valid JSON object.
No markdown, no code fences, no explanation — just raw JSON.

Extract these 13 fields exactly as named:

1.  invoice_number  — Unique invoice ID or reference number.
2.  invoice_date    — Date issued. Format: DD-MM-YYYY.
3.  due_date        — Payment deadline. DD-MM-YYYY or exact phrase (e.g. "Upon Receipt").
4.  billed_by       — Company or person issuing the invoice.
5.  billed_to       — Client or recipient being billed.
6.  line_items      — Array of objects with "item" and "amount" keys.
7.  subtotal        — Total before tax/discount (string with currency symbol).
8.  discount        — Discount amount (string with currency symbol).
9.  tax_or_gst      — Tax/GST amount (string). Return "Not Applicable" if explicitly stated.
10. total_amount    — Final payable amount (string with currency symbol).
11. currency        — ISO currency code (INR, USD, EUR).
12. payment_method  — How to pay (e.g. "NEFT / UPI").
13. notes           — Any additional remarks.

RULES:
- Missing fields MUST be null (not "N/A", not "", not "Not found").
- Do NOT invent or guess values not in the text.
- line_items is always an array. If none found → null.
- Return ONLY the JSON object. Nothing else.

{
  "invoice_number": "...",
  "invoice_date": "...",
  "due_date": "...",
  "billed_by": "...",
  "billed_to": "...",
  "line_items": [{"item": "...", "amount": "..."}],
  "subtotal": "...",
  "discount": "...",
  "tax_or_gst": "...",
  "total_amount": "...",
  "currency": "...",
  "payment_method": "...",
  "notes": "..."
}
""".strip()



# SECTION 6 — Step 5: Call the Groq LLM

def call_groq_llm(cleaned_text: str, system_prompt: str) -> str:
   

    print("  → Sending request to Groq Llama 3.1 8B ...")

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # Free and fast
        temperature=0,                   # Deterministic output
        max_tokens=2048,                 # Enough for full JSON
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                    "Extract invoice data from this text. "
                    "Return ONLY the JSON object:\n\n"
                    + cleaned_text
                )
            }
        ]
    )

    raw_response = response.choices[0].message.content
    print(f"  → Response received | {len(raw_response)} chars")

    return raw_response


# ─────────────────────────────────────────────
# SECTION 7 — Step 6: Parse the JSON
# ─────────────────────────────────────────────
def parse_json_response(raw_response: str) -> dict:
    """Parse the LLM response into a Python dict with fallback cleaning."""

    # First attempt: parse directly
    try:
        data = json.loads(raw_response)
        print("  → JSON parsed successfully")
        return data
    except json.JSONDecodeError:
        print("  → Direct parse failed — stripping markdown fences ...")

    # Fallback: remove markdown code fences
    cleaned = raw_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    # Second attempt
    try:
        data = json.loads(cleaned)
        print("  → JSON parsed after cleanup")
        return data
    except json.JSONDecodeError as e:
        print(f"\n  ✗ PARSE FAILED. Raw response:\n{raw_response}")
        raise ValueError(f"Could not parse JSON: {e}")


# ─────────────────────────────────────────────
# SECTION 8 — Step 7: Display Extracted Data
# ─────────────────────────────────────────────
def display_extracted_data(invoice_data: dict, invoice_path: str) -> None:
    """Print extracted fields in a clean format."""

    print("\n" + "═" * 60)
    print(f"  EXTRACTED INVOICE DATA")
    print(f"  Source: {invoice_path}")
    print("═" * 60)

    labels = {
        "invoice_number": "Invoice Number ",
        "invoice_date":   "Invoice Date   ",
        "due_date":       "Due Date       ",
        "billed_by":      "Billed By      ",
        "billed_to":      "Billed To      ",
        "line_items":     "Line Items     ",
        "subtotal":       "Subtotal       ",
        "discount":       "Discount       ",
        "tax_or_gst":     "Tax / GST      ",
        "total_amount":   "Total Amount   ",
        "currency":       "Currency       ",
        "payment_method": "Payment Method ",
        "notes":          "Notes          ",
    }

    for key, label in labels.items():
        value = invoice_data.get(key)

        if key == "line_items" and value and isinstance(value, list):
            print(f"  {label}:")
            for i, item in enumerate(value, 1):
                print(f"    {i}. {item.get('item', 'N/A')} — {item.get('amount', 'N/A')}")
        else:
            display_value = value if value is not None else "null"
            print(f"  {label}: {display_value}")

    print("═" * 60 + "\n")


# ─────────────────────────────────────────────
# SECTION 9 — Save Output to JSON File
# ─────────────────────────────────────────────
def save_output_to_json(invoice_data: dict, output_path: str) -> None:
    """Save extracted data as a formatted JSON file."""

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(invoice_data, f, indent=2, ensure_ascii=False)

    print(f"  → Saved to: {output_path}")


# ─────────────────────────────────────────────
# SECTION 10 — Master Pipeline
# ─────────────────────────────────────────────
def process_invoice(pdf_path: str, output_path: str) -> dict:
    """Run the full 7-step pipeline on one invoice."""

    print("\n" + "─" * 60)
    print(f"  PROCESSING: {pdf_path}")
    print("─" * 60)

    # Steps 1-2: Extract text
    print("\n[Step 1-2] Extracting text ...")
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        raise ValueError(f"No text extracted from {pdf_path}")

    # Step 3: Clean text
    print("\n[Step 3] Cleaning text ...")
    cleaned_text = clean_extracted_text(raw_text)

    # Step 4: Build prompt
    print("\n[Step 4] Building prompt ...")
    system_prompt = build_extraction_prompt()
    print(f"  → Prompt: {len(system_prompt)} chars")

    # Step 5: Call Groq LLM
    print("\n[Step 5] Calling Groq API ...")
    raw_response = call_groq_llm(cleaned_text, system_prompt)

    # Step 6: Parse JSON
    print("\n[Step 6] Parsing JSON ...")
    invoice_data = parse_json_response(raw_response)

    # Step 7: Display
    print("\n[Step 7] Displaying results ...")
    display_extracted_data(invoice_data, pdf_path)

    # Save to file
    save_output_to_json(invoice_data, output_path)

    return invoice_data



# SECTION 11 — Main Entry Point

if __name__ == "__main__":

    invoices_to_process = [
        ("invoices/invoice_1.pdf", "outputs/output_invoice_1.json"),
        ("invoices/invoice_2.pdf", "outputs/output_invoice_2.json"),
        ("invoices/invoice_3.pdf", "outputs/output_invoice_3.json"),
    ]

    results_summary = []

    for invoice_path, output_path in invoices_to_process:
        try:
            data = process_invoice(invoice_path, output_path)
            results_summary.append({
                "invoice": invoice_path,
                "status": "SUCCESS",
                "fields": len(data)
            })
        except FileNotFoundError as e:
            print(f"\n  ✗ FILE NOT FOUND: {e}")
            results_summary.append({
                "invoice": invoice_path, "status": "FAILED", "fields": 0
            })
        except Exception as e:
            print(f"\n  ✗ ERROR: {e}")
            results_summary.append({
                "invoice": invoice_path, "status": f"FAILED: {e}", "fields": 0
            })

    # Final Summary
    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE — SUMMARY")
    print("═" * 60)
    for r in results_summary:
        icon = "✓" if r["status"] == "SUCCESS" else "✗"
        print(f"  {icon} {r['invoice']}")
        print(f"    Status: {r['status']}")
        if r["status"] == "SUCCESS":
            print(f"    Fields: {r['fields']}")
    print("═" * 60 + "\n")
