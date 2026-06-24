# Automated Invoice Data Extractor

A Python pipeline that reads any invoice PDF and extracts structured,
machine-readable JSON data automatically — using **pdfplumber** for
text extraction and **OpenAI GPT-4o-mini** for intelligent field parsing.

Built as a submission for **Gen-AI Session 03** (Instructor: Harsh Agrawal,
Vulcan Consulting Group).

---

## What This Project Does

Businesses receive hundreds of invoices every month. Manually reading
each one and entering the data into spreadsheets is slow, error-prone,
and unscalable.

This pipeline automates that process end-to-end: