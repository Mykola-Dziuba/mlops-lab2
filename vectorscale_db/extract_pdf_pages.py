# extract_pdf_pages.py
# --- imports -------------------------------------------------------------
import os, json, fitz  # PyMuPDF

data_dir  = "data"
file_pdf  = "Przewodnik-po-sztucznej-inteligencji-2024_IAB-Polska.pdf"
file_json = "Przewodnik-po-sztucznej-inteligencji-2024_IAB-Polska.json"

pdf_path  = os.path.join(data_dir, file_pdf)
json_path = os.path.join(data_dir, file_json)

# --- pagination ----------------------------------------------------------
doc   = fitz.open(pdf_path)
pages = [{"page_num": i, "text": doc.load_page(i).get_text()}
         for i in range(len(doc))]
doc.close()

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(pages, f, indent=4, ensure_ascii=False)

print(f"Saved {len(pages)} pages → {json_path}")

