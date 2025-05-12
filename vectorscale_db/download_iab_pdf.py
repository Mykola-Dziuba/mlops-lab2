# download_iab_pdf.py
import os, requests

pdf_url  = "https://www.iab.org.pl/wp-content/uploads/2024/04/Przewodnik-po-sztucznej-inteligencji-2024_IAB-Polska.pdf"
data_dir = "data"
file_pdf = "Przewodnik-po-sztucznej-inteligencji-2024_IAB-Polska.pdf"

os.makedirs(data_dir, exist_ok=True)
path = os.path.join(data_dir, file_pdf)

if not os.path.exists(path):
    print("Downloading…")
    r = requests.get(pdf_url, stream=True, timeout=30)
    with open(path, "wb") as f:
        for chunk in r.iter_content(1024):
            if chunk:
                f.write(chunk)
    print("Saved to", path)
else:
    print("PDF already present:", path)

