import os
import shutil
import time
import fitz  # PyMuPDF for normal PDF extraction
import pdf2image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def extract_text_from_multiple_pdfs(pdf_paths, lang, output_folder="output", text_folder="text", failed_folder="failed", max_retries=3):
    combined_text = ""

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(text_folder, exist_ok=True)
    os.makedirs(failed_folder, exist_ok=True)

    # Merge PDFs if more than one PDF is provided
    if len(pdf_paths) > 1:
        merged_pdf_path = os.path.join(output_folder, "merged.pdf")
        merge_pdfs(pdf_paths, merged_pdf_path)
        pdf_paths = [merged_pdf_path]  # Replace with merged PDF path

    try:
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"Error: The file '{pdf_path}' does not exist. Skipping...")
                continue

            print(f"Processing PDF: {pdf_path}")

            # Split the PDF into chunks of 40 pages each
            split_pdfs = split_pdf(pdf_path, output_folder, max_pages=40)

            for split_pdf_info in split_pdfs:
                # Convert each split PDF to images
                convert_pdf_to_images(split_pdf_info['pdf_path'], split_pdf_info['start_page'], output_folder)

            # Extract text from each image in the output folder
            combined_text += extract_text_from_images(output_folder, text_folder, failed_folder, max_retries)

        # Combine all extracted text and print it
        combined_text += combine_extracted_text(text_folder)
        print(f"\nFinal Extracted Text:\n{combined_text}")

    finally:
        # Clean up the output, text, and failed folders
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
            print(f"Deleted output folder: {output_folder}")
        if os.path.exists(text_folder):
            shutil.rmtree(text_folder)
            print(f"Deleted text folder: {text_folder}")
        if os.path.exists(failed_folder):
            shutil.rmtree(failed_folder)
            print(f"Deleted failed folder: {failed_folder}")

    return combined_text

def merge_pdfs(pdf_paths, output_path):
    from PyPDF2 import PdfMerger

    merger = PdfMerger()
    try:
        for pdf in pdf_paths:
            merger.append(pdf)
            print(f"Merging {pdf}")
        merger.write(output_path)
        merger.close()
        print(f"Merged PDF saved at {output_path}")
    except Exception as e:
        print(f"Error merging PDFs: {e}")
        merger.close()

def split_pdf(pdf_path, output_folder, max_pages=40):
    pdf_splits = []
    try:
        with fitz.open(pdf_path) as doc:
            total_pages = doc.page_count
            for start in range(0, total_pages, max_pages):
                end = min(start + max_pages, total_pages)
                split_pdf_path = os.path.join(output_folder, f"split_{start + 1}_to_{end}.pdf")
                with fitz.open() as new_doc:
                    for i in range(start, end):
                        new_doc.insert_pdf(doc, from_page=i, to_page=i)
                    new_doc.save(split_pdf_path)
                pdf_splits.append({'pdf_path': split_pdf_path, 'start_page': start + 1, 'end_page': end})
                print(f"Created split PDF: {split_pdf_path} containing pages {start + 1} to {end}")
    except Exception as e:
        print(f"Error splitting PDF: {e}")
    return pdf_splits

def convert_pdf_to_images(pdf_path, start_page, output_folder):
    try:
        images = pdf2image.convert_from_path(pdf_path)
        for idx, image in enumerate(images):
            page_number = start_page + idx
            image_path = os.path.join(output_folder, f"page_{page_number}.png")
            image.save(image_path, "PNG")
            print(f"Converted page {page_number} to image and saved at {image_path}")
    except Exception as e:
        print(f"Error converting PDF to images: {e}")

def extract_text_from_images(output_folder, text_folder, failed_folder, max_retries):
    extracted_text = ""

    # Sort files numerically based on the page number
    sorted_files = sorted(
        os.listdir(output_folder), 
        key=lambda x: int(x.split('_')[1].split('.')[0]) if x.endswith(".png") else float('inf')
    )

    # Process all images in the output folder in proper order
    for file_name in sorted_files:
        if file_name.endswith(".png"):
            image_path = os.path.join(output_folder, file_name)
            page_number = int(file_name.split("_")[1].split(".")[0])
            text_file_path = os.path.join(text_folder, f"page_{page_number}.txt")

            # Process the image and extract text
            success = process_page(image_path, text_file_path, failed_folder, max_retries)
            if success:
                extracted_text += f"\nPage {page_number}:\n"
                with open(text_file_path, "r", encoding="utf-8") as text_file:
                    extracted_text += text_file.read().strip() + "\n"

    return extracted_text

def process_page(image_path, text_file_path, failed_folder, max_retries):
    retries = 0
    while retries < max_retries:
        try:
            text = extract_text_using_google_lens(image_path)
            if text:
                with open(text_file_path, "w", encoding="utf-8") as text_file:
                    text_file.write(text)
                print(f"Text saved to {text_file_path}")
                return True
            else:
                print(f"No text extracted for {image_path}, retrying... ({retries + 1}/{max_retries})")
                retries += 1
        except Exception as e:
            print(f"Error processing {image_path}: {e}, retrying... ({retries + 1}/{max_retries})")
            retries += 1

    # Move to failed folder if extraction fails after retries
    shutil.move(image_path, os.path.join(failed_folder, os.path.basename(image_path)))
    print(f"Failed to extract text from {image_path} after {max_retries} retries. Moved to {failed_folder}")
    return False

def extract_text_using_google_lens(file_path):
    chrome_options = Options()
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36"
    )

    driver = webdriver.Remote(
        command_executor="http://localhost:4444/wd/hub",
        options=chrome_options
    )

    try:
        driver.get("https://lens.google.com/")
        print("Navigating to Google Lens...")

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, '//input[@type="file"]'))
        )
        upload_input = driver.find_element(By.XPATH, '//input[@type="file"]')
        upload_input.send_keys(os.path.abspath(file_path))
        print(f"Uploading file: {os.path.abspath(file_path)}")

        text_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(., "Text")]'))
        )
        text_button.click()
        print("Clicked on 'Text' button.")

        select_all_text_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(., "Select all text")]'))
        )
        select_all_text_button.click()
        print("Clicked on 'Select all text' button.")

        time.sleep(2)

        extracted_text = driver.execute_script("return window.getSelection().toString();")
        return extracted_text if extracted_text.strip() else None

    except Exception as e:
        print(f"Error during extraction: {e}")
        return None

    finally:
        driver.quit()
        print("Browser closed.")

def combine_extracted_text(text_folder):
    combined_text = ""
    for file_name in sorted(os.listdir(text_folder), key=lambda x: int(x.split('_')[1].split('.')[0])):
        if file_name.endswith(".txt"):
            file_path = os.path.join(text_folder, file_name)
            with open(file_path, "r", encoding="utf-8") as text_file:
                text = text_file.read().strip()
                if text:
                    combined_text += f"\nPage {file_name.split('_')[1].split('.')[0]}:\n{text}\n"
                    print(f"Adding text from {file_name} to combined output.")
    return combined_text

# Example usage
#if __name__ == "__main__":
   # pdf_paths = ["/root/classes/testapp/garde 6 science.pdf"]  # Replace with actual paths to your PDFs
   # final_extracted_text = extract_text_from_multiple_pdfs(pdf_paths, lang="eng")
   # print("Final Extracted Text:\n", final_extracted_text)
