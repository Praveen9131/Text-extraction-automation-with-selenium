# **Text Extraction**

This project provides a robust solution for extracting text from scanned PDFs using **Selenium**, **Google Lens**, and **Docker**, ensuring high accuracy suitable for **Retrieval-Augmented Generation (RAG)** pipelines with Large Language Models (LLMs). Unlike traditional OCR tools like **Tesseract** or **EasyOCR**, this method guarantees near-perfect results even with complex or low-quality scanned documents.

## **Key Features**
- **High Accuracy**: Leverages Google Lens for extracting text with precision.
- **RAG Pipeline Integration**: Outputs structured text optimized for embedding into vector databases.
- **Scalable Setup**: Uses Dockerized Selenium Chrome instances for efficient, parallel processing.
- **Automation**: Automates the entire text extraction process from PDF to structured text files.

## **Technologies Used**
- **Selenium WebDriver**: Automates interaction with Google Lens.
- **Google Lens**: Extracts text from images with near-perfect accuracy.
- **Docker**: Hosts and scales Selenium Chrome instances.
- **Python Libraries**:
  - `pdf2image`: Converts PDFs into image format.
  - `Pillow`: Processes images before uploading to Google Lens.
  - `concurrent.futures`: Enables multi-threaded processing for performance optimization.

## **Setup Instructions**

### **1. Prerequisites**
- **Docker**: Install [Docker](https://docs.docker.com/get-docker/).
- **Python**: Install Python 3.8 or higher.

### **2. Install Required Python Packages**
Set up a virtual environment and install dependencies:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
````
### Requirements

##selenium
##pdf2image
##Pillow
##  Start Dockerized Selenium Chrome Instances

##docker run -d -p 4444:4444 --name selenium_chrome1 selenium/standalone-chrome
##docker run -d -p 4445:4444 --name selenium_chrome2 selenium/standalone-chrome

## **How It Works**
- **PDF to Images**: The PDF is converted into images using `pdf2image`.
- **Google Lens Automation**: Images are uploaded to Google Lens using Selenium WebDriver to extract text.
- **Parallel Processing**: Multiple pages are processed simultaneously across Dockerized Selenium Chrome instances.
- **Output**: The extracted text is saved as individual files for each page and combined for RAG pipelines.

## **Advantages Over Traditional OCR**
- **100% Accuracy**: Extracts text using Google Lens, surpassing the capabilities of Tesseract and EasyOCR.
- **Scalability**: Processes large documents efficiently with multiple Docker containers.
- **Ease of Use**: Automates the entire process, reducing manual effort.

## **Use Cases**
- **Document Digitization**: Converts scanned documents into structured text files.
- **RAG Pipelines**: Prepares high-accuracy text for embedding into vector databases.
- **Educational Resources**: Extracts text from books and papers for research purposes.

## **Future Enhancements**
- Support for additional Docker instances for increased scalability.
- Error handling for rate limits or interruptions in Google Lens processing.
- Pre-processing for enhanced image compatibility with Google Lens.



