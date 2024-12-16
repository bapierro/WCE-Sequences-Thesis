import sys
import string
from collections import Counter
import PyPDF2

def extract_text_from_pdf(pdf_path):
    text_content = []
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text_content.append(page.extract_text())
    return "\n".join(text_content)

def clean_and_tokenize(text):
    text = text.lower()
    for p in string.punctuation:
        text = text.replace(p, " ")
    words = text.split()
    return words

def main():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("Enter the path to the PDF file: ").strip()

    text = extract_text_from_pdf(pdf_path)
    words = clean_and_tokenize(text)
    counter = Counter(words)
    top_200 = counter.most_common(200)

    for word, freq in top_200:
        print(f"{word}: {freq}")

if __name__ == "__main__":
    main()

