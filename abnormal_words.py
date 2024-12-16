import sys
import string
from collections import Counter
import PyPDF2
from wordfreq import word_frequency

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
    words = [w for w in words if w.isalpha()]
    return words

def main():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("Enter the path to the PDF file: ").strip()

    text = extract_text_from_pdf(pdf_path)
    words = clean_and_tokenize(text)

    if not words:
        print("No words extracted from the document.")
        return

    counter = Counter(words)
    total_words = len(words)

    # Increase the threshold and add a minimum count requirement
    ABNORMAL_THRESHOLD = 500.0  # Increased from 50 to 500
    MIN_COUNT = 5  # Only consider words that appear at least 5 times

    abnormal_words = []
    for word, count in counter.items():
        if count < MIN_COUNT:
            continue  # Skip words that don't meet the minimum frequency requirement

        observed_freq = count / total_words
        baseline_freq = word_frequency(word, 'en')
        if baseline_freq > 0:
            ratio = observed_freq / baseline_freq
            if ratio > ABNORMAL_THRESHOLD:
                abnormal_words.append((word, count, ratio, observed_freq, baseline_freq))

    abnormal_words.sort(key=lambda x: x[2], reverse=True)

    if abnormal_words:
        print("Words used abnormally often compared to a normal text corpus:")
        for word, count, ratio, observed_freq, baseline_freq in abnormal_words:
            print(f"{word}: Count={count}, ObservedFreq={observed_freq:.6f}, "
                  f"BaselineFreq={baseline_freq:.6f}, Ratio={ratio:.2f}")
    else:
        print("No words found that meet the abnormal frequency criteria.")

if __name__ == "__main__":
    main()
