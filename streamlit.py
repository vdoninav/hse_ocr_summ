import streamlit as st
import torch
from transformers import AutoTokenizer, MBartForConditionalGeneration, pipeline
import pytesseract
import subprocess
import os
from pdf2image import convert_from_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_orig = AutoTokenizer.from_pretrained("models/ru-mbart-large-summ")
model_orig = MBartForConditionalGeneration.from_pretrained(
    "models/ru-mbart-large-summ"
).to(device)
summ_orig = pipeline(
    "summarization",
    device=0 if torch.cuda.is_available() else -1,
    model=model_orig,
    tokenizer=tokenizer_orig,
    do_sample=False,
    max_length=120,
    min_length=60,
)

tokenizer_final = AutoTokenizer.from_pretrained("models/final_model")
model_final = MBartForConditionalGeneration.from_pretrained("models/final_model").to(
    device
)
summ_final = pipeline(
    "summarization",
    device=0 if torch.cuda.is_available() else -1,
    model=model_final,
    tokenizer=tokenizer_final,
    do_sample=False,
    max_length=120,
    min_length=60,
)


def convert_docx_to_txt(docx_path):
    from docx import Document

    doc = Document(docx_path)
    full_text = [para.text for para in doc.paragraphs]
    return "\n".join(full_text)


def convert_doc_to_txt(doc_path):
    try:
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", doc_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout
        else:
            print("Ошибка:", result.stderr)
            return ""
    except Exception as e:
        print("Ошибка при запуске textutil/catdoc:", e)
        return ""


def ocr_pdf(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300)
    extracted_text = ""

    for page in pages:
        text = pytesseract.image_to_string(page, lang="rus")
        extracted_text += text

    return extracted_text


def get_text_from_file(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".docx":
        return convert_docx_to_txt(file_path)
    elif ext == ".doc":
        return convert_doc_to_txt(file_path)
    elif ext == ".pdf":
        return ocr_pdf(file_path)
    else:
        print(f"Неподдерживаемый формат файла: {file_path}")
        return ""


def summarize_long_text(text, summ, tokenizer, max_input_length=1024):
    full_ids = tokenizer.encode(text, add_special_tokens=True)
    if len(full_ids) <= max_input_length:
        truncated_text = tokenizer.decode(
            tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_input_length,
            ),
            skip_special_tokens=True,
        )
        return summ(truncated_text)[0]["summary_text"]
    else:
        progress_container = st.empty()
        progress_bar = progress_container.progress(0)

        num_chunks = len(range(0, len(full_ids), max_input_length))
        total_steps = num_chunks * 2 + 1
        step = 0

        chunks = []
        for i in range(0, len(full_ids), max_input_length):
            chunk_ids = full_ids[i : i + max_input_length]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
            step += 1
            progress_bar.progress(step / total_steps)

        chunk_summaries = []
        for chunk in chunks:
            truncated_chunk = tokenizer.decode(
                tokenizer.encode(
                    chunk,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_input_length,
                ),
                skip_special_tokens=True,
            )
            summary = summ(truncated_chunk)[0]["summary_text"]
            chunk_summaries.append(summary)
            step += 1
            progress_bar.progress(step / total_steps)

        combined_summary = " ".join(chunk_summaries)
        truncated_combined = tokenizer.decode(
            tokenizer.encode(
                combined_summary,
                add_special_tokens=True,
                truncation=True,
                max_length=max_input_length,
            ),
            skip_special_tokens=True,
        )
        final_summary = summ(truncated_combined)[0]["summary_text"]
        step += 1
        progress_bar.progress(step / total_steps)

        progress_container.empty()

        return final_summary


st.title("HSE docs summ")
st.write(
    "Загрузите файл или введите текст для получения краткого содержания от двух моделей:"
)

uploaded_file = st.file_uploader(
    "Загрузите файл в одном из форматов: txt, doc, docx, pdf.",
    type=["txt", "doc", "docx", "pdf"],
)

input_text = ""
if uploaded_file is not None:
    filename = uploaded_file.name
    _, extension = os.path.splitext(filename)
    temp_file_path = "temp_uploaded_file" + extension
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    input_text = get_text_from_file(temp_file_path)
else:
    input_text = st.text_area("Или введите текст здесь:", "Настоящим приказываю...")

if st.button("Суммаризовать"):
    if input_text:
        summary_orig = summarize_long_text(input_text, summ_orig, tokenizer_orig)
        summary_final = summarize_long_text(input_text, summ_final, tokenizer_final)

        st.subheader("Суммаризация от исходной модели:")
        st.write(summary_orig)

        st.subheader("Суммаризация от finetuned модели:")
        st.write(summary_final)
    else:
        st.error("Пожалуйста, введите текст.")
