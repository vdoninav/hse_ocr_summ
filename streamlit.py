import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    MBartForConditionalGeneration,
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    pipeline,
)
import pytesseract
import subprocess
import os
import time
from pdf2image import convert_from_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


orig_model = "models/ru-mbart-large-summ"
tokenizer_orig = AutoTokenizer.from_pretrained(orig_model)
model_orig = MBartForConditionalGeneration.from_pretrained(orig_model).to(device)

summ_orig = pipeline(
    "summarization",
    device=0 if torch.cuda.is_available() else -1,
    model=model_orig,
    tokenizer=tokenizer_orig,
    do_sample=False,
    max_length=120,
    min_length=60,
)

final_model = "models/sota"
tokenizer_final = AutoTokenizer.from_pretrained(final_model)
model_final = MBartForConditionalGeneration.from_pretrained(final_model).to(device)
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
    start_time = time.perf_counter()

    pages = convert_from_path(pdf_path, dpi=300)
    extracted_text = ""

    progress_container = st.empty()
    progress_bar = progress_container.progress(0)
    total_pages = len(pages)

    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page, lang="rus")
        extracted_text += text
        progress_bar.progress((i + 1) / total_pages)

    progress_container.empty()
    st.info(f"OCR Time: {time.perf_counter() - start_time:.2f} s")

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
        overlap = 128
        chunk_size = max_input_length

        progress_container = st.empty()
        progress_bar = progress_container.progress(0)

        chunks = []
        start = 0
        step = chunk_size - overlap
        while start < len(full_ids):
            end = start + chunk_size
            chunk_ids = full_ids[start:end]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
            start += step

        total_steps = len(chunks) * 2 + 1
        step_count = 0

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
            step_count += 1
            progress_bar.progress(step_count / total_steps)
            time.sleep(0.05)

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
        step_count += 1
        progress_bar.progress(step_count / total_steps)
        time.sleep(0.05)

        progress_container.empty()

        return final_summary


def run_summarization_with_timing(text, summ_pipeline, tokenizer, model_debug_name):
    start_time = time.perf_counter()
    summary = summarize_long_text(text, summ_pipeline, tokenizer)
    end_time = time.perf_counter()

    st.info(f"Summarize by [{model_debug_name}]: {end_time - start_time:.2f} s")
    return summary


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
        summary_orig = run_summarization_with_timing(
            input_text,
            summ_orig,
            tokenizer_orig,
            orig_model,
        )
        summary_final = run_summarization_with_timing(
            input_text,
            summ_final,
            tokenizer_final,
            final_model,
        )

        st.subheader("Суммаризация от исходной модели:")
        st.write(summary_orig)

        st.subheader("Суммаризация от finetuned модели:")
        st.write(summary_final)
    else:
        st.error("Пожалуйста, введите текст.")
