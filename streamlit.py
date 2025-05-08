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

from distill.lstm import BiLSTM
from distill.lstm_mha import BiLSTMSeq2SeqMHA_Residual
from distill.distill import beam_search_decode as beam_search_decode_lstm
from distill.distill_mha import (
    beam_search_decode as beam_search_decode_lstm_mha,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
#                               MBART MODELS
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
#                               Pure LSTM
# -----------------------------------------------------------------------------
lstm_tokenizer = AutoTokenizer.from_pretrained("models/sota")
vocab_size_lstm = len(lstm_tokenizer)

lstm_model = BiLSTM(
    vocab_size=vocab_size_lstm,
    embed_dim=512,
    enc_hidden_dim=256,
    dec_hidden_dim=512,
    pad_idx=lstm_tokenizer.pad_token_id,
    num_layers=2,
    dropout=0.3,
).to(device)

ckpt_lstm = torch.load("models/lstm/student_model.pt", map_location=device)
lstm_model.load_state_dict(ckpt_lstm)
lstm_model.eval()

# -----------------------------------------------------------------------------
#                               LSTM + MHA
# -----------------------------------------------------------------------------
vocab_size_mha = len(lstm_tokenizer)

lstm_mha_model = BiLSTMSeq2SeqMHA_Residual(
    vocab_size=vocab_size_mha,
    embed_dim=512,
    enc_hidden_dim=256,
    dec_hidden_dim=512,
    pad_idx=lstm_tokenizer.pad_token_id,
    num_layers=2,
    dropout=0.25,
    num_heads=8,
).to(device)

ckpt_mha = torch.load("models/lstm_mha/student_model.pt", map_location=device)
lstm_mha_model.load_state_dict(ckpt_mha)
lstm_mha_model.eval()


# -----------------------------------------------------------------------------
#                                   FUNCS
# -----------------------------------------------------------------------------


def convert_docx_to_txt(docx_path):
    from docx import Document

    doc = Document(docx_path)
    full_text = [para.text for para in doc.paragraphs]
    return "\n".join(full_text)


def convert_doc_to_txt(doc_path):
    try:
        result = subprocess.run(
            ["antiword", ">", doc_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout
        else:
            print("DOC processing error:", result.stderr)
            return ""
    except Exception as e:
        print("DOC processing error (antiword):", e)
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


# -----------------------------------------------------------------------------
#                               LSTM / LSTM+MHA FUNCS
# -----------------------------------------------------------------------------
def summarize_long_text_lstm(
    text,
    model,
    tokenizer,
    max_input_length=1024,
    beam_size=8,
    max_length=120,
    min_length=10,
    use_mha=False,
):
    full_ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)

    if len(full_ids) <= max_input_length:
        inputs = tokenizer(
            text, max_length=max_input_length, truncation=True, return_tensors="pt"
        )
        src_ids = inputs["input_ids"].to(device)
        if use_mha:
            src_mask = (src_ids != tokenizer.pad_token_id).long().to(device)
            summary_text = beam_search_decode_lstm_mha(
                model=model,
                src_ids=src_ids,
                src_mask=src_mask,
                tokenizer=tokenizer,
                beam_size=beam_size,
                max_length=max_length,
                min_len=min_length,
                device=device,
            )
        else:
            summary_text = beam_search_decode_lstm(
                model=model,
                src_ids=src_ids,
                tokenizer=tokenizer,
                beam_size=beam_size,
                max_length=max_length,
                min_len=min_length,
                device=device,
            )
        return summary_text
    else:
        # Много кусочков
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
            inputs = tokenizer(
                chunk, max_length=max_input_length, truncation=True, return_tensors="pt"
            )
            src_ids = inputs["input_ids"].to(device)

            if use_mha:
                src_mask = (src_ids != tokenizer.pad_token_id).long().to(device)
                summary_chunk = beam_search_decode_lstm_mha(
                    model=model,
                    src_ids=src_ids,
                    src_mask=src_mask,
                    tokenizer=tokenizer,
                    beam_size=beam_size,
                    max_length=max_length,
                    min_len=min_length,
                    device=device,
                )
            else:
                summary_chunk = beam_search_decode_lstm(
                    model=model,
                    src_ids=src_ids,
                    tokenizer=tokenizer,
                    beam_size=beam_size,
                    max_length=max_length,
                    min_len=min_length,
                    device=device,
                )

            chunk_summaries.append(summary_chunk)

            step_count += 1
            progress_bar.progress(step_count / total_steps)
            time.sleep(0.05)

        combined_summary = " ".join(chunk_summaries)

        inputs_final = tokenizer(
            combined_summary,
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt",
        )
        src_ids_final = inputs_final["input_ids"].to(device)

        if use_mha:
            src_mask_final = (src_ids_final != tokenizer.pad_token_id).long().to(device)
            final_summary = beam_search_decode_lstm_mha(
                model=model,
                src_ids=src_ids_final,
                src_mask=src_mask_final,
                tokenizer=tokenizer,
                beam_size=beam_size,
                max_length=max_length,
                min_len=min_length,
                device=device,
            )
        else:
            final_summary = beam_search_decode_lstm(
                model=model,
                src_ids=src_ids_final,
                tokenizer=tokenizer,
                beam_size=beam_size,
                max_length=max_length,
                min_len=min_length,
                device=device,
            )

        step_count += 1
        progress_bar.progress(step_count / total_steps)
        time.sleep(0.05)
        progress_container.empty()

        return final_summary


def run_lstm_summarization_with_timing(
    text,
    model,
    tokenizer,
    model_debug_name,
    use_mha=False,
    beam_size=8,
    max_length=120,
    min_length=10,
):
    start_time = time.perf_counter()
    summary_text = summarize_long_text_lstm(
        text=text,
        model=model,
        tokenizer=tokenizer,
        max_input_length=1024,
        beam_size=beam_size,
        max_length=max_length,
        min_length=min_length,
        use_mha=use_mha,
    )
    end_time = time.perf_counter()

    st.info(f"Summarize by [{model_debug_name}]: {end_time - start_time:.2f} s")
    return summary_text


example_text = """Приложение 1

УТВЕРЖДЕНО
приказом НИУ ВШЭ  Пермь
от 19.10.2023 8.2.6.2-10191023-2

ПОЛОЖЕНИЕ
о профильной олимпиаде НИУ ВШЭ  Пермь для школьников 10-11 классов

Общие положения
Положение о профильной олимпиаде НИУ ВШЭ  Пермь для школьников 10-11 классов (далее Положение) разработано на основании Федерального закона от 29.12.2012  273-ФЗ Об образовании в Российской Федерации (далее Закон  273-ФЗ) и определяет порядок проведения профильной олимпиады НИУ ВШЭ  Пермь для школьников 10-11 классов (далее Олимпиада), ее организационное и методическое обеспечение, порядок отбора победителей и призеров.
Основными целями Олимпиады являются: выявление и развитие интеллектуальных и творческих способностей и интереса к научной деятельности, творческой деятельности у талантливой молодежи; популяризация (пропаганда) научных знаний; создание условий для интеллектуального развития и поддержки одаренных школьников; оказание содействия молодежи в профессиональной ориентации и выборе образовательных траекторий.
Олимпиада проводится Пермским филиалом федерального государственного автономного образовательного учреждения высшего образования Национальный исследовательский университет Высшая школа экономики (далее НИУ ВШЭ  Пермь) в рамках организации и проведения профильного обучения и профессиональной ориентации обучающихся общеобразовательных организаций проекта Открытый университет, реализуемого совместно с Министерством образовании и науки Пермского края в соответствии с постановлением Правительства Пермского края от 31.03.2021  195-п.
Олимпиада проводится по следующим предметам: математика, обществознание, английский язык, информатика.
Последовательность этапов проведения Олимпиады, условия и порядок участия школьников в олимпиадных состязаниях регулируются Регламентом Олимпиады (далее Регламент).
Соответствие предметов профилям Олимпиады ежегодно устанавливается решением Организационного комитета Олимпиады, которое оформляется протоколом и размещается на странице Олимпиады в сети Интернет.
Для обеспечения единого информационного пространства для участников 
и организаторов Олимпиады создана страница Олимпиады на корпоративном сайте (портале) НИУ ВШЭ  Пермь в информационно-телекоммуникационной сети Интернет (далее сеть Интернет) по адресу:
:..2023.

Порядок организации и проведения Олимпиады
Для организационно-методического обеспечения Олимпиады создаются Организационный комитет (далее Оргкомитет) и методическая комиссия. Председателем Оргкомитета является заместитель директора НИУ ВШЭ  Пермь. Состав Оргкомитета утверждается приказом НИУ ВШЭ  Пермь. 
Оргкомитет и методическая комиссия Олимпиады формируются из профессорско-преподавательского состава и иных категорий работников
НИУ ВШЭ  Пермь. Состав методической комиссии утверждается ежегодно председателем Оргкомитета Олимпиады, протокол с решением председателя Оргкомитета размещается на странице НИУ ВШЭ  Пермь корпоративного сайта (портала) НИУ ВШЭ.
Оргкомитет Олимпиады:
устанавливает сроки проведения этапов Олимпиады;
разрабатывает расписание проведения олимпиадных состязаний;
обеспечивает непосредственное проведение Олимпиады;
аннулирует результаты участников в случае нарушения ими правил участия в Олимпиаде, установленных Регламентом;
формирует рейтинговые таблицы участников этапов Олимпиады в порядке, установленном Регламентом, и публикует на странице Олимпиады на странице НИУ ВШЭ  Пермь корпоративного сайта (портала) НИУ ВШЭ в сети Интернет;
совместно с методической комиссией Олимпиады определяет и утверждает списки победителей и призеров Олимпиады;
заблаговременно информирует совершеннолетних лиц, заявивших о своем участии в Олимпиаде, родителей (законных представителей) несовершеннолетних лиц, заявивших о своем участии в Олимпиаде;
обеспечивает сбор и хранение согласий совершеннолетних лиц, заявивших о своем участии в Олимпиаде, несовершеннолетних в возрасте от 14 до 18 лет, заявивших о своем участии в Конференции, также родителей (законных представителей) несовершеннолетних лиц, заявивших о своем участии в Олимпиаде на сбор, хранение, использование, распространение (предоставление) персональных данных;
осуществляет иные функции в соответствии с Положением и Регламентом.
"""

# -----------------------------------------------------------------------------
#                               STREAMLIT
# -----------------------------------------------------------------------------

st.set_page_config(page_title="HSE Legal Documents Summarization")
st.title("HSE Legal Documents Summarization")
st.write(
    "Загрузите файл или введите текст для получения краткого содержания от нескольких моделей:"
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
    input_text = st.text_area("Или введите текст здесь:", example_text)

if st.button("Суммаризовать"):
    if input_text:
        # --- LSTM ---
        summary_lstm = run_lstm_summarization_with_timing(
            text=input_text,
            model=lstm_model,
            tokenizer=lstm_tokenizer,
            model_debug_name="LSTM",
            use_mha=False,
            beam_size=8,
            max_length=120,
            min_length=60,
        )
        st.subheader("Суммаризация от LSTM модели:")
        st.write(summary_lstm)

        # --- LSTM + MHA ---
        summary_lstm_mha = run_lstm_summarization_with_timing(
            text=input_text,
            model=lstm_mha_model,
            tokenizer=lstm_tokenizer,
            model_debug_name="LSTM+MHA",
            use_mha=True,
            beam_size=8,
            max_length=120,
            min_length=60,
        )
        st.subheader("Суммаризация от LSTM+MHA модели:")
        st.write(summary_lstm_mha)

        # --- SOTA ---
        summary_final = run_summarization_with_timing(
            input_text,
            summ_final,
            tokenizer_final,
            final_model,
        )
        st.subheader("Суммаризация от SOTA модели:")
        st.write(summary_final)

        # --- MBART orig ---
        summary_orig = run_summarization_with_timing(
            input_text,
            summ_orig,
            tokenizer_orig,
            orig_model,
        )
        st.subheader("Суммаризация от обычной модели (d0rj/ru-mbart-large-summ):")
        st.write(summary_orig)

    else:
        st.error("Пожалуйста, введите текст.")
