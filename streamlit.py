import streamlit as st
import torch
from transformers import AutoTokenizer, MBartForConditionalGeneration, pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_orig = AutoTokenizer.from_pretrained("ru-mbart-large-summ")
model_orig = MBartForConditionalGeneration.from_pretrained("ru-mbart-large-summ").to(
    device
)

summ_orig = pipeline(
    "summarization",
    device=0 if torch.cuda.is_available() else -1,
    model=model_orig,
    tokenizer=tokenizer_orig,
    do_sample=False,
    max_length=120,
    min_length=60,
)

tokenizer_final = AutoTokenizer.from_pretrained("final_model")
model_final = MBartForConditionalGeneration.from_pretrained("final_model").to(device)

summ_final = pipeline(
    "summarization",
    device=0 if torch.cuda.is_available() else -1,
    model=model_final,
    tokenizer=tokenizer_final,
    do_sample=False,
    max_length=120,
    min_length=60,
)


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
st.write("Введите текст ниже для получения краткого содержания от двух моделей:")

input_text = st.text_area("", "Настоящим приказываю....")

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
