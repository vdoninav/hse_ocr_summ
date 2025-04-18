import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from bert_score import score as bert_score_metric


# ====================== Beam Search ======================


@torch.no_grad()
def beam_search_decode(
    model,
    src_ids,
    src_mask,
    tokenizer,
    beam_size=8,
    max_length=128,
    sos_token_id=None,
    eos_token_id=None,
    verbose=False,
    min_len=10,
    device="cpu",
):
    model.eval()
    if sos_token_id is None:
        sos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    encoder_outputs, (h_enc, c_enc) = model.forward_encoder(src_ids)
    dec_h, dec_c = model.init_decoder_state(h_enc, c_enc)

    if src_ids.size(0) != 1:
        raise ValueError("only batch_size=1")

    beams = [(0.0, [sos_token_id], (dec_h, dec_c))]

    rng = tqdm(range(max_length)) if verbose else range(max_length)
    for _ in rng:
        # print(f"\nStep {_}, current beams:")
        # for lp, tkns, _ in beams:
        #     print(f"  log_prob={lp:.3f}, tokens={tkns}")
        new_beams = []
        for log_prob, tokens, (h, c) in beams:
            if (tokens[-1] == eos_token_id) and (len(tokens) >= min_len):
                new_beams.append((log_prob, tokens, (h, c)))
                continue

            cur_input = torch.tensor([tokens[-1]], device=device)
            logits, (new_h, new_c), _ = model.decoder.forward_step(
                cur_input, h, c, encoder_outputs, src_mask
            )
            # logits: (1, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)  # (vocab_size,)

            topk_vals, topk_ids = torch.topk(log_probs, beam_size)
            for i in range(beam_size):
                token_id = topk_ids[i].item()
                prob_val = topk_vals[i].item()
                new_lp = log_prob + prob_val
                new_toks = tokens + [token_id]
                new_beams.append((new_lp, new_toks, (new_h, new_c)))

        new_beams = sorted(new_beams, key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]

        all_done = all(b[1][-1] == eos_token_id for b in beams)
        if all_done:
            break

    best_tokens = beams[0][1]
    if best_tokens[0] == sos_token_id:
        best_tokens = best_tokens[1:]
    if len(best_tokens) > 0 and best_tokens[-1] == eos_token_id:
        best_tokens = best_tokens[:-1]

    return tokenizer.decode(best_tokens, skip_special_tokens=True)


# ====================== Дистилляция ======================
def soft_cross_entropy(student_logits, teacher_logits, temperature=1.0):
    teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
    log_student_probs = torch.nn.functional.log_softmax(
        student_logits / temperature, dim=-1
    )
    ce = -(teacher_probs * log_student_probs).sum(dim=-1).mean()
    return ce


# ====================== DATA / COLLATE ======================
def preprocess_function(example, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN):
    encoded_src = tokenizer(
        example["text"],
        max_length=MAX_SOURCE_LEN,
        truncation=True,
        padding="max_length",
    )
    with tokenizer.as_target_tokenizer():
        encoded_tgt = tokenizer(
            example["summary"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding="max_length",
        )
    labels = [
        (tid if tid != tokenizer.pad_token_id else -100)
        for tid in encoded_tgt["input_ids"]
    ]
    return {
        "input_ids": encoded_src["input_ids"],
        "attention_mask": encoded_src["attention_mask"],
        "labels": labels,
    }


def collate_fn(batch, pad_token_id):
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    # src_mask = attention_mask (1/0)
    src_mask = attention_mask.clone()

    return {"input_ids": input_ids, "src_mask": src_mask, "labels": labels}


def shift_tokens_right(labels, pad_token_id, eos_token_id):
    dec_in = labels.clone()
    dec_in[dec_in == -100] = pad_token_id
    dec_in = torch.cat(
        [
            torch.full(
                (dec_in.size(0), 1),
                eos_token_id,
                device=dec_in.device,
                dtype=torch.long,
            ),
            dec_in[:, :-1],
        ],
        dim=1,
    )
    return dec_in


# ====================== Оценка (Validation) ======================
def evaluate_student(
    model, val_dataloader, tokenizer, device, beam_size, beam_max_length
):
    ce_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total_ce_loss = 0.0
    total_tokens = 0
    all_preds = []
    all_refs = []

    model.eval()

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="BERT eval", leave=False):
            input_ids = batch["input_ids"].to(device)
            src_mask = batch["src_mask"].to(device)
            labels = batch["labels"].to(device)

            # === CE loss ===
            dec_in = shift_tokens_right(
                labels, tokenizer.pad_token_id, tokenizer.eos_token_id
            )
            logits = model(
                input_ids, dec_in, src_mask=src_mask
            )  # (bs, tgt_len, vocab_size)

            vocab_size = logits.size(-1)
            logits_2d = logits.view(-1, vocab_size)
            labels_1d = labels.view(-1)

            ce_loss = ce_fn(logits_2d, labels_1d)
            valid_tokens = (labels_1d != -100).sum().item()
            total_ce_loss += ce_loss.item()
            total_tokens += valid_tokens

            # === BERT Score: beam search ===
            bs = input_ids.size(0)
            for i in range(bs):
                src_ = input_ids[i].unsqueeze(0)
                mask_ = src_mask[i].unsqueeze(0)
                pred_text = beam_search_decode(
                    model=model,
                    src_ids=src_,
                    src_mask=mask_,
                    tokenizer=tokenizer,
                    beam_size=beam_size,
                    max_length=beam_max_length,
                    device=device,
                )
                # Gold
                ref_ids = labels[i].clone()
                ref_ids[ref_ids < 0] = tokenizer.pad_token_id
                ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)

                all_preds.append(pred_text)
                all_refs.append(ref_text)

    avg_ce = total_ce_loss / total_tokens if total_tokens > 0 else 0.0

    # BERT Score
    P, R, F1 = bert_score_metric(
        all_preds,
        all_refs,
        lang="ru",
        model_type="google-bert/bert-base-multilingual-cased",
        num_layers=9,
        verbose=False,
    )
    p_mean = float(torch.mean(P))
    r_mean = float(torch.mean(R))
    f1_mean = float(torch.mean(F1))

    model.train()
    return avg_ce, p_mean, r_mean, f1_mean


# ====================== TRAIN DISTILLATION ======================
def train_distillation(
    teacher_model,
    student_model,
    train_dataloader,
    val_dataloader,
    tokenizer,
    num_epochs=3,
    lr=1e-3,
    temperature=1.0,
    device="cpu",
    wandb_run_name="default",
    beam_size=32,
    beam_max_length=128,
):
    """
    Функция дистилляции:
     - teacher_model: обученная модель (MBart)
     - student_model: Bi-LSTM/LSTM+MHA/и т.д.
     - train_dataloader: DataLoader с обучающей выборкой
     - val_dataloader: DataLoader с валидационной выборкой
     - tokenizer: токенайзер (MBart)
     - num_epochs: количество эпох обучения
     - lr: learning rate
     - temperature: "температура" для soft-кросс-энтропии
    """

    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.4)

    teacher_model.eval()
    student_model.train()

    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

        progress_bar = tqdm(
            train_dataloader, desc=f"Training (epoch {epoch+1})", leave=False
        )

        total_loss = 0.0

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            src_mask = batch["src_mask"].to(device)
            labels = batch["labels"].to(device)

            dec_in = shift_tokens_right(
                labels, tokenizer.pad_token_id, tokenizer.eos_token_id
            )
            with torch.no_grad():
                t_out = teacher_model(
                    input_ids=input_ids,
                    attention_mask=src_mask,
                    decoder_input_ids=dec_in,
                    output_hidden_states=False,
                    use_cache=False,
                )
                teacher_logits = t_out.logits

            student_logits = student_model(input_ids, dec_in, src_mask=src_mask)

            loss = soft_cross_entropy(student_logits, teacher_logits, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Distill Loss (Train): {avg_train_loss:.4f}")

        scheduler.step(avg_train_loss)

        # === Val ===
        val_ce, val_p, val_r, val_f1 = evaluate_student(
            student_model, val_dataloader, tokenizer, device, beam_size, beam_max_length
        )
        print(
            f"Validation -- CE: {val_ce:.4f}, BERT-P: {val_p:.4f}, R: {val_r:.4f}, F1: {val_f1:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train/distill_loss": avg_train_loss,
                "val/ce": val_ce,
                "val/bert_p": val_p,
                "val/bert_r": val_r,
                "val/bert_f1": val_f1,
            }
        )

        save_dir = f"students/{wandb_run_name}/e{epoch+1}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            student_model.state_dict(), os.path.join(save_dir, "student_model.pt")
        )

    tokenizer.save_pretrained(f"students/{wandb_run_name}")
