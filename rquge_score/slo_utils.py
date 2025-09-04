# ===== Abstractive QA answer extraction (no clamping to context) =====
import re
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

# --- light cleaners ---
_ROLE_PREFIX_RE = re.compile(r"^\s*(model|assistant|asistent|assistantu|odgovor)\s*[:\-]?\s*", re.IGNORECASE)
_END_TOK_RE     = re.compile(r"\[?\s*END\s*\]?", re.IGNORECASE)

def clean_answer_text(s: str) -> str:
    # remove common scaffolding & collapse whitespace (do NOT clamp to context)
    s = s.replace("\u200b", " ").replace("\r", " ")
    if "Odgovor:" in s:
        s = s.split("Odgovor:")[-1]
    s = _END_TOK_RE.sub("", s)
    s = _ROLE_PREFIX_RE.sub("", s)
    s = s.replace("\n", " ")
    s = " ".join(s.split())
    return s.strip()

# --- optional stop on simple phrases / layout boundaries ---
class StopOnPhrases(StoppingCriteria):
    def __init__(self, phrases: List[str], tokenizer):
        super().__init__()
        self.tok = tokenizer
        self.ids = [self.tok.encode(p, add_special_tokens=False) for p in phrases]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq = input_ids[0].tolist()
        for pid in self.ids:
            if len(pid) and len(seq) >= len(pid) and seq[-len(pid):] == pid:
                return True
        return False

def build_prompt_chat(tok, question: str, context: str) -> str:
    user = (
        "Na podlagi podanega vprašanja in besedila generiraj samo en smiseln in pravilen odgovor, "
        "ki izhaja izključno iz konteksta besedila. "
        "Odgovor naj bo kratek (nekaj besed). "
        "Če odgovora ni v besedilu, vrni prazen niz.\n\n"
        f"Vprašanje: {question}\nBesedilo: {context}\nOdgovor:"
    )
    msgs = [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def build_prompt_plain(question: str, context: str) -> str:
    return (
        "Navodila: Odgovori na vprašanje izključno z informacijami iz besedila. "
        "Odgovor naj bo kratek. "
        "Če ni odgovora v besedilu, vrni prazen niz.\n\n"
        f"Vprašanje: {question}\nBesedilo: {context}\nOdgovor:"
    )

@torch.no_grad()
def gams_generate_answer(
    model,
    tokenizer,
    question: str,
    context: str,
    use_chat_template: bool = True,
    max_new_tokens: int = 24,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    trim_to_first_clause: bool = True,
) -> str:
    # 1) prompt
    prompt = (
        build_prompt_chat(tokenizer, question, context)
        if use_chat_template else
        build_prompt_plain(question, context)
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 2) basic stopping (newline / section headings)
    stops = StoppingCriteriaList([
        StopOnPhrases(["\n", "\n\n", "Vprašanje:", "Besedilo:", "Odgovor:"], tokenizer)
    ])

    # ensure pad token for CausalLM
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) generate only the continuation
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=stops,
        return_dict_in_generate=True,
        output_scores=False,
    )

    # 4) decode ONLY newly generated tokens
    gen_ids  = out.sequences[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    ans = clean_answer_text(gen_text)

    # 5) keep it concise (optional)
    if trim_to_first_clause and ans:
        ans = re.split(r"[\.!?;\n]", ans)[0].strip()

    return ans

# -------------------------
# Example wiring (commented)
# -------------------------
# model_path = "cjvt/GaMS-2B-Instruct"    # or your fine-tuned QA
# tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
# if tok.pad_token is None and tok.eos_token is not None:
#     tok.pad_token = tok.eos_token
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", attn_implementation="eager").eval()
# ctx = "Bohinj je znan po Bohinjskem jezeru, največjem stalnem jezeru v Sloveniji."
# q   = "Po katerem jezeru je znan Bohinj?"
# print(generate_answer_abstractive(model, tok, q, ctx, use_chat_template=True))
