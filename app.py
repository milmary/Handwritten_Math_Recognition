#!/usr/bin/env python3
"""
Gradio-based UI for OCR ➜ LaTeX (Image → LaTeX)
"""

import sys, os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
from PIL import ImageOps
import re
import gradio as gr
import tempfile

# --- Tokenizer ---
class SimpleTokenizer:
    """Light-weight tokenizer serialized in checkpoint."""
    def __init__(self, token2id: dict[str, int], blank_idx: int = 0):
        self.token2id = token2id
        self.id2token = {i: t for t, i in token2id.items()}
        self.blank_idx = blank_idx

    def encode(self, text: str) -> list[int]:
        return [self.token2id[t] for t in text.strip().split() if t in self.token2id]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.id2token[i] for i in ids if i in self.id2token)

    def __len__(self) -> int:
        return len(self.token2id) + 1  # blank

# --- Model ---
class CRNN(nn.Module):
    def __init__(self, img_h: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        self.map_to_seq = nn.Linear(512, hidden)
        self.birnn = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden * 2, n_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.map_to_seq(x)
        x, _ = self.birnn(x)
        x = self.classifier(x)
        return x.permute(1, 0, 2)

# --- Attention tokenizer (light wrapper) ---
class AttnTokenizerLight:
    def __init__(self, token2id: dict[str, int], pad_id: int, sos_id: int, eos_id: int):
        self.token2id = token2id
        self.id2token = {i: t for t, i in token2id.items()}
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

    def decode(self, ids: list[int]) -> str:
        specials = {self.pad_id, self.sos_id, self.eos_id}
        return " ".join(self.id2token[i] for i in ids if i in self.id2token and i not in specials)

    def __len__(self) -> int:
        return len(self.token2id)

# --- Attention model (encoder-decoder) ---
class CRNNEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, hidden: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        self.map2seq = nn.Linear(512, hidden)
        self.birnn = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.map2seq(x)
        out, _ = self.birnn(x)
        return out

class AttnDecoder(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, hid_enc: int = 512, hid_dec: int = 256, emb: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.attn = nn.Linear(hid_enc + hid_dec, hid_dec)
        self.v = nn.Linear(hid_dec, 1, bias=False)
        self.rnn = nn.LSTMCell(emb + hid_enc, hid_dec)
        self.fc = nn.Linear(hid_dec, vocab_size)

    def forward_step(self, prev_y: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor], enc_out: torch.Tensor, mask: torch.Tensor):
        emb = self.embedding(prev_y)
        h, c = hidden
        T = enc_out.size(1)
        h_rep = h.unsqueeze(1).repeat(1, T, 1)
        e = self.v(torch.tanh(self.attn(torch.cat([h_rep, enc_out], dim=2)))).squeeze(2)
        e = e.masked_fill(mask == 0, -1e9)
        alpha = e.softmax(1)
        ctx = (alpha.unsqueeze(2) * enc_out).sum(1)
        h, c = self.rnn(torch.cat([emb, ctx], dim=1), (h, c))
        logits = self.fc(h)
        return logits, (h, c), alpha

class CRNN_Attn(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, hidden: int = 256):
        super().__init__()
        self.encoder = CRNNEncoder(hidden=hidden)
        self.decoder = AttnDecoder(vocab_size=vocab_size, pad_id=pad_id, hid_enc=hidden * 2, hid_dec=hidden)

def _sanitize_latex(s: str) -> str:
    """Replace unicode symbols that SymPy's parser doesn't understand."""
    replacements = {
        "−": "-",  # minus sign (U+2212)
        "–": "-",  # en dash
        "—": "-",  # em dash
        "×": "*",  # multiplication sign
        "·": "*",  # middle dot
        "÷": "/",  # division sign
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)

    # Handle bare logarithm base notation like `log_2 x` → `\log_{2}{x}`
    def _log_repl(match: re.Match) -> str:
        base = match.group(1)
        arg = match.group(2).strip()
        # Ensure the argument is wrapped in braces if it isn't already
        if not arg.startswith("{"):
            arg = "{" + arg + "}"
        return f"\\log_{{{base}}}{arg}"

    s = re.sub(r"\\?log_(\d+)\s*([a-zA-Z\\]+)", _log_repl, s)
    return s

# --- Decoding ---
def greedy_decode(logits: torch.Tensor, tokenizer: SimpleTokenizer) -> list[str]:
    with torch.no_grad():
        preds = logits.softmax(2).argmax(2).cpu().numpy()
    decoded = []
    for b in range(preds.shape[1]):
        seq, prev = [], -1
        for t in preds[:, b]:
            if t != prev and t != tokenizer.blank_idx:
                seq.append(t)
            prev = t
        decoded.append(tokenizer.decode(seq))
    return decoded

def attn_greedy_decode(enc_out: torch.Tensor, decoder: AttnDecoder, tokenizer: AttnTokenizerLight, max_len: int = 256) -> list[str]:
    device = enc_out.device
    batch_size = enc_out.size(0)
    mask = torch.ones(enc_out.size()[:2], dtype=torch.bool, device=device)
    h = torch.zeros(batch_size, decoder.rnn.hidden_size, device=device)
    c = torch.zeros_like(h)
    prev_y = torch.full((batch_size,), tokenizer.sos_id, dtype=torch.long, device=device)
    out_ids: list[list[int]] = [[] for _ in range(batch_size)]

    for _ in range(max_len):
        logits, (h, c), _ = decoder.forward_step(prev_y, (h, c), enc_out, mask)
        next_token = logits.argmax(dim=1)
        for i in range(batch_size):
            out_ids[i].append(int(next_token[i].item()))
        if torch.all(next_token == tokenizer.eos_id):
            break
        prev_y = next_token

    return [tokenizer.decode(seq) for seq in out_ids]

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = _get_device()

def _detect_model_type(model_state: dict) -> str:
    """Return 'attn' if attention architecture, else 'ctc'."""
    first_key = next(iter(model_state.keys()))
    return "attn" if first_key.startswith("encoder.") else "ctc"

def _build_tokenizer(tk: dict, model_type: str):
    if model_type == "ctc":
        return SimpleTokenizer(token2id=tk["token2id"], blank_idx=tk.get("blank_idx", 0))
    # attention
    token2id = tk.get("token2id", {})
    pad_id = tk.get("pad_id")
    sos_id = tk.get("sos_id")
    eos_id = tk.get("eos_id")
    # try to derive from special token strings
    if pad_id is None or sos_id is None or eos_id is None:
        pad_tok = tk.get("PAD", "<pad>")
        sos_tok = tk.get("SOS", "<sos>")
        eos_tok = tk.get("EOS", "<eos>")
        pad_id = token2id.get(pad_tok, pad_id if pad_id is not None else 0)
        sos_id = token2id.get(sos_tok, sos_id if sos_id is not None else 1)
        eos_id = token2id.get(eos_tok, eos_id if eos_id is not None else 2)
    return AttnTokenizerLight(token2id=token2id, pad_id=pad_id, sos_id=sos_id, eos_id=eos_id)

_MODEL_CACHE: dict[Path, tuple[str, torch.nn.Module, object]] = {}

def load_model_from_ckpt(ckpt_path: Path) -> tuple[str, torch.nn.Module, object]:
    ckpt_path = Path(ckpt_path)
    if ckpt_path in _MODEL_CACHE:
        return _MODEL_CACHE[ckpt_path]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_type = _detect_model_type(ckpt["model_state"]) if "model_state" in ckpt else "ctc"
    tokenizer_dict = ckpt.get("tokenizer")

    # If tokenizer missing, try falling back to sibling V1/V2 file tokenizers
    if tokenizer_dict is None:
        sibling = ckpt_path.parent / ("epoch_20_V1.pth" if ckpt_path.name.endswith("V2.pth") else "epoch_20_V2.pth")
        if sibling.exists():
            sibling_ckpt = torch.load(sibling, map_location="cpu")
            tokenizer_dict = sibling_ckpt.get("tokenizer")
    if tokenizer_dict is None:
        raise RuntimeError(f"No tokenizer found in checkpoint {ckpt_path} and no fallback available.")

    tokenizer = _build_tokenizer(tokenizer_dict, model_type)

    if model_type == "ctc":
        model = CRNN(img_h=64, n_classes=len(tokenizer), hidden=256)
    else:
        # attn
        model = CRNN_Attn(vocab_size=len(tokenizer), pad_id=tokenizer.pad_id, hidden=256)

    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval().to(device)
    _MODEL_CACHE[ckpt_path] = (model_type, model, tokenizer)
    return _MODEL_CACHE[ckpt_path]

TRANSFORM = T.Compose([
    T.Resize(64),
    T.ToTensor(),
    T.Normalize(0.5, 0.5),
])

def ocr_image_ctc(img: torch.Tensor, model: CRNN, tokenizer: SimpleTokenizer) -> str:
    with torch.no_grad():
        logits = model(img)
    return greedy_decode(logits, tokenizer)[0]

def ocr_image_attn(img: torch.Tensor, model: CRNN_Attn, tokenizer: AttnTokenizerLight) -> str:
    with torch.no_grad():
        enc = model.encoder(img)
        outs = attn_greedy_decode(enc, model.decoder, tokenizer)
    return outs[0]

def ocr_image(image: Image.Image, ckpt_path: Path) -> str:
    # Apply EXIF orientation if present so mobile photos/camera captures
    # are correctly oriented before transforming.
    image = ImageOps.exif_transpose(image)

    model_type, model, tokenizer = load_model_from_ckpt(ckpt_path)
    img = TRANSFORM(image.convert("L")).unsqueeze(0).to(device)
    if model_type == "ctc":
        return _sanitize_latex(ocr_image_ctc(img, model, tokenizer))
    else:
        return _sanitize_latex(ocr_image_attn(img, model, tokenizer))

def _latex_preview_md(latex_text: str) -> str:
    """Return Markdown string that renders LaTeX with MathJax in Gradio."""
    if not latex_text:
        return ""
    safe = _sanitize_latex(latex_text)
    return f"$$\n{safe}\n$$"


def _choose_solve_symbol(symbols: set) -> "object | None":
    """Choose a primary symbol to solve for. Prefer x, y, z, t, else lexicographic."""
    if not symbols:
        return None
    preferred = ["x", "y", "z", "t"]
    name_to_sym = {str(s): s for s in symbols}
    for name in preferred:
        if name in name_to_sym:
            return name_to_sym[name]
    # fallback: first by alphabetical order of name
    return name_to_sym[sorted(name_to_sym.keys())[0]]


def _solve_latex(latex_text: str) -> str:
    """Parse LaTeX into a SymPy expression/equation and try to solve it.

    Returns a Markdown string with the solution (or an informative message).
    """
    latex_text = (latex_text or "").strip()
    if not latex_text:
        return ""

    try:
        from sympy.parsing.latex import parse_latex  # lazy import
        import sympy as sp
    except Exception:
        # SymPy not available or parser missing
        return (
            "SymPy not available to solve equations. Ensure dependencies are installed."
        )

    sanitized = _sanitize_latex(latex_text)
    try:
        expr = parse_latex(sanitized)
    except Exception:
        return (
            "Couldn't parse the LaTeX into an equation. Edit the LaTeX above and it will update automatically."
        )

    # Convert to an equality f(x) = 0 when an explicit equality wasn't provided
    if bool(getattr(expr, "is_Relational", False)):
        eq = expr
        free_syms = eq.free_symbols
    else:
        free_syms = expr.free_symbols
        eq = sp.Eq(expr, 0)

    if not free_syms:
        # No variables: check truth value of the equation
        try:
            is_true = bool(sp.simplify(eq.lhs - eq.rhs) == 0)
        except Exception:
            is_true = False
        return (
            "The expression contains no variables. "
            + ("Identity holds." if is_true else "No variable to solve for.")
        )

    var = _choose_solve_symbol(free_syms)
    if var is None:
        return "No variable detected to solve for. Edit the LaTeX above."

    try:
        # Solve f(x) = 0 form
        f = sp.simplify(eq.lhs - eq.rhs)
        solset = sp.solveset(f, var, domain=sp.S.Complexes)
    except Exception:
        return (
            "Couldn't solve the equation. Edit the LaTeX above to correct the equation."
        )

    # Format output as Markdown using MathJax for nice rendering
    try:
        var_latex = sp.latex(var)
        # Keep the bold label separate from the math expression so MathJax
        # doesn't get confused by Markdown bold formatting.
        header = f"**Solving for** {var_latex}:\n\n"

        if solset == sp.S.EmptySet:
            return header + "_No solutions found._"

        if isinstance(solset, sp.FiniteSet):
            sols = [sp.latex(s) for s in solset]
            # Render each solution as its own display equation
            sol_blocks = "\n\n".join([f"$$ {var_latex} = {s} $$" for s in sols])
            return header + sol_blocks

        # General set (Interval, Union, ConditionSet, Complexes, etc.)
        return header + f"Solution set:\n\n$$ {sp.latex(solset)} $$"
    except Exception:
        return "Solved, but couldn't format the solution for display."


# --- Gradio App ---
with gr.Blocks(title="OCR → LaTeX", theme=gr.themes.Base()) as demo:
    gr.Markdown("""
    ### OCR → LaTeX
    Provide a math image and choose a model. The recognized LaTeX is shown with a live preview.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(
                type="pil",
                label="Input Image (Upload or Camera)",
                image_mode="RGB",
                sources=["upload", "webcam"],
                webcam_options=gr.WebcamOptions(mirror=False),
            )
            model_sel = gr.Dropdown(choices=["V2 (Attention)", "V1 (CTC)"], value="V1 (CTC)", label="Model")
            run_btn = gr.Button("Recognize", variant="primary")
            load_solution_btn = gr.Button("Load solution", variant="secondary")
        with gr.Column(scale=1):
            preview = gr.Markdown()
            solution_md = gr.Markdown()
            latex_out = gr.Textbox(label="Recognized LaTeX", lines=8, placeholder="LaTeX will appear here", show_copy_button=True)

    # Inference wiring
    def _infer_and_preview(img: Image.Image, model_choice: str) -> tuple[str, str, str]:
        if img is None:
            return "", "", ""
        base_dir = Path(__file__).resolve().parent / "models"
        ckpt = base_dir / ("epoch_20_V2.pth" if "V2" in model_choice else "epoch_20_V1.pth")
        latex = ocr_image(img, ckpt)
        md = _latex_preview_md(latex)
        sol = _solve_latex(latex)
        return latex, md, sol

    run_btn.click(_infer_and_preview, inputs=[image_in, model_sel], outputs=[latex_out, preview, solution_md])

    # Live preview when user edits LaTeX
    def _on_edit_text(latex_text: str):
        latex_text = _sanitize_latex(latex_text)
        md = _latex_preview_md(latex_text)
        return latex_text, md

    def _compute_solution_only(latex_text: str) -> str:
        latex_text = _sanitize_latex(latex_text)
        sol = _solve_latex(latex_text)
        return sol

    latex_out.change(_on_edit_text, inputs=[latex_out], outputs=[latex_out, preview])
    load_solution_btn.click(_compute_solution_only, inputs=[latex_out], outputs=[solution_md])

if __name__ == "__main__":
    demo.launch(share=True)


