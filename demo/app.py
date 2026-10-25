"""Gradio disaster-triage demo. Run: HF_HUB_OFFLINE=1 python demo/app.py"""
import json
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import DATA, Pipeline  # noqa: E402

PIPE = Pipeline()
EX_FILE = Path(__file__).parent / "examples.json"   # written after the examples are chosen
EXAMPLES = json.load(open(EX_FILE)) if EX_FILE.exists() else []


def badge(t1):
    informative = t1["label"] == "informative"
    color, word = ("#1a7f37", "INFORMATIVE") if informative else ("#6e7781", "NOT INFORMATIVE")
    return (f"<span style='background:{color};color:#fff;padding:6px 14px;border-radius:14px;font-weight:600'>"
            f"{word}</span> &nbsp; text-only model, p = {t1['prob']:.2f}")


def disagreement(t1, t2):
    inf, nh = t1["label"] == "informative", t2["label"] == "Not humanitarian"
    if inf == nh:  # informative but "Not humanitarian", or not informative but a humanitarian class
        return ("**Note:** the two models disagree. Both outputs are shown as-is. Task 1 is trained on *tweet-text* labels; "
                "Task 2 is trained on *image* labels, so a tweet can be informative as text while its image is not humanitarian (or vice versa).")
    return ""


def run(image, text):
    if image is None:
        raise gr.Error("Please upload an image.")
    r = PIPE.predict(image, text)
    gallery = [(n["image_path"], f"{n['label']} | sim {n['score']:.2f}\n{n['tweet'][:140]}") for n in r["neighbours"]]
    t3_md = f"**Predicted damage severity: {r['t3']['label']}**"
    if "knn_label" in r["t3"]:      # hybrid mode (default; CAPSTONE_T3_HYBRID=0 restores the 5-NN vote): do the retrieved cases agree with the probe?
        if r["t3"]["neighbours_agree"]:
            t3_md = f"**{r['t3']['label']}.** Similar past cases agree."
        else:
            t3_md = (f"**{r['t3']['label']} (lower confidence).** Similar past cases suggest {r['t3']['knn_label']}. "
                     "Recommend human review.")
        t3_md += f"\n\nConfidence: {r['t3']['probe_confidence']:.2f}"
    return (badge(r["t1"]), disagreement(r["t1"], r["t2"]), r["t2"]["probs"],
            t3_md, r["t3"]["vote_share"], gallery,
            f"{r['latency_s']:.2f} s on {PIPE.device}")


with gr.Blocks(title="Disaster triage") as demo:
    gr.Markdown("# Disaster tweet triage\nUpload an image and paste the tweet text.")
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="Image")
            txt = gr.Textbox(label="Tweet text", lines=3)
            go = gr.Button("Analyse", variant="primary")
            lat = gr.Textbox(label="Latency", interactive=False)
        with gr.Column():
            gr.Markdown("### 1. Informative? (Task 1)")
            b1 = gr.HTML()
            note = gr.Markdown()
            gr.Markdown("### 2. Humanitarian category (Task 2)")
            l2 = gr.Label(num_top_classes=8, label="Class probabilities")
            gr.Markdown("### 3. Damage severity (Task 3)")
            t3 = gr.Markdown()
            l3 = gr.Label(label="agreement among similar past cases (k=5 vote share)")
    gr.Markdown("### 4. Top-5 similar past cases (train-only index)")
    gal = gr.Gallery(columns=5, height=320, object_fit="contain")
    if EXAMPLES:
        gr.Examples(examples=[[str(DATA / e["rel_image"]), e["tweet_text"]] for e in EXAMPLES],
                    inputs=[img, txt], label="Preloaded examples (test split)")
    go.click(run, [img, txt], [b1, note, l2, t3, l3, gal, lat])

if __name__ == "__main__":
    demo.launch(allowed_paths=[str(DATA)])
