"""Day 2, before D3: frozen TOKEN features of the CHOSEN text encoder for the cross-attention head (SELECTION_RULE.md: first 64 tokens, first token = the text query).

  python d2_extract_tokens.py minilm|mpnet
Saved to artifacts/ablation/d2_feats/<name>/tokens_humanitarian_<split>.npz: tok fp16 [N, 64, D] (the transformer's last hidden states, before pooling) and pad bool [N, 64] (True = padding).
The fraction of tweets longer than 64 tokens (truncated) is recorded in tokens_meta.json.
"""
import json
import sys
import time

import numpy as np
import torch

from common import ABL_ART, ART, DATA, SPLIT_DIR, cc

SPECS = {"minilm": ART / "sbert", "mpnet": "sentence-transformers/all-mpnet-base-v2"}
L = 64


def main(name):
    out = ABL_ART / "d2_feats" / name
    st = cc.load_sbert(torch.device("cuda"), SPECS[name]); st.max_seq_length = L
    meta, t0 = {}, time.time()
    for s in ("train", "dev", "test"):
        df = cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0]
        texts = df["text"].tolist(); n = len(texts)
        lens = np.array([len(x) for x in st.tokenizer(texts, truncation=False)["input_ids"]])
        D = st.get_sentence_embedding_dimension()
        tok = np.zeros((n, L, D), dtype="float16"); pad = np.ones((n, L), dtype=bool)
        for i in range(0, n, 1024):
            embs = st.encode(texts[i:i + 1024], batch_size=128, output_value="token_embeddings", convert_to_numpy=False, show_progress_bar=False)
            for j, e in enumerate(embs):
                e = e.detach().cpu().float().numpy()[:L]
                tok[i + j, :len(e)] = e; pad[i + j, :len(e)] = False
        assert np.isfinite(tok.astype("float32")).all() and (~pad[:, 0]).all()
        np.savez(out / f"tokens_humanitarian_{s}.npz", tok=tok, pad=pad)
        meta[s] = dict(n=n, dim=D, truncated_fraction=float((lens > L).mean()), mean_tokens=float(np.minimum(lens, L).mean()))
        print(f"{name} {s}: {tok.shape}, truncated {meta[s]['truncated_fraction']:.4f}, mean tokens {meta[s]['mean_tokens']:.1f}", flush=True)
    json.dump(meta, open(out / "tokens_meta.json", "w"))
    print(f"{name} tokens done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
