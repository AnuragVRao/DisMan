"""Day 4 step 4 of the original plan: re-infer dev AND test probabilities for the existing Task 2 checkpoints (seeds 42/43/44), without retraining, and store them through metrics_all.
  t2_L4a              artifacts/ablation/ckpt/t2_L4a_seed{s}.pt         full model: CLIP B/32 cached features + pretrained DistilBERT, cross-attention (the demo's Task 2 model, seed 42)
  t2_distilbert_text  artifacts/ablation/ckpt/t2_text_only_seed{s}.pt   DistilBERT text-only (the 0.294 run)
  t2_img_clip_b32     artifacts/ablation/ckpt/t2_image_only_seed{s}.pt  image-only head on cached CLIP B/32
plus run JSONs from the stored image-only head logits (artifacts/ablation/imbalance/{b32,l14}_inv_seed{s}.npz): t2_img_clip_b32_inv, t2_img_clip_l14_inv.
Each re-inference must reproduce the stored test accuracy and macro-F1 (results/ablation/A and B JSONs); a mismatch is reported and aborts."""
import hashlib
import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import DistilBertConfig, DistilBertTokenizerFast

import d4_prov
import d4_t2_data as d4
import metrics_all as M
from common import ABL_ART, ART, RES, cc
from t2_models import Variant

DEV = torch.device("cuda")
SEEDS = (42, 43, 44)


def feats(D, fdir):
    return {s: torch.tensor(cc.cached(fdir / f"clip_humanitarian_{s}.npy", D["ids"][s], lambda: (_ for _ in ()).throw(RuntimeError("cache miss"))), dtype=torch.float32) for s in D["S"]}


def clip_prov(fdir, encoder):
    h = hashlib.sha256()
    for p in sorted(fdir.glob("clip_humanitarian_*.npy")) + sorted(fdir.glob("clip_humanitarian_*.ids.json")):
        h.update(p.name.encode()); h.update(d4_prov.sha256_file(p).encode())
    return dict(path=str(fdir.relative_to(ART.parent)), sha256=h.hexdigest(), encoder=encoder, frozen=True)


@torch.no_grad()
def infer(model, F, ids, mask):
    model.eval(); out = []
    for i in range(0, len(ids), 128):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(F[i:i + 128].to(DEV), ids[i:i + 128].to(DEV), mask[i:i + 128].to(DEV)).float().cpu())
    return torch.cat(out).numpy()


if __name__ == "__main__":
    D = d4.data(); S, y, labels = D["S"], D["y"], D["labels"]; NC = len(labels)
    tok = DistilBertTokenizerFast.from_pretrained(ART / "tokenizer"); bcfg = DistilBertConfig.from_pretrained(ART / "distilbert_config")   # weights come from the state dict
    T = {s: cc.tokenize(tok, S[s]["text"].tolist(), 128) for s in S}
    F = feats(D, ART / "emb"); fp = clip_prov(ART / "emb", "openai/clip-vit-base-patch32")
    specs = [("t2_L4a", "t2_L4a", "A", lambda: cc.DualStreamClassifier(NC, bert_config=bcfg, img_dim=512), "distilbert-base-uncased", fp),
             ("t2_distilbert_text", "t2_text_only", "B", lambda: Variant("text_only", NC, bert_config=bcfg), "distilbert-base-uncased", None),
             ("t2_img_clip_b32", "t2_image_only", "B", lambda: Variant("image_only", NC, bert_config=bcfg), "none (head trained from scratch on frozen openai/clip-vit-base-patch32 features)", fp)]
    for name, ck, part, mk, init, feat in specs:
        for s in SEEDS:
            m = mk().to(DEV); m.load_state_dict(torch.load(ABL_ART / "ckpt" / f"{ck}_seed{s}.pt", map_location="cpu"))
            zd, zt = (infer(m, F[sp], *T[sp]) for sp in ("dev", "test"))
            ref = json.load(open(RES / part / f"{ck}_seed{s}.json"))
            acc, mf = float((zt.argmax(1) == y["test"]).mean()), float(f1_score(y["test"], zt.argmax(1), labels=list(range(NC)), average="macro", zero_division=0))
            ok = abs(acc - ref["test_acc"]) < 1e-9 and abs(mf - ref["test_macro_f1"]) < 1e-9
            print(f"{name} seed {s}: re-inferred test acc {acc:.4f} macro-F1 {mf:.4f} | stored {ref['test_acc']:.4f} / {ref['test_macro_f1']:.4f} | {'MATCH' if ok else 'MISMATCH'}", flush=True)
            assert ok, f"{name} seed {s} does not reproduce the stored result"
            M.save_run("T2", name, s, labels, y["dev"], zd, y["test"], zt, extra=dict(reinferred_from=f"artifacts/ablation/ckpt/{ck}_seed{s}.pt", stored_result=f"results/ablation/{part}/{ck}_seed{s}.json",
                       **dict(d4_prov.prov("humanitarian", init), feature_cache=feat)))
            del m; torch.cuda.empty_cache()
    # stored logits of the imbalance study's image-only heads (inv arm): dev and test (2237, 8), rows in the T2 split order
    for tag, enc, fdir in (("b32", "openai/clip-vit-base-patch32", ART / "emb"), ("l14", "openai/clip-vit-large-patch14", ABL_ART / "clip_l14")):
        for s in SEEDS:
            z = np.load(ABL_ART / "imbalance" / f"{tag}_inv_seed{s}.npz")
            M.save_run("T2", f"t2_img_clip_{tag}_inv", s, labels, y["dev"], z["dev"], y["test"], z["test"],
                       extra=dict(source=f"artifacts/ablation/imbalance/{tag}_inv_seed{s}.npz", **dict(d4_prov.prov("humanitarian", f"none (head trained from scratch on frozen {enc} features)"), feature_cache=clip_prov(fdir, enc))))
    a = np.mean([(M.load_run("T2", "t2_img_clip_b32", s)["probs_test"].argmax(1) == M.load_run("T2", "t2_img_clip_b32_inv", s)["probs_test"].argmax(1)).mean() for s in SEEDS])
    print(f"image-only B/32: checkpoint vs stored imbalance logits, test argmax agreement {a:.4f}")
