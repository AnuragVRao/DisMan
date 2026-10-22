"""S1 feature sanity checks: projected text/image features, trivial cat-vs-disaster test, raw cosine range. No tuning."""
import json
from pathlib import Path
import numpy as np, torch
from PIL import Image
import crisis_common as cc

cc.seed_all()
D = Path("data/CrisisMMD_v2.0"); S = D / "crisismmd_datasplit_all" / "crisismmd_datasplit_all"
dev = "cuda" if torch.cuda.is_available() else "cpu"
clip, proc = cc.load_clip(dev, "artifacts/clip")
print("text_projection:", tuple(clip.text_projection.weight.shape), "| visual_projection:", tuple(clip.visual_projection.weight.shape),
      "| text hidden", clip.config.text_config.hidden_size, "| vision hidden", clip.config.vision_config.hidden_size, "| projection_dim", clip.config.projection_dim)
test, _ = cc.load_split("damage", "test", D, S, verbose=False)
P = json.load(open("results/s1_prompts.json"))
texts = [p for c in P["E2"].values() for p in c] + ["a photo of cat"]
with torch.no_grad():
    tok = proc.tokenizer(texts, padding=True, return_tensors="pt").to(dev)
    f = clip.get_text_features(**tok); f = f if isinstance(f, torch.Tensor) else f.pooler_output
    manual = clip.text_projection(clip.text_model(**tok).pooler_output)
    print(f"(a) text: max|get_text_features - text_projection(text_model.pooler_output)| = {(f - manual).abs().max().item():.3e} (n={len(texts)} prompts)")
    unproj = clip.text_model(**tok).pooler_output
    print(f"    unprojected pooler is {tuple(unproj.shape)} (hidden {unproj.shape[1]}-d), projected is {tuple(f.shape)}; mean|projected - unprojected| = {(f[:, :512]-unproj[:, :512]).abs().mean().item():.3f} (not equal -> projection applied)")

    # (b) image: recompute a few cached rows via the explicit vision path and compare with the cache
    cache = np.load("artifacts/emb/clip_damage_test.npy"); ids = json.loads(Path("artifacts/emb/clip_damage_test.ids.json").read_text())
    assert ids == list(test.image_id)
    rows = list(range(0, len(test), len(test) // 5))[:5]
    ims = [Image.open(D / test.rel_image[i]).convert("RGB") for i in rows]
    px = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
    man_img = clip.visual_projection(clip.vision_model(pixel_values=px).pooler_output).float().cpu().numpy()
    print(f"(b) image: max|cached - visual_projection(vision_model.pooler_output)| over {len(rows)} cached rows = {np.abs(cache[rows] - man_img).max():.3e}"
          f" (cached magnitude ~{np.abs(cache[rows]).mean():.3f})")

    # (c) trivial zero-shot, 20 test images
    rng = np.random.RandomState(cc.SEED); pick = rng.choice(len(test), 20, replace=False)
    tt = ["a photo of a cat", "a photo of a building or street after a disaster"]
    tk = proc.tokenizer(tt, padding=True, return_tensors="pt").to(dev)
    tf = clip.get_text_features(**tk); tf = tf if isinstance(tf, torch.Tensor) else tf.pooler_output
    tf = cc.l2(tf.float().cpu().numpy())
    sims = cc.l2(cache[pick]) @ tf.T
    choice = sims.argmax(1)
    print(f"(c) cat vs disaster on 20 random damage-test images: picked 'disaster' for {(choice == 1).sum()}/20 (seed 42)")

    # (d) raw cosine range: image vs the 9 E2 descriptive prompts (all 529 test images), best-matching prompt per image
    e2 = cc.l2(f[:9].float().cpu().numpy()); allsim = cc.l2(cache) @ e2.T
    best = allsim.max(1)
    q = lambda a: ", ".join(f"p{p}={np.percentile(a, p):.3f}" for p in (5, 25, 50, 75, 95))
    print(f"(d) raw cosine, image vs E2 prompts (all pairs, n={allsim.size}): min {allsim.min():.3f} max {allsim.max():.3f}; {q(allsim)}")
    print(f"    best-matching prompt per image (n={len(best)}): {q(best)}")
