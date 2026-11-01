"""Variant C: minimal surgical cleaning of raw tweet_text (ftfy, html.unescape, drop truncated tail, collapse whitespace). Keeps #, RT, case, emoji."""
import html, re
import ftfy
import pandas as pd
import d4_t1_data as d4
from common import RES

def clean_c(t):
    t = ftfy.fix_text(t or "")
    for _ in range(3):
        u = html.unescape(t)
        if u == t: break
        t = u
    # drop only a truncated tail: trailing ellipsis, or a t.co fragment that is cut off (shorter than a full 10-char id)
    t = re.sub(r"\s*(?:…|\.\.\.)\s*$", "", t)
    t = re.sub(r"\s*https?:(?://(?:t\.co/\w{0,9})?)?$", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

if __name__ == "__main__":
    D = d4.data(); out = RES / "D5_prep"; out.mkdir(parents=True, exist_ok=True)
    for sp, df in D["S"].items():
        raw = df.tweet_text.fillna(""); c = raw.map(clean_c)
        print(sp, len(df), "changed", round(float((c != raw).mean()), 4), "empty", int((c == "").sum()))
        pd.DataFrame(dict(image_id=df.image_id.values, text=c.values)).to_csv(out / f"task1_text_C_{sp}.tsv", sep="\t", index=False)
        if sp == "train":
            for i in (raw[raw.str.contains("&amp;|‚Ä|â€")].index[:3].tolist() + raw[raw.str.contains("…$")].index[:2].tolist()):
                print(" RAW:", raw[i][:120].encode("ascii", "replace").decode(), "\n  C :", c[i][:120].encode("ascii", "replace").decode())
