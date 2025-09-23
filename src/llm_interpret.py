import os
import pandas as pd

def _has_openai():
    return any(k in os.environ for k in ["OPENAI_API_KEY","OPENAI_API_BASE"])

def _format_prompt(component: str, df_comp: pd.DataFrame) -> str:
    lines = []
    for _, r in df_comp.head(15).iterrows():
        txt = str(r.get("text_norm") or r.get("text") or "")
        cst = str(r.get("construct") or "")
        score = float(r["score"])
        lines.append(f"- ({score:+.3f}) {cst} :: {txt[:160]}")
    return "Component: " + component + "\n" + "\n".join(lines)

def run_llm_interpretation(
    pca_top_items_csv: str,
    items_file: str,
    out_csv: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2
):
    df = pd.read_csv(pca_top_items_csv)
    comps = sorted(df["component"].unique(), key=lambda x: int(x.replace("PC","")))
    records = []
    if _has_openai():
        import json, requests
        api_base = os.environ.get("OPENAI_API_BASE","https://api.openai.com/v1")
        api_key  = os.environ["OPENAI_API_KEY"]
        headers = {"Authorization": f"Bearer {api_key}","Content-Type":"application/json"}
        for c in comps:
            prompt = (
                "You are an expert in measurement and construct mapping. Given representative items for one PCA component, "
                "assign a short, precise label (2-5 words) and a one-sentence rationale. Return JSON with fields: label, rationale.\n\n"
                + _format_prompt(c, df[df['component']==c])
            )
            payload = {"model": model, "messages":[{"role":"user","content":prompt}],
                       "temperature": temperature, "response_format":{"type":"json_object"}}
            try:
                resp = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                records.append({"component": c, "label": parsed.get("label",""), "rationale": parsed.get("rationale",""), "prompt_hint": ""})
            except Exception:
                records.append({"component": c, "label": "", "rationale": "", "prompt_hint": _format_prompt(c, df[df['component']==c])})
    else:
        for c in comps:
            records.append({"component": c, "label": "", "rationale": "", "prompt_hint": _format_prompt(c, df[df['component']==c])})
    pd.DataFrame(records, columns=["component","label","rationale","prompt_hint"]).to_csv(out_csv, index=False, encoding="utf-8")
