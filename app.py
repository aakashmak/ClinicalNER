"""
ClinicalNER — Streamlit UI
Biomedical Named Entity Recognition powered by NVIDIA NeMo
"""

import streamlit as st
import pandas as pd
import json
import re
from src.ner_engine import ClinicalNEREngine, SAMPLE_TEXTS, ENTITY_COLORS

# ── Page config ──
st.set_page_config(
    page_title="ClinicalNER | NVIDIA NeMo",
    page_icon="🧬",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .title-block {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        border: 1px solid #76b900;
    }
    .title-block h1 { color: #76b900; font-size: 2.2rem; margin: 0; }
    .title-block p  { color: #aaaaaa; margin: 0.3rem 0 0 0; font-size: 1rem; }
    .entity-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        margin: 3px; font-size: 0.8rem; font-weight: 600; color: #1a1a1a;
    }
    .metric-card {
        background: #1e1e2e; border: 1px solid #333; border-radius: 8px;
        padding: 1rem; text-align: center;
    }
    .metric-card h3 { color: #76b900; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #aaa; margin: 0; font-size: 0.85rem; }
    .nemo-badge {
        background: #76b900; color: #000; padding: 2px 8px;
        border-radius: 4px; font-size: 0.75rem; font-weight: bold;
    }
    .highlighted-text {
        background: #1e1e2e; padding: 1.2rem; border-radius: 8px;
        line-height: 2.2rem; font-size: 1rem; color: #e0e0e0;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="title-block">
    <h1>🧬 ClinicalNER</h1>
    <p>Biomedical Named Entity Recognition &nbsp;|&nbsp;
       <span class="nemo-badge">NVIDIA NeMo</span> &nbsp;
       Powered by BioBERT Token Classification
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    use_nemo = st.toggle("Use NVIDIA NeMo Model", value=True,
                         help="Toggle NeMo BioBERT model. Fallback uses rule-based tagger.")

    st.markdown("---")
    st.markdown("### 📋 Entity Types")
    for entity, color in ENTITY_COLORS.items():
        if entity != "O":
            st.markdown(
                f'<span class="entity-badge" style="background-color:{color}">{entity}</span>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("### 📂 Sample Abstracts")
    selected_sample = st.selectbox("Load a sample", ["— Select —"] + list(SAMPLE_TEXTS.keys()))

    st.markdown("---")
    st.markdown("### 🔗 About")
    st.markdown("""
    **ClinicalNER** extracts biomedical entities from clinical trial text using
    [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)'s pretrained token classification model.

    **Entities detected:**
    - Drugs & Vaccines
    - Diseases & Conditions
    - Dosages
    - Adverse Events
    - Trial Phases
    - Biomarkers
    - Patient Populations
    - Procedures

    Built by [Aakash Mathivanan](https://github.com/aakashmak)
    """)

# ── Load model ──
@st.cache_resource
def load_engine(use_nemo_flag):
    return ClinicalNEREngine(use_nemo=use_nemo_flag)

with st.spinner("Loading NeMo NER model..."):
    engine = load_engine(use_nemo)

model_status = "🟢 NVIDIA NeMo BioBERT" if (use_nemo and engine.use_nemo) else "🟡 Rule-Based Fallback"
st.caption(f"Model: {model_status}")

# ── Input area ──
st.markdown("### 📝 Input Clinical Text")

default_text = SAMPLE_TEXTS[selected_sample] if selected_sample != "— Select —" else ""

col1, col2 = st.columns([3, 1])
with col1:
    input_text = st.text_area(
        "Paste clinical trial abstract, EHR notes, or CDISC documentation:",
        value=default_text,
        height=180,
        placeholder="Enter clinical trial abstract here..."
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🔍 Extract Entities", use_container_width=True, type="primary")
    clear_btn = st.button("🗑️ Clear", use_container_width=True)
    if clear_btn:
        st.rerun()

# ── Run NER ──
if run_btn and input_text.strip():
    with st.spinner("Running NeMo NER inference..."):
        entities = engine.predict(input_text)
        df = engine.to_dataframe(entities)
        summary = engine.get_entity_summary(entities)

    # ── Metrics ──
    st.markdown("### 📊 Extraction Summary")
    cols = st.columns(len(summary) if summary else 1)
    for i, (etype, words) in enumerate(summary.items()):
        color = ENTITY_COLORS.get(etype, "#ccc")
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color:{color}">{len(words)}</h3>
                <p>{etype}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Highlighted text ──
    st.markdown("### 🎨 Annotated Text")
    highlighted = input_text
    if entities:
        # Sort by length desc to avoid partial replacements
        sorted_entities = sorted(entities, key=lambda x: len(x["word"]), reverse=True)
        seen = set()
        for ent in sorted_entities:
            word = ent["word"]
            label = ent["label"]
            color = ENTITY_COLORS.get(label, "#ccc")
            if word not in seen and word in highlighted:
                badge = (f'<span class="entity-badge" style="background-color:{color}" '
                         f'title="{label}">{word}</span>')
                highlighted = highlighted.replace(word, badge, 1)
                seen.add(word)

    st.markdown(f'<div class="highlighted-text">{highlighted}</div>', unsafe_allow_html=True)

    # ── Entity table ──
    st.markdown("### 📋 Extracted Entities")
    tab1, tab2 = st.tabs(["Table View", "JSON View"])

    with tab1:
        if not df.empty:
            display_df = df[["Entity", "Type"]].copy()
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No entities detected. Try a different text or toggle the model.")

    with tab2:
        st.json(summary)

    # ── Export ──
    st.markdown("### 💾 Export Results")
    col_a, col_b = st.columns(2)
    with col_a:
        csv = df[["Entity", "Type"]].to_csv(index=False)
        st.download_button("⬇️ Download CSV", csv, "clinical_entities.csv", "text/csv",
                           use_container_width=True)
    with col_b:
        json_str = json.dumps(summary, indent=2)
        st.download_button("⬇️ Download JSON", json_str, "clinical_entities.json", "application/json",
                           use_container_width=True)

elif run_btn and not input_text.strip():
    st.warning("Please enter some clinical text first.")
else:
    st.info("👆 Paste a clinical trial abstract above and click **Extract Entities** to begin.")

# v1.1 - entity badge rendering and CSV/JSON export added

# v1.2 - metrics cards added for entity count per type

# v1.3 - download buttons for CSV and JSON export

# v1.4 - sidebar legend and NeMo model toggle added
