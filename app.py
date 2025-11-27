import streamlit as st
from transformers import pipeline
import re # Added for regex-based sentence tokenization
import html # Added for HTML escaping

# --- Helper Functions & Setup ---

def simple_sentence_tokenize(text: str) -> list[str]:
    """
    A simple regex-based sentence tokenizer.
    """
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    final_sentences = [s.strip() for s in sentences if s.strip()]
    if not final_sentences and text.strip():
        final_sentences.append(text.strip())
    return final_sentences

@st.cache_resource(show_spinner=False)
def load_detector():
    """Loads the AI text detection model from Hugging Face."""
    return pipeline("text-classification", model="openai-community/roberta-base-openai-detector")

def analyze_and_highlight(detector, text: str):
    """
    Analyzes text sentence by sentence, returns highlighted HTML and overall AI probability.
    """
    sentences = simple_sentence_tokenize(text)
    if not sentences:
        return "", 0.0

    results = detector(sentences, truncation=True)
    
    sentence_scores = []
    for result in results:
        label = result["label"]
        score = float(result["score"])
        ai_prob = score if label.lower() == "fake" else 1.0 - score
        sentence_scores.append(max(0.0, min(1.0, ai_prob)))

    highlighted_text = ""
    for sent, score in zip(sentences, sentence_scores):
        if score > 0.75:
            color = "rgba(255, 77, 77, 0.6)"  # Strong red
        elif score > 0.5:
            color = "rgba(255, 165, 0, 0.5)"  # Orange
        else:
            color = "transparent"
        
        safe_sent = html.escape(sent)
        highlighted_text += f'<span style="background-color: {color}; padding: 2px 1px; line-height: 1.7;">{safe_sent}</span> '

    overall_prob = sum(sentence_scores) / len(sentence_scores) if sentence_scores else 0.0
    return highlighted_text, overall_prob

# --- Page Config & Style ---
st.set_page_config(
    page_title="AI 文字偵測器",
    page_icon="🤖",
    layout="centered"
)

st.markdown(
    """
    <style>
    /* Force all text within the main Streamlit app container to be black */
    .stApp {
        color: #000000 !important;
    }
    body { color: #000000 !important; } /* Keep as a fallback */
    [data-testid="stAppViewContainer"] { background-color: #fffacd; }
    [data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
    textarea[data-testid="stTextAreaInput"] {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 1.5rem;
        color: #000000 !important;
        border: none;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        resize: vertical;
    }
    .result-card {
        background-color: #f0f8ff;
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        color: #000000 !important;
    }
    .result-card h5, .result-card p, .result-card span { /* Explicitly set color for common tags */
        color: #000000 !important;
    }
    .small-muted { font-size: 0.85rem; color: #000000 !important; text-align: center; }
    .stButton>button { border-radius: 10px; }

    /* --- Custom Loader --- */
    .custom-loader {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        border: 5px solid #f3f3f3;
        border-top: 5px solid #3498db;
        animation: spin 1s linear infinite;
        margin: 20px auto; /* Center the loader */
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* --- Custom Progress Bar --- */
    .custom-progress-bar {
        width: 100%;
        background-color: #e0e0e0;
        border-radius: 10px;
        height: 28px;
        position: relative;
        overflow: hidden; /* Ensures inner bar respects the border radius */
    }
    .custom-progress-bar-inner {
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease-in-out;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 0.9rem;
        line-height: 28px; /* Vertically center text */
        text-shadow: 1px 1px 2px rgba(0,0,0,0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- App Layout ---

st.markdown("<h3 style='text-align: center; color: black;'>🤖 AI 文字偵測器</h3>", unsafe_allow_html=True)
st.markdown(
    "<p class='small-muted'>▲ 僅供教學/示範使用，請勿拿來做學術違規或懲處判斷。</p>",
    unsafe_allow_html=True
)
st.write("")

# --- Text Area ---
text = st.text_area(
    "Text input area",
    height=250,
    placeholder="在此貼上您要分析的文字...",
    label_visibility="collapsed",
)

# --- Analysis Controls ---
col_btn, col_slider, _ = st.columns([1.2, 2, 1])
with col_btn:
    analyze = st.button("🔍 分析文字", use_container_width=True, type="primary")
with col_slider:
    min_len = st.slider("建議最少字元", 50, 600, 200, 50, label_visibility="collapsed")

# --- Analysis & Results ---
if analyze:
    if not text.strip():
        st.warning("請先輸入一段文字。")
    elif len(text) < min_len:
        st.markdown(
            f"<div style='background-color: #fff3cd; color: black; padding: 1rem; border-radius: 0.5rem;'>"
            f"目前字數約 {len(text)}，少於建議長度 {min_len}，偵測結果可能不太可靠。"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        # --- Custom Spinner Logic ---
        spinner_placeholder = st.empty()
        spinner_html = '<div class="custom-loader"></div>'
        spinner_placeholder.markdown(spinner_html, unsafe_allow_html=True)
        
        # Load model (will be fast after first run due to caching)
        detector = load_detector()
        
        # Clear the spinner
        spinner_placeholder.empty()
        
        # --- Analysis and Display ---
        highlighted_html, overall_ai_prob = analyze_and_highlight(detector, text)
        percent = round(overall_ai_prob * 100, 1)

        # Determine overall verdict
        if overall_ai_prob >= 0.7:
            verdict = "😲 **這段文字「很有可能」是 AI 生成的。**"
        elif overall_ai_prob > 0.4:
            verdict = "🤔 **這段文字「可能」帶有 AI 生成的特徵。**"
        else:
            verdict = "😊 **這段文字「比較像」是人類撰寫的。**"

        # --- Create the custom progress bar ---
        if overall_ai_prob >= 0.7:
            bar_color = "linear-gradient(to right, #ffafbd, #ffc3a0)" # Reddish
        elif overall_ai_prob > 0.4:
            bar_color = "linear-gradient(to right, #ffc3a0, #ffdf7e)" # Orangey
        else:
            bar_color = "linear-gradient(to right, #a1c4fd, #c2e9fb)" # Bluish
        
        bar_html = f"""
        <div class="custom-progress-bar">
            <div class="custom-progress-bar-inner" style="width: {percent}%; background: {bar_color};">
                {percent}%
            </div>
        </div>
        """

        # --- Display Results ---
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("##### 📝 整體分析結果")
        st.markdown(verdict)
        st.markdown(f"**AI 生成可能性：**")
        st.markdown(bar_html, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### 🕵️ 可疑段落標示")
        st.markdown(
            f'<div style="font-size: 0.95rem; border: 1px solid #ddd; padding: 1rem; border-radius: 10px; background-color: #fafafa;">{highlighted_html}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='font-size: 0.75rem; color: #000000; text-align: right; margin-top: 5px;'>■ 高度可能AI &nbsp; ■ 可能AI</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.markdown(
    "<p class='small-muted'>模型: `openai-community/roberta-base-openai-detector` on Hugging Face.<br>對新模型 (如 GPT-4o) 和中文內容的準確度有限。</p>",
    unsafe_allow_html=True,
)
