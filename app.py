import streamlit as st
import pandas as pd
import json
import os
import base64
import datetime
import numpy as np
from groq import Groq

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL   = "llama-3.3-70b-versatile"

GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="Mali AI", page_icon="🌿", layout="wide",
                   initial_sidebar_state="expanded")

# ─────────────────────────────────────────────
# LANGUAGE DATA
# ─────────────────────────────────────────────
LANG_DATA = {
    "English": {
        "title": "Mali AI",
        "subtitle": "AI Farm Supervisor",
        "dashboard": "Dashboard",
        "mali_agent": "Mali Agent",
        "rain": "Rain Reliability", "soil": "Soil Health",
        "hum": "Humidity", "co2": "CO2 Level",
        "alerts": "Active Alerts",
        "alert_msg": "Low Nitrogen detected in Sector 4",
        "upload": "📎 Attach plant photo (optional)",
        "ask": "Ask about weather, soil, crops, fertilizer...",
        "online": "Online · All Agents Active",
        "reset": "🔄 Clear Chat",
        # Router system prompt — tells the LLM to pick an agent
        "router_sys": """You are Mali, an AI farming supervisor that manages 4 specialist agents:
- 🌦️ Weather Agent: handles weather forecasts, rain, climate suitability, temperature, humidity
- 🌱 Soil Agent: handles soil health, pH, nutrients, soil type, moisture
- 👁️ Vision Agent: handles plant disease detection, leaf analysis, pest identification (used when an image is attached)
- 🧪 Fertilizer Agent: handles fertilizer recommendations, NPK ratios, crop nutrition

When the user sends a message (with or without an image):
1. Decide which agent is most relevant based on the question.
2. Start your reply with exactly this tag on its own line: [AGENT: <AgentName>]
   e.g.  [AGENT: 🌦️ Weather Agent]
3. Then give your expert answer in plain conversational text.

If an image is attached, ALWAYS use the Vision Agent regardless of the text.
If the question spans multiple agents, pick the single most relevant one.
Keep answers concise and practical. Respond in English only.""",

        "router_sys_tamil": """நீங்கள் மாலி, ஒரு AI விவசாய மேற்பார்வையாளர். 4 நிபுண முகவர்களை நிர்வகிக்கிறீர்கள்:
- 🌦️ வானிலை முகவர்: வானிலை, மழை, வெப்பநிலை
- 🌱 மண் முகவர்: மண் ஆரோக்கியம், pH, ஊட்டச்சத்துக்கள்
- 👁️ பார்வை முகவர்: தாவர நோய், புகைப்பட பகுப்பாய்வு
- 🧪 உர முகவர்: உர பரிந்துரைகள், NPK

பயனர் செய்தி வரும்போது:
1. மிகவும் பொருத்தமான முகவரை தேர்ந்தெடுக்கவும்.
2. பதிலை இந்த வடிவத்தில் தொடங்கவும்: [AGENT: <முகவர் பெயர்>]
   எ.கா: [AGENT: 🌦️ வானிலை முகவர்]
3. பின்னர் நிபுண ஆலோசனை வழங்கவும்.

படம் இணைக்கப்பட்டால் எப்போதும் பார்வை முகவரை பயன்படுத்தவும்.
பதில்களை தமிழில் மட்டுமே வழங்கவும்.""",
    },
    "Tamil": {
        "title": "மாலி AI",
        "subtitle": "AI விவசாய மேற்பார்வையாளர்",
        "dashboard": "டாஷ்போர்டு",
        "mali_agent": "மாலி முகவர்",
        "rain": "மழை நம்பகத்தன்மை", "soil": "மண் ஆரோக்கியம்",
        "hum": "ஈரப்பதம்", "co2": "கார்பன் அளவு",
        "alerts": "செயலில் உள்ள விழிப்பூட்டல்கள்",
        "alert_msg": "பிரிவு 4-இல் குறைந்த நைட்ரஜன் கண்டறியப்பட்டது",
        "upload": "📎 தாவர படம் இணைக்கவும் (விருப்பம்)",
        "ask": "வானிலை, மண், பயிர், உரம் பற்றி கேளுங்கள்...",
        "online": "இணையத்தில் உள்ளது · அனைத்து முகவர்களும் செயலில்",
        "reset": "🔄 அரட்டையை அழி",
        "router_sys": "",       # filled below
        "router_sys_tamil": "", # filled below
    }
}

# Tamil uses the Tamil router prompt
LANG_DATA["Tamil"]["router_sys"]       = LANG_DATA["English"]["router_sys_tamil"]
LANG_DATA["Tamil"]["router_sys_tamil"] = LANG_DATA["English"]["router_sys_tamil"]

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_data():
    w, s, df = None, None, None
    try:
        if os.path.exists("data/agent_b_to_supervisor (1).json"):
            with open("data/agent_b_to_supervisor (1).json") as f: w = json.load(f)
        if os.path.exists("data/soil_agent_output.json"):
            with open("data/soil_agent_output.json") as f: s = json.load(f)
        if os.path.exists("data/WeatherStation.csv"):
            df = pd.read_csv("data/WeatherStation.csv")
    except: pass
    return w, s, df

w_data, s_data, df_station = load_data()

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'DM Sans', sans-serif !important; box-sizing: border-box; }

.stApp { background: #ECE5DD !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #1B5E20 !important; border-right: none !important; }
[data-testid="stSidebar"] * { color: #fff !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #A5D6A7 !important; font-size:0.7rem !important; text-transform:uppercase; letter-spacing:.05em; }
[data-testid="stSidebar"] [data-baseweb="select"] { background: rgba(255,255,255,0.12) !important; border:none !important; border-radius:8px !important; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color:#fff !important; }
[data-testid="stSidebar"] .stRadio > label { color:#A5D6A7 !important; font-size:0.7rem !important; text-transform:uppercase; letter-spacing:.08em; }
[data-testid="stSidebar"] .stRadio div[role="radio"] { background:rgba(255,255,255,0.07) !important; border-radius:10px !important; padding:10px 14px !important; color:#E8F5E9 !important; font-size:0.9rem !important; font-weight:500; margin-bottom:2px !important; }
[data-testid="stSidebar"] .stRadio div[role="radio"][aria-checked="true"] { background:rgba(255,255,255,0.22) !important; font-weight:700 !important; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,0.18) !important; }
[data-testid="stSidebar"] .stButton button { background:rgba(255,255,255,0.12) !important; color:#E8F5E9 !important; border:1px solid rgba(255,255,255,0.25) !important; border-radius:10px !important; width:100%; }
[data-testid="stSidebar"] .stButton button:hover { background:rgba(255,255,255,0.22) !important; }

/* Agent badge pills in sidebar */
.agent-pill { display:flex; align-items:center; gap:8px; background:rgba(255,255,255,0.08); border-radius:10px; padding:8px 12px; margin-bottom:4px; }
.agent-dot { width:8px; height:8px; border-radius:50%; background:#25D366; flex-shrink:0; }
.agent-pill span { font-size:0.85rem; color:#E8F5E9; }

/* ── Chat header ── */
.chat-header { background:#075E54; padding:12px 20px; display:flex; align-items:center; gap:14px; box-shadow:0 2px 8px rgba(0,0,0,0.18); }
.chat-header-avatar { width:44px; height:44px; background:linear-gradient(135deg,#25D366,#128C7E); border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:22px; flex-shrink:0; }
.chat-header-info h3 { margin:0; color:#fff; font-size:1rem; font-weight:700; line-height:1.2; }
.chat-header-info span { color:#A7DFB3; font-size:0.75rem; }

/* ── Bubbles ── */
.msg-row { display:flex; margin:3px 16px; }
.msg-row.user { justify-content:flex-end; }
.msg-row.bot  { justify-content:flex-start; }

.bubble { max-width:65%; padding:8px 12px 5px; border-radius:8px; font-size:0.88rem; line-height:1.5; box-shadow:0 1px 2px rgba(0,0,0,0.13); word-wrap:break-word; }
.bubble.user { background:#DCF8C6; color:#111; border-top-right-radius:2px; }
.bubble.bot  { background:#ffffff; color:#111; border-top-left-radius:2px; }

.agent-tag { display:inline-block; font-size:0.68rem; font-weight:700; color:#075E54; background:#E8F5E9; border-radius:20px; padding:2px 8px; margin-bottom:5px; }
.msg-time { font-size:0.64rem; color:#999; text-align:right; margin-top:3px; }

/* image bubble */
.img-bubble { max-width:260px; background:#DCF8C6; border-radius:8px; border-top-right-radius:2px; padding:5px; box-shadow:0 1px 2px rgba(0,0,0,0.13); }
.img-bubble img { width:100%; border-radius:5px; display:block; }
.img-bubble .caption { font-size:0.82rem; color:#111; padding:4px 3px 1px; }

/* date separator */
.date-sep { text-align:center; margin:10px 0; }
.date-sep span { background:rgba(0,0,0,0.12); color:#555; font-size:0.7rem; padding:3px 12px; border-radius:10px; }

/* ── Upload ── */
[data-testid="stFileUploaderDropzone"] { background:#fff !important; border:2px dashed #25D366 !important; border-radius:12px !important; }

/* ── Chat input ── */
[data-testid="stChatInput"] { background:#F0F0F0 !important; padding:8px 16px !important; }
[data-testid="stChatInput"] textarea { background:#fff !important; border-radius:24px !important; border:none !important; padding:10px 16px !important; font-size:0.9rem !important; color:#111 !important; box-shadow:0 1px 3px rgba(0,0,0,0.1); }
[data-testid="stChatInput"] button { background:#25D366 !important; border-radius:50% !important; }

/* ── Dashboard ── */
.dash-header { background:#075E54; padding:18px 24px; margin-bottom:20px; }
.dash-header h2 { margin:0; font-size:1.3rem; font-weight:800; color:#fff; }
.dash-header p { margin:3px 0 0; font-size:0.8rem; color:#A7DFB3; }
.metric-card { background:white; padding:18px 14px; border-radius:14px; border-left:4px solid #25D366; box-shadow:0 2px 8px rgba(0,0,0,0.07); text-align:center; margin-bottom:16px; }
.metric-title { color:#666; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; margin-bottom:6px; }
.metric-value { color:#075E54; font-size:1.8rem; font-weight:800; }
.alert-card { background:#FFF8E1; border-left:4px solid #FFA000; border-radius:10px; padding:12px 16px; font-size:0.88rem; color:#5D4037; margin-top:4px; }

/* hide streamlit chrome */
#MainMenu, footer, header { visibility:hidden; }
[data-testid="stChatMessage"] { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "chat" not in st.session_state:
    st.session_state.chat = []   # unified message list

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    lang_choice = st.selectbox("🌐 Language", ["English", "Tamil"])
    t = LANG_DATA[lang_choice]

    st.markdown("---")
    st.markdown(f"""
        <div style='text-align:center; padding:4px 0 12px;'>
            <div style='font-size:2rem;'>🌿</div>
            <div style='font-weight:800; font-size:1.1rem; color:#fff;'>{t['title']}</div>
            <div style='font-size:0.72rem; color:#A5D6A7;'>{t['subtitle']}</div>
        </div>
    """, unsafe_allow_html=True)

    # Page nav — only Dashboard vs Mali Agent
    nav_labels = [f"📊  {t['dashboard']}", f"🤖  {t['mali_agent']}"]
    menu_label = st.radio("NAVIGATE", nav_labels)
    menu = "dashboard" if t["dashboard"] in menu_label else "agent"

    st.markdown("---")

    # Active agents display
    st.markdown("<div style='font-size:0.68rem; color:#A5D6A7; text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px;'>ACTIVE AGENTS</div>", unsafe_allow_html=True)
    for agent in ["🌦️ Weather Agent", "🌱 Soil Agent", "👁️ Vision Agent", "🧪 Fertilizer Agent"]:
        st.markdown(f"<div class='agent-pill'><div class='agent-dot'></div><span>{agent}</span></div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button(t["reset"]):
        st.session_state.chat = []
        st.rerun()

# ─────────────────────────────────────────────
# PARSE AGENT TAG from LLM response
# ─────────────────────────────────────────────
def parse_agent_response(raw: str):
    """Extract [AGENT: XYZ] tag and return (agent_name, clean_text)."""
    lines = raw.strip().split("\n")
    agent_name = "🌿 Mali Agent"
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[AGENT:") and stripped.endswith("]"):
            agent_name = stripped[7:-1].strip()
        else:
            clean_lines.append(line)
    return agent_name, "\n".join(clean_lines).strip()

# ─────────────────────────────────────────────
# DASHBOARD PAGE
# ─────────────────────────────────────────────
def render_dashboard():
    st.markdown(f"""
    <div class="dash-header">
        <h2>📊 {t['dashboard']}</h2>
        <p>Real-time farm monitoring</p>
    </div>""", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div style='padding:0 20px;'>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            val = (100 - (w_data['ensemble_mae']['rainfall'] * 100)) if w_data else 96.6
            st.markdown(f"<div class='metric-card'><div class='metric-title'>{t['rain']}</div><div class='metric-value'>{val:.1f}%</div></div>", unsafe_allow_html=True)
        with m2:
            status = s_data['Ensemble']['prediction'].upper() if s_data else "STABLE"
            st.markdown(f"<div class='metric-card'><div class='metric-title'>{t['soil']}</div><div class='metric-value'>{status}</div></div>", unsafe_allow_html=True)
        with m3:
            hum = df_station.iloc[0]['hum'] if df_station is not None else 84
            st.markdown(f"<div class='metric-card'><div class='metric-title'>{t['hum']}</div><div class='metric-value'>{hum}%</div></div>", unsafe_allow_html=True)
        with m4:
            co2 = df_station.iloc[0]['co2'] if df_station is not None else 415
            st.markdown(f"<div class='metric-card'><div class='metric-title'>{t['co2']}</div><div class='metric-value'>{co2} ppm</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown("#### 🌡️ Temperature Telemetry")
            if df_station is not None:
                st.line_chart(df_station.head(25).set_index('Time')['temp'])
            else:
                dummy = pd.DataFrame({"temp": np.random.uniform(28, 38, 25)})
                st.line_chart(dummy)
        with col_b:
            st.markdown(f"#### 🔔 {t['alerts']}")
            st.markdown(f"<div class='alert-card'>⚠️ {t['alert_msg']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# UNIFIED MALI AGENT PAGE
# ─────────────────────────────────────────────
def render_mali_agent():
    # Header
    st.markdown(f"""
    <div class="chat-header">
        <div class="chat-header-avatar">🌿</div>
        <div class="chat-header-info">
            <h3>{t['mali_agent']}</h3>
            <span>● {t['online']}</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # Image uploader — always visible at top
    uploaded_file = st.file_uploader(
        t["upload"], type=["jpg", "png", "jpeg"], key="main_uploader",
        label_visibility="collapsed"
    )
    if uploaded_file:
        st.markdown(f"<div style='padding:4px 16px; font-size:0.78rem; color:#075E54;'>📎 {uploaded_file.name} attached — Vision Agent will be used</div>", unsafe_allow_html=True)

    # Render chat history
    st.markdown("<div style='padding:6px 0 90px;'>", unsafe_allow_html=True)
    today = datetime.date.today().strftime("%B %d, %Y")
    st.markdown(f"<div class='date-sep'><span>TODAY · {today}</span></div>", unsafe_allow_html=True)

    for msg in st.session_state.chat:
        role     = msg["role"]
        content  = msg["content"]
        time_str = msg.get("time", "")

        if role == "user":
            if msg.get("has_image"):
                st.markdown(f"""
                <div class="msg-row user">
                    <div class="img-bubble">
                        <img src="data:image/jpeg;base64,{msg['img_b64']}" />
                        <div class="caption">{content}</div>
                        <div class="msg-time">{time_str}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-row user">
                    <div class="bubble user">
                        {content}
                        <div class="msg-time">{time_str}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            agent_name = msg.get("agent", "🌿 Mali Agent")
            content_html = content.replace("\n", "<br>")
            st.markdown(f"""
            <div class="msg-row bot">
                <div class="bubble bot">
                    <div class="agent-tag">{agent_name}</div><br>
                    {content_html}
                    <div class="msg-time">{time_str}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Chat input ──
    if prompt := st.chat_input(t["ask"], key="main_input"):
        now = datetime.datetime.now().strftime("%H:%M")
        env_info = (
            f"Weather MAE: {w_data['ensemble_mae']['rainfall'] if w_data else 'N/A'}. "
            f"Soil status: {s_data['Ensemble']['prediction'] if s_data else 'Stable'}."
        )

        # Save user message
        user_entry = {"role": "user", "content": prompt, "time": now}
        img_b64 = None

        if uploaded_file:
            img_b64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
            user_entry["has_image"] = True
            user_entry["img_b64"]   = img_b64

        st.session_state.chat.append(user_entry)

        # ── Call Groq ──
        sys_prompt = t["router_sys"]
        model_id   = VISION_MODEL if uploaded_file else TEXT_MODEL

        try:
            if uploaded_file and img_b64:
                user_content = [
                    {"type": "text",      "text": f"Farm context: {env_info}\n\nUser question: {prompt}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            else:
                user_content = f"Farm context: {env_info}\n\nUser question: {prompt}"

            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": user_content}
                ],
                stream=False
            )
            raw = response.choices[0].message.content
            agent_name, clean_text = parse_agent_response(raw)

        except Exception as e:
            agent_name = "⚠️ System"
            clean_text = f"Agent error: {str(e)}. Please try again."

        st.session_state.chat.append({
            "role":    "assistant",
            "agent":   agent_name,
            "content": clean_text,
            "time":    now
        })
        st.rerun()

# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
if menu == "dashboard":
    render_dashboard()
else:
    render_mali_agent()