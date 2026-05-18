import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import base64
from data_loader import get_all_sections
import random
import re


ALL_CARDS = [
    ("Composites",  "Material class",  "material", "Composites"),
    ("Polymers",    "Material class",  "material", "Polymers"),
    ("Fibers",      "Material class",  "material", "Fibers"),
]

sections = get_all_sections()
for section in sections:
    ALL_CARDS.append((section, "Property type", "section", section))

if "visible_cards" not in st.session_state:
    random.shuffle(ALL_CARDS)
    st.session_state.visible_cards = ALL_CARDS[:4]  

VISIBLE_CARDS = st.session_state.visible_cards 
prop_count = len([c for c in ALL_CARDS if c[2] == "section"])

def get_card_icon(title: str, card_type: str) -> str:
    if card_type == "material":
        icons = {"composites": "🧱", "polymers": "🔬", "fibers": "🧵"}
        return icons.get(title.lower(), "🧱")
    t = title.lower()
    if "mechanical" in t: return "⚙️"
    if "thermal" in t: return "🔥"
    if "electrical" in t: return "⚡"
    if "physical" in t: return "⚖️"
    if "processing" in t: return "🔧"
    if "optical" in t: return "🔭"
    if "chemical" in t: return "🧪"
    if "flammab" in t: return "🔴"
    if "component" in t: return "🧩"
    if "descriptive" in t: return "📋"
    return "📋"

def img_to_b64(path):
    try:
        ext = Path(path).suffix.lower().replace(".", "")
        mime = "png" if ext == "png" else "jpeg"
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/{mime};base64,{data}"
    except Exception:
        return ""


home_img = img_to_b64("images/Home.png")
logo_img = img_to_b64("logo.png")

st.markdown("""
  <style>
  section[data-testid="stMain"] {
      background: #fff !important;
  }
  .stApp {
      background: #fff !important;
  }
  </style>
  """, 
  unsafe_allow_html=True
  )

#  Global style overrides 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

/* Hide default Streamlit chrome */
.block-container { padding: 0 !important; max-width: 100% !important; background: #fff !important;}
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stHeader"] { display: none !important; }





/* ── Search row ── */
/* Only target the search bar's horizontal block, not all columns */
.st-emotion-cache-y1l7l5 .st-emotion-cache-1permvm {
    gap: 0 !important;
    align-items: stretch !important;
}

.st-emotion-cache-16s2yzk button {
    background: #f2f6f4 !important;
    border: 1.5px solid #d0d8d4 !important;
    border-right: none !important;
    border-radius: 50px 0 0 50px !important;
    color: #3a5248 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    height: 46px !important;
    padding: 0 16px !important;
    white-space: nowrap !important;
    box-shadow: none !important;
}

.st-emotion-cache-4dubyl [data-baseweb="input"] {
    background: #000 !important;
    height: 46px !important;
    min-height: 46px !important;
    border-radius: 0 !important;
    border-color: #d0d8d4 !important;
    border-left: none !important;
    border-right: none !important;
}
            
.st-emotion-cache-4dubyl input {
    background: #fff !important;
    height: 46px !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    color: #000 !important;
    box-shadow: none !important;
    padding-left: 18px !important;
}

.st-emotion-cache-4dubyl > div {

    height: 46px !important;
}
.st-emotion-cache-4dubyl input::placeholder {
    color: #0f1f1a !important;
    opacity: 0.4 !important;
}
.st-emotion-cache-mpgwbc button {
    background: #8ACAFF !important;
    border: 1.5px solid #8ACAFF !important;
    border-left: none !important;
    border-radius: 0 50px 50px 0 !important;
    color: #0f1f1a !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    height: 46px !important;
    padding: 0 28px !important;
    box-shadow: none !important;
}

.st-key-view_all_btn button {
    background: transparent !important;
    border: none !important;
    color: #8ACAFF !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0 !important;
    height: auto !important;
    box-shadow: none !important;
    cursor: pointer !important;
}
.st-key-view_all_btn button:hover {
    color: #8ACAFF !important;
    text-decoration: underline !important;
    background: transparent !important;
}
.footer-nav-head {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #0f1f1a;
    margin-bottom: 14px;
    font-family: 'DM Sans', sans-serif;
}

</style>
""", unsafe_allow_html=True)


#  Helper 
def go_categorized(material=None, search=None):
    if material:
        st.session_state.material_type = material
    if search:
        st.session_state.search_term = search
    st.switch_page("page_files/Categorized_Search.py")


def go_upload():
    st.switch_page("page_files/Upload_Data.py")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ANIMATION SECTION  (pure HTML, no clickables needed)
# ══════════════════════════════════════════════════════════════════════════════
about_img_html = (
    f"<div class='aim-about-img'><img src='{home_img}' alt='AIM platform diagram'/></div>"
    if home_img else
    "<div class='aim-about-img-placeholder'>[ Platform diagram ]</div>"
)

logo_html = (
    f"<img src='{logo_img}' alt='AIM Logo' style='height:52px;width:52px;object-fit:contain;border-radius:14px;'/>"
    if logo_img else ""
)

components.html(f"""
<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"/>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'DM Sans', sans-serif; background: #f7f7f5; color: #1a2e26; overflow-x: hidden; }}

/*  ANIMATION SECTION  */
.anim-section {{
    background: #000;
    padding: 80px 40px 60px;
    display: flex; flex-direction: column; align-items: center;
}}
.anim-title {{
    font-size: clamp(1.6rem, 3vw, 2.4rem); font-weight: 800;
    color: #fff; text-align: center; letter-spacing: -1px; margin-bottom: 60px;
}}
.anim-stage {{
    position: relative; width: 560px; height: 320px;
    display: flex; align-items: center;
    justify-content: space-between; padding: 0 40px;
}}

/* MAG GLASS */
.mag-wrap {{ position: relative; width: 100px; display: flex; flex-direction: column; align-items: center; }}
.mag-label {{ font-size: 0.65rem; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: #4a7a6a; margin-bottom: 10px; }}
.mag-svg {{ width: 90px; height: 160px; overflow: visible; }}
.mag-lens-flash {{ opacity: 0; transition: opacity 0.6s ease 0.5s; }}
.mag-wrap.active .mag-lens-flash {{ opacity: 1; animation: lensFlash 1.8s ease-in-out infinite 0.5s; }}
@keyframes lensFlash {{ 0%,100% {{ opacity:.2; r:26; }} 50% {{ opacity:.7; r:28; }} }}
.mag-scan {{ opacity: 0; transition: opacity 0.3s ease 0.8s; }}
.mag-wrap.active .mag-scan {{ opacity: 1; animation: scanMove 1.2s ease-in-out infinite 0.8s; }}
@keyframes scanMove {{ 0% {{ transform:translateY(-14px);opacity:.8; }} 50% {{ transform:translateY(0);opacity:1; }} 100% {{ transform:translateY(14px);opacity:.8; }} }}

/* BUBBLES */
.bubble {{ position:absolute; border-radius:50%; background:radial-gradient(circle at 35% 35%,rgba(255,255,255,.6),rgba(80,160,255,.25)); border:1px solid rgba(100,180,255,.4); opacity:0; pointer-events:none; backdrop-filter:blur(2px); }}
.b1 {{ width:18px;height:18px;left:108px;bottom:155px; }}
.b2 {{ width:13px;height:13px;left:120px;bottom:175px; }}
.b3 {{ width:22px;height:22px;left:96px; bottom:140px; }}
.b4 {{ width:10px;height:10px;left:128px;bottom:185px; }}
.b5 {{ width:16px;height:16px;left:112px;bottom:165px; }}
.b6 {{ width:8px; height:8px; left:124px;bottom:195px; }}
.anim-stage.active .b1 {{ animation:floatBubble1 2.6s cubic-bezier(.25,.46,.45,.94) 1.2s forwards; }}
.anim-stage.active .b2 {{ animation:floatBubble2 2.2s cubic-bezier(.25,.46,.45,.94) 1.5s forwards; }}
.anim-stage.active .b3 {{ animation:floatBubble3 3.0s cubic-bezier(.25,.46,.45,.94) 1.0s forwards; }}
.anim-stage.active .b4 {{ animation:floatBubble4 2.0s cubic-bezier(.25,.46,.45,.94) 1.8s forwards; }}
.anim-stage.active .b5 {{ animation:floatBubble5 2.8s cubic-bezier(.25,.46,.45,.94) 1.3s forwards; }}
.anim-stage.active .b6 {{ animation:floatBubble6 1.8s cubic-bezier(.25,.46,.45,.94) 2.0s forwards; }}
@keyframes floatBubble1 {{ 0% {{ opacity:0;transform:translate(0,0) scale(.4); }} 15% {{ opacity:1;transform:translate(4px,-20px) scale(1); }} 60% {{ opacity:.9;transform:translate(120px,-55px) scale(.9); }} 85% {{ opacity:.5;transform:translate(280px,-30px) scale(.6); }} 100% {{ opacity:0;transform:translate(340px,-10px) scale(.2); }} }}
@keyframes floatBubble2 {{ 0% {{ opacity:0;transform:translate(0,0) scale(.3); }} 15% {{ opacity:1;transform:translate(-5px,-28px) scale(1); }} 60% {{ opacity:.9;transform:translate(110px,-70px) scale(.85); }} 85% {{ opacity:.4;transform:translate(265px,-40px) scale(.5); }} 100% {{ opacity:0;transform:translate(330px,-15px) scale(.15); }} }}
@keyframes floatBubble3 {{ 0% {{ opacity:0;transform:translate(0,0) scale(.5); }} 15% {{ opacity:1;transform:translate(6px,-15px) scale(1); }} 55% {{ opacity:.9;transform:translate(130px,-45px) scale(.95); }} 80% {{ opacity:.4;transform:translate(278px,-20px) scale(.55); }} 100% {{ opacity:0;transform:translate(345px,-5px) scale(.2); }} }}
@keyframes floatBubble4 {{ 0% {{ opacity:0;transform:translate(0,0) scale(.3); }} 20% {{ opacity:1;transform:translate(-8px,-35px) scale(1); }} 65% {{ opacity:.8;transform:translate(105px,-80px) scale(.8); }} 88% {{ opacity:.3;transform:translate(255px,-50px) scale(.4); }} 100% {{ opacity:0;transform:translate(320px,-20px) scale(.1); }} }}
@keyframes floatBubble5 {{ 0% {{ opacity:0;transform:translate(0,0) scale(.4); }} 15% {{ opacity:1;transform:translate(3px,-22px) scale(1); }} 58% {{ opacity:.9;transform:translate(115px,-60px) scale(.88); }} 83% {{ opacity:.45;transform:translate(270px,-35px) scale(.55); }} 100% {{ opacity:0;transform:translate(335px,-12px) scale(.2); }} }}
@keyframes floatBubble6 {{ 0% {{ opacity:0;transform:translate(0,0) scale(.25); }} 20% {{ opacity:1;transform:translate(-3px,-40px) scale(1); }} 62% {{ opacity:.7;transform:translate(100px,-85px) scale(.75); }} 85% {{ opacity:.25;transform:translate(250px,-55px) scale(.35); }} 100% {{ opacity:0;transform:translate(318px,-22px) scale(.1); }} }}

/* DATABASE */
.db-wrap {{ position:relative;width:130px;display:flex;flex-direction:column;align-items:center; }}
.db-label {{ font-size:.65rem;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;color:#4a7a6a;margin-bottom:10px; }}
.cyl {{ opacity:.15;transition:opacity .5s ease; }}
.anim-stage.active .cyl-left  {{ opacity:1;transition-delay:2.0s; }}
.anim-stage.active .cyl-right {{ opacity:1;transition-delay:2.2s; }}
.anim-stage.active .cyl-front {{ opacity:1;transition-delay:2.4s;animation:frontPulse 2.4s ease-in-out infinite 2.8s; }}
@keyframes frontPulse {{ 0%,100% {{ filter:brightness(1); }} 50% {{ filter:brightness(1.2); }} }}

/* CAPTION */
.anim-caption {{ margin-top:48px;display:flex;gap:40px;align-items:center; }}
.anim-step {{ display:flex;align-items:center;gap:10px;opacity:0;transform:translateY(10px);transition:opacity .5s ease,transform .5s ease; }}
.anim-stage.active ~ .anim-caption .anim-step:nth-child(1) {{ opacity:1;transform:none;transition-delay:.5s; }}
.anim-stage.active ~ .anim-caption .anim-step:nth-child(2) {{ opacity:1;transform:none;transition-delay:1s; }}
.anim-stage.active ~ .anim-caption .anim-step:nth-child(3) {{ opacity:1;transform:none;transition-delay:1.5s; }}
.step-text {{ font-size:.78rem;color:#5a8a7a;font-weight:500; }}
</style>
</head><body>

<section class="anim-section">
    <h2 class="anim-title">From Experiment to Database</h2>
    <div class="anim-stage" id="animStage">
        <!-- MAG GLASS -->
        <div class="mag-wrap" id="magWrap">
            <div class="mag-label">Research</div>
            <svg class="mag-svg" viewBox="0 0 90 160" fill="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <radialGradient id="lensGrad" cx="40%" cy="38%" r="55%">
                        <stop offset="0%" stop-color="#4da6ff" stop-opacity="0.25"/>
                        <stop offset="100%" stop-color="#1a4a8a" stop-opacity="0.7"/>
                    </radialGradient>
                    <radialGradient id="flashGrad" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stop-color="#8ACAFF" stop-opacity="0.9"/>
                        <stop offset="100%" stop-color="#8ACAFF" stop-opacity="0"/>
                    </radialGradient>
                </defs>
                <circle cx="38" cy="38" r="32" fill="none" stroke="#2a5a8a" stroke-width="3"/>
                <circle cx="38" cy="38" r="29" fill="url(#lensGrad)"/>
                <line class="mag-scan" x1="16" y1="38" x2="60" y2="38" stroke="#8ACAFF" stroke-width="1.5" stroke-linecap="round" opacity="0.8"/>
                <circle class="mag-lens-flash" cx="38" cy="38" r="26" fill="url(#flashGrad)"/>
                <circle cx="24" cy="24" r="5" fill="white" opacity="0.15"/>
                <circle cx="30" cy="34" r="2" fill="#8ACAFF" opacity="0.7"/>
                <circle cx="42" cy="30" r="2" fill="#8ACAFF" opacity="0.5"/>
                <circle cx="38" cy="44" r="2" fill="#8ACAFF" opacity="0.6"/>
                <circle cx="48" cy="38" r="1.5" fill="#8ACAFF" opacity="0.4"/>
            </svg>
        </div>
        <!-- BUBBLES -->
        <div class="bubble b1"></div><div class="bubble b2"></div>
        <div class="bubble b3"></div><div class="bubble b4"></div>
        <div class="bubble b5"></div><div class="bubble b6"></div>
        <!-- DATABASE -->
        <div class="db-wrap">
            <div class="db-label">Database</div>
            <svg width="220" height="200" viewBox="0 0 220 200" fill="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="sideBlue" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stop-color="#2a55b0"/><stop offset="50%" stop-color="#4a7fd8"/><stop offset="100%" stop-color="#2a55b0"/>
                    </linearGradient>
                    <linearGradient id="sideDark" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stop-color="#0d1f5c"/><stop offset="50%" stop-color="#1a3a8a"/><stop offset="100%" stop-color="#0d1f5c"/>
                    </linearGradient>
                    <linearGradient id="topBlue" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="#6a9fe0"/><stop offset="100%" stop-color="#3a6abf"/>
                    </linearGradient>
                    <linearGradient id="topDark" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="#2a4a9a"/><stop offset="100%" stop-color="#0d1f5c"/>
                    </linearGradient>
                </defs>
                <!-- back left -->
                <g class="cyl cyl-left">
                    <rect x="10" y="88" width="66" height="20" rx="1" fill="url(#sideBlue)"/>
                    <ellipse cx="43" cy="88" rx="33" ry="10" fill="url(#topBlue)"/>
                    <rect x="10" y="66" width="66" height="24" rx="1" fill="url(#sideBlue)"/>
                    <ellipse cx="43" cy="66" rx="33" ry="10" fill="url(#topBlue)"/>
                    <rect x="10" y="46" width="66" height="22" rx="1" fill="url(#sideBlue)"/>
                    <ellipse cx="43" cy="46" rx="33" ry="10" fill="url(#topBlue)"/>
                    <ellipse cx="43" cy="108" rx="33" ry="10" fill="#1e3a80"/>
                </g>
                <!-- back right -->
                <g class="cyl cyl-right">
                    <rect x="144" y="88" width="66" height="20" rx="1" fill="url(#sideBlue)"/>
                    <ellipse cx="177" cy="88" rx="33" ry="10" fill="url(#topBlue)"/>
                    <rect x="144" y="66" width="66" height="24" rx="1" fill="url(#sideBlue)"/>
                    <ellipse cx="177" cy="66" rx="33" ry="10" fill="url(#topBlue)"/>
                    <rect x="144" y="46" width="66" height="22" rx="1" fill="url(#sideBlue)"/>
                    <ellipse cx="177" cy="46" rx="33" ry="10" fill="url(#topBlue)"/>
                    <ellipse cx="177" cy="108" rx="33" ry="10" fill="#1e3a80"/>
                </g>
                <!-- front center -->
                <g class="cyl cyl-front">
                    <rect x="70" y="128" width="80" height="24" rx="1" fill="url(#sideDark)"/>
                    <ellipse cx="110" cy="128" rx="40" ry="13" fill="url(#topDark)"/>
                    <rect x="70" y="102" width="80" height="28" rx="1" fill="url(#sideDark)"/>
                    <ellipse cx="110" cy="102" rx="40" ry="13" fill="url(#topDark)"/>
                    <rect x="70" y="76" width="80" height="28" rx="1" fill="url(#sideDark)"/>
                    <ellipse cx="110" cy="76" rx="40" ry="13" fill="url(#topDark)"/>
                    <rect x="70" y="54" width="80" height="24" rx="1" fill="url(#sideDark)"/>
                    <ellipse cx="110" cy="54" rx="40" ry="13" fill="url(#topDark)"/>
                    <ellipse cx="110" cy="152" rx="40" ry="13" fill="#080f2a"/>
                </g>
            </svg>
        </div>
    </div>
    <div class="anim-caption">
        <div class="anim-step"><span class="step-text">Collect measurements</span></div>
        <div class="anim-step"><span class="step-text">Process &amp; validate</span></div>
        <div class="anim-step"><span class="step-text">Stored in AIM</span></div>
    </div>
</section>

<script>
const stage = document.getElementById('animStage');
const magWrap = document.getElementById('magWrap');
const observer = new IntersectionObserver(entries => {{
    entries.forEach(e => {{
        if (e.isIntersecting) {{
            setTimeout(() => {{ magWrap.classList.add('active'); stage.classList.add('active'); }}, 300);
        }} else {{
            magWrap.classList.remove('active'); stage.classList.remove('active');
        }}
    }});
}}, {{ threshold: 0.4 }});
observer.observe(document.querySelector('.anim-section'));
</script>
</body></html>
""", height=700, scrolling=False)

# ══════════════════════════════════════════════════════════════════════════════
# 2.  HERO , heading + description (static HTML)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.aim-hero-text {
    background: #fff;
    text-align: center;
    padding: 64px 40px 24px;
    border-bottom: none;
}
.aim-hero-text h1 {
    font-family: 'DM Sans', sans-serif;
    font-size: clamp(2rem, 4.5vw, 3.2rem);
    font-weight: 800; color: #0f1f1a;
    line-height: 1.1; letter-spacing: -1.5px;
    margin-bottom: 18px;
}
.aim-hero-text p {
    color: #5a6b65; font-size: 1rem; line-height: 1.65;
    max-width: 500px; margin: 0 auto 28px;
}
</style>
""", unsafe_allow_html=True)





# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAJOR CATEGORIES , Streamlit columns + containers + buttons
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Card buttons */
.st-emotion-cache-r3ry0f button {
    background: #fff !important;
    border: 1.5px solid #e8eceb !important;
    border-radius: 16px !important;
    padding: 28px 24px !important;
    height: 180px !important;
    width: 100% !important;
    text-align: left !important;
    white-space: pre-wrap !important;
    line-height: 1.5 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    transition: box-shadow 0.22s ease, border-color 0.22s ease, transform 0.18s ease !important;
}

.st-emotion-cache-r3ry0f button:hover {
    border-color: #8ACAFF !important;
    box-shadow: 0 0 0 3px rgba(138,202,255,0.18), 0 8px 28px rgba(0,0,0,0.09) !important;
    background: #fff !important;
    transform: translateY(-2px) !important;
}

/* Icon badge */
.st-emotion-cache-r3ry0f button p:first-child {
    font-size: 1.25rem !important;
    background: linear-gradient(135deg, #f0f7ff 0%, #e8f4ff 100%) !important;
    border: 1px solid #d4eaff !important;
    border-radius: 10px !important;
    padding: 8px 10px !important;
    display: inline-block !important;
    margin-bottom: 16px !important;
    line-height: 1 !important;
    box-shadow: 0 1px 3px rgba(138,202,255,0.15) !important;
}

/* Title */
.st-emotion-cache-r3ry0f button p:nth-child(2) {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #0f1f1a !important;
    margin-bottom: 8px !important;
    letter-spacing: -0.2px !important;
}

/* Tag label */
.st-emotion-cache-r3ry0f button p:last-child {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    color: #8ACAFF !important;
    text-transform: uppercase !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("<div style='padding: 64px 0 0 0;'></div>", unsafe_allow_html=True)

# st.markdown("""
#     <div class="cards-section-wrap">
#         <h2>Quick Links</h2>
#         <p>Open the pages you need most.</p>
#     </div>
#     """, unsafe_allow_html=True)

quick_cards = [
    ("Search", "Data Page", "page_files/Categorized_Search.py", "search", "🔎"),
    ("Extract Data", "Platform", "page_files/Upload_Data.py", "about", "📋"),
]

col1, col2 = st.columns([5, 4], gap="large")
with col1:
    
    st.markdown("""
        <div style="padding: 48px 40px 24px 0px;">
        <h1 style="font-family:'DM Sans',sans-serif; font-size:clamp(2rem,4.5vw,3.2rem); font-weight:800; color:#0f1f1a; line-height:1.1; letter-spacing:-1.5px; margin-bottom:18px;">
            Accelerate Your Composites Research
        </h1>
        <p style="color:#5a6b65; font-size:1rem; line-height:1.65; margin:0;">
            Access a centralized, open-source database for experimental composite material properties.
            Polymer, fiber, and composite datasets — all in one place.
        </p>
        </div>
""", unsafe_allow_html=True)
with col2:
    st.markdown("<div style='padding-top: 48px;'></div>", unsafe_allow_html=True)
    cols = st.columns(2, gap="large")
    for col, (title, tag, target_page, key_suffix, badge) in zip(cols, quick_cards):
        with col:
            if st.button(
                f"{badge}\n\n{title}\n\n{tag}",
                key=f"quick_{key_suffix}_page",
                use_container_width=True,
            ):
                st.switch_page(target_page)
            
        st.markdown("<div style='background:#fff; padding: 24px 0; border-bottom: 1px solid #e8e8e4;'></div>", unsafe_allow_html=True)

        

# ══════════════════════════════════════════════════════════════════════════════
# 4.  STATS (static HTML)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
.aim-stats {{
    background: #f7f7f5; border-bottom: 1px solid #e4e8e5;
    display: flex; justify-content: center;
}} 
.aim-stat {{
    text-align: center; padding: 34px 56px;
    border-right: 1px solid #e4e8e5;
}}
.aim-stat:last-child {{ border-right: none; }}
.aim-stat-num {{
    font-family: 'DM Sans', sans-serif;
    font-size: 2.1rem; font-weight: 800;
    color: #8ACAFF; line-height: 1; margin-bottom: 6px;
}}
.aim-stat-label {{
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 1.4px; color: #7a8e87; text-transform: uppercase;
}}
</style>
<div class="aim-stats">
    <div class="aim-stat"><div class="aim-stat-num">3</div><div class="aim-stat-label">Material Classes</div></div>
    <div class="aim-stat"><div class="aim-stat-num">{prop_count}</div><div class="aim-stat-label">Properties Tracked</div></div>
    <div class="aim-stat"><div class="aim-stat-num">8</div><div class="aim-stat-label">Research Teams</div></div>
    <div class="aim-stat"><div class="aim-stat-num">2</div><div class="aim-stat-label">Universities</div></div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 6.  ABOUT + FOOTER (static HTML)
# ══════════════════════════════════════════════════════════════════════════════
components.html(f"""
<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"/>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'DM Sans', sans-serif; color: #1a2e26; }}

.aim-about {{ background: #fff; padding: 60px; border-bottom: 1px solid #e8e8e4; }}
.aim-about h2 {{ font-size: 1.65rem; font-weight: 700; color: #0f1f1a; letter-spacing: -0.5px; margin-bottom: 8px; }}
.aim-about-sub {{ color: #6a7e77; font-size: 0.9rem; margin-bottom: 30px; }}
.aim-about-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 48px; align-items: start; }}
.aim-about-text p {{ color: #3a4e47; font-size: 0.91rem; line-height: 1.75; margin-bottom: 14px; }}
.aim-about-img {{ border-radius: 10px; overflow: hidden; border: 1px solid #e4e8e5; }}
.aim-about-img img {{ width: 100%; display: block; }}
.aim-about-img-placeholder {{
    background: #f0f5f3; border-radius: 10px; height: 280px;
    display: flex; align-items: center; justify-content: center;
    color: #7a9e8f; font-size: 0.85rem; border: 1.5px dashed #c8d6d0;
}}

.aim-footer {{ background: #fff; border-top: 1px solid #e8e8e4; padding: 50px 60px 28px; }}
.aim-footer-brand {{ display: flex; align-items: center; gap: 8px; font-weight: 700; color: #0f1f1a; font-size: 0.95rem; margin-bottom: 10px; }}
.aim-footer-desc {{ font-size: 0.82rem; color: #7a8e87; line-height: 1.6; }}
</style>
</head><body>

<section class="aim-about">
    <h2>About the Platform</h2>
    <p class="aim-about-sub">Artificially Intelligent Manufacturing Paradigm (AIM) for Composites</p>
    <div class="aim-about-grid">
        <div class="aim-about-text">
            <p>The AIM Database tool serves as a powerful, centralized hub designed to streamline
            collaboration and information exchange within the composite materials research community.
            The platform enables researchers to contribute to a shared knowledge base by uploading
            experimental datasets through secure terminals.</p>
            <p>Users can submit specific measurements regarding mechanical properties, thermal behavior,
            and rheology, alongside their published journal papers , ensuring that both raw data and
            peer-reviewed findings are integrated into one cohesive system.</p>
            <p>All contributed information is securely aggregated within a central cloud architecture,
            allowing for efficient storage, organization, and retrieval across polymer, fiber, and
            composite categories.</p>
        </div>
        <div>{about_img_html}</div>
    </div>
</section>



</body></html>
""", height=550, scrolling=False)

st.markdown("""
<style>
            
/* Footer nav wrapper */
div[data-testid="stHorizontalBlock"]:has(.footer-nav-head) {
    background: #fff !important;
    border-bottom: 1px solid #e8e8e4 !important;
    padding: 0 60px 28px !important;
    margin-top: -16px !important;
    
}
.aim-footer-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 700;
    color: #0f1f1a;
    font-size: 0.95rem;
    margin-bottom: 10px;
    font-family: 'DM Sans', sans-serif;
}
.aim-footer-desc {
    font-size: 0.82rem;
    color: #7a8e87;
    line-height: 1.6;
    font-family: 'DM Sans', sans-serif;
}
.footer-nav-head {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #0f1f1a;
    margin-bottom: 14px;
    font-family: 'DM Sans', sans-serif;
}
.st-emotion-cache-1ofqig9 {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.st-emotion-cache-1qeq59m {
    background: transparent !important;
    border: none !important;
    text-decoration: none !important;
    padding: 0 0 9px 0 !important;
    display: block !important;
}
.st-emotion-cache-1qeq59m:hover {
    background: transparent !important;
    border: none !important;
}
.st-emotion-cache-1qeq59m p {
    color: #6a7e77 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 400 !important;
    margin: 0 !important;
}
.st-emotion-cache-1qeq59m:hover p {
    color: #8ACAFF !important;
}

/* copyright bar */
.aim-footer-bottom {
    background: #fff;
    border-top: 1px solid #e8e8e4;
    padding: 22px 60px;
    font-size: 0.76rem;
    color: #a0b0aa;
    font-family: 'DM Sans', sans-serif;
}

</style>
""", unsafe_allow_html=True)

brand_col, db_col, sup_col, _ = st.columns([4, 2, 2, 2], vertical_alignment="top", width="stretch")

with brand_col:
    st.markdown(f"""
    <div class="aim-footer-brand">{logo_html} AIM Composites</div>
    <p class="aim-footer-desc">Advancing composites research through open data and collaborative tools.
    A joint initiative of Clemson University and University of Delaware.</p>
    """, unsafe_allow_html=True)

with sup_col:
    st.markdown('<p class="footer-nav-head">DATABASE</p>', unsafe_allow_html=True)
    st.page_link(
        "page_files/Categorized_Search.py",
        label="Browse Materials",
    )
    st.page_link(
        "page_files/Categorized_Search.py",
        label="Categorized Search",
    )
    st.page_link(
        "page_files/Upload_Data.py",
        label="Upload Data",
    )

with _:
    st.markdown('<p class="footer-nav-head">SUPPORT</p>', unsafe_allow_html=True)
    st.page_link(
        "page_files/Contact_Team.py",
        label="Contact Team",
    )



