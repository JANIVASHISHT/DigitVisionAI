
from pathlib import Path
from urllib.parse import quote
import time

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# =========================================================
# DIGITVISION AI — V.R PROGRAMMER
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "training_runs" / "digitvision_emnist_candidate.keras"
LOGO_PATH = BASE_DIR / "vr_programmer_favicon.png"

SUPPORT_EMAIL = "vashishtvrjani@gmail.com"


def gmail_compose_url(subject_text, body_text):
    """Build a Gmail web-compose link so no desktop email app is required."""
    subject = quote(str(subject_text), safe="")
    body = quote(str(body_text), safe="")
    recipient = quote(SUPPORT_EMAIL, safe="@")
    return (
        "https://mail.google.com/mail/?view=cm&fs=1"
        f"&to={recipient}&su={subject}&body={body}"
    )

st.set_page_config(
    page_title="DigitVision AI | V.R PROGRAMMER",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #172b55 0%, transparent 36%),
        radial-gradient(ellipse at 90% 15%, #30164c 0%, transparent 30%),
        #080d19;
    color: #eef3ff;
}

.block-container {
    padding-top: 4.25rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1450px;
    overflow: visible !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101a30, #0b1020);
    border-right: 1px solid #263653;
}

h1, h2, h3 {
    color: #f4f7ff !important;
}

p, li, label {
    color: #c4cee2;
}

.hero {
    padding: 32px;
    border-radius: 24px;
    background: linear-gradient(120deg, #13264b, #20143e 75%, #13223a);
    border: 1px solid #334a77;
    margin-bottom: 22px;
    box-shadow: 0 15px 45px #00000035;
}

.hero-title {
    font-size: clamp(30px, 4vw, 48px);
    font-weight: 800;
    letter-spacing: -1.5px;
    line-height: 1.12;
    color: white;
}

.gradient-text {
    background: linear-gradient(90deg, #55e6ff, #a99aff, #ff9bda);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    font-size: 16px;
    line-height: 1.8;
    color: #c8d7f3;
    max-width: 760px;
    margin-top: 12px;
}

.panel {
    background: linear-gradient(145deg, #121d32, #101628);
    border: 1px solid #293954;
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 18px;
}

.metric {
    background: linear-gradient(145deg, #172641, #15152e);
    border: 1px solid #344568;
    border-radius: 17px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    color: #aab9d5;
    font-size: 13px;
    margin-bottom: 9px;
}

.metric-value {
    font-size: 27px;
    font-weight: 800;
    color: #f4f7ff;
}

.small-note {
    color: #9cacc9;
    font-size: 12px;
    line-height: 1.7;
}

.section-tag {
    color: #61e6ff;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.stButton {
    margin-top: 0.35rem !important;
    margin-bottom: 0.35rem !important;
    min-width: 0 !important;
}

.stButton > button {
    border-radius: 12px;
    border: 1px solid #3d5278;
    background: linear-gradient(100deg, #1c3b67, #35245f);
    color: white;
    font-weight: 650;
    min-height: 48px !important;
    height: auto !important;
    line-height: 1.3 !important;
    padding: 0.65rem 0.7rem !important;
    white-space: normal !important;
    overflow: visible !important;
    transition: all .2s ease;
}

.stButton > button:hover {
    border-color: #63e5ff;
    color: #ffffff;
    transform: translateY(-1px);
    box-shadow: 0 5px 22px #36cfff22;
}

.stDownloadButton > button {
    border-radius: 12px;
}

div[data-testid="stFileUploader"] {
    background: #10192a;
    border: 1px dashed #52678d;
    border-radius: 15px;
    padding: 10px;
}

div[data-testid="stImage"] img {
    border-radius: 14px;
}

hr {
    border-color: #263653;
}

/* Branded one-time startup splash */
.dv-splash {
    position: fixed;
    inset: 0;
    z-index: 999999;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    background:
        radial-gradient(circle at 20% 20%, #173b70 0%, transparent 36%),
        radial-gradient(circle at 80% 75%, #492065 0%, transparent 34%),
        #080d19;
    color: #f4f7ff;
}
.dv-splash-card {
    width: min(88vw, 560px);
    padding: 38px 28px;
    border: 1px solid #3b5584;
    border-radius: 28px;
    background: linear-gradient(145deg, #111d34ee, #1b1435ee);
    box-shadow: 0 25px 100px #0009;
}
.dv-orbit {
    width: 78px;
    height: 78px;
    margin: 0 auto 22px;
    border: 4px solid #263b62;
    border-top-color: #55e6ff;
    border-right-color: #b09cff;
    border-radius: 50%;
    animation: dv-spin 1s linear infinite;
    position: relative;
}
.dv-orbit:after {
    content: "7";
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    color: #fff;
    font-size: 30px;
    font-weight: 800;
    animation: dv-counterspin 1s linear infinite;
}
.dv-splash-title {
    font-size: clamp(28px, 5vw, 42px);
    font-weight: 800;
    letter-spacing: -1px;
}
.dv-splash-sub {
    margin-top: 8px;
    color: #b9c9e8;
}
.dv-loading-track {
    height: 7px;
    overflow: hidden;
    margin: 26px 0 12px;
    background: #243451;
    border-radius: 20px;
}
.dv-loading-bar {
    height: 100%;
    width: 40%;
    border-radius: 20px;
    background: linear-gradient(90deg, #55e6ff, #a99aff, #ff9bda);
    animation: dv-load 1.5s ease-in-out infinite;
}
.dv-loading-caption {
    color: #8ea5ce;
    font-size: 12px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
@keyframes dv-spin { to { transform: rotate(360deg); } }
@keyframes dv-counterspin { to { transform: rotate(-360deg); } }
@keyframes dv-load {
    0% { transform: translateX(-120%); }
    100% { transform: translateX(300%); }
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# ONE-TIME BRANDED STARTUP SCREEN
# =========================================================
# This splash is shown once per browser session. The short pause gives
# the CSS animation time to be visible; the AI model loads on its first use.
if not st.session_state.get("startup_splash_done", False):
    st.markdown("""
    <div class="dv-splash">
        <div class="dv-splash-card">
            <div class="dv-orbit"></div>
            <div class="dv-splash-title">
                DigitVision <span class="gradient-text">AI</span>
            </div>
            <div class="dv-splash-sub">A V.R PROGRAMMER project</div>
            <div class="dv-loading-track"><div class="dv-loading-bar"></div></div>
            <div class="dv-loading-caption">Preparing your AI workspace · Please wait</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(2.2)
    st.session_state["startup_splash_done"] = True
    st.rerun()


# =========================================================
# STATE AND NAVIGATION
# =========================================================

PAGES = [
    "Home",
    "Recognize Digit",
    "How It Works",
    "About the Team",
    "Report a Bug",
]

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_filename" not in st.session_state:
    st.session_state.last_filename = None


def navigate(page):
    st.session_state.page = page


# =========================================================
# SIDEBAR BRANDING AND NAVIGATION
# =========================================================

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=115)
    else:
        st.markdown("## 🔢")

    st.markdown("### V.R PROGRAMMER")
    st.markdown("**DigitVision AI**")
    st.caption("Handwritten digit recognition")

    st.divider()
    st.markdown("#### NAVIGATION")

    for page in PAGES:
        label = {
            "Home": "🏠  Home",
            "Recognize Digit": "🔍  Recognize Digit",
            "How It Works": "⚙️  How It Works",
            "About the Team": "👥  About the Team",
            "Report a Bug": "🐞  Report a Bug",
        }[page]

        if st.button(
            label,
            key=f"nav_{page}",
            use_container_width=True,
            type="primary" if st.session_state.page == page else "secondary",
        ):
            navigate(page)
            st.rerun()

    st.divider()
    st.markdown("#### PROJECT STATUS")
    st.success("Interface ready")

    st.markdown(
        '<p class="small-note">Built with Python, Streamlit, '
        'OpenCV and TensorFlow/Keras.</p>',
        unsafe_allow_html=True,
    )


# Top navigation buttons: extra spacing prevents clipping under Streamlit's toolbar.
st.markdown('<div style="height:0.35rem"></div>', unsafe_allow_html=True)
nav_cols = st.columns(len(PAGES), gap="small")

for i, page in enumerate(PAGES):
    with nav_cols[i]:
        if st.button(
            page,
            key=f"topnav_{page}",
            use_container_width=True,
        ):
            navigate(page)
            st.rerun()


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Check that the configured model file exists."
        )

    return tf.keras.models.load_model(str(MODEL_PATH))


def get_model():
    try:
        with st.spinner("Loading the DigitVision AI model..."):
            time.sleep(0.25)
            return load_model()
    except Exception as exc:
        st.error("The AI model could not be loaded.")
        st.code(str(exc))
        st.info(
            "Check that the configured model file exists and "
            "that your virtual environment has TensorFlow installed."
        )
        return None


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def correct_shadow(gray):
    """
    Reduce broad, uneven lighting and shadows while preserving digit strokes.
    Input and output are 8-bit grayscale images.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("correct_shadow expects an 8-bit grayscale image")

    # Estimate slowly changing background illumination.
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=21, sigmaY=21)

    # Divide by the estimated illumination to reduce broad shadows.
    background = np.maximum(background, 16).astype(np.float32)
    corrected = gray.astype(np.float32) / background * 200.0

    return np.clip(corrected, 0, 255).astype(np.uint8)


def remove_notebook_lines(binary):
    """
    Input: white foreground on a black background.
    Removes long horizontal lines while attempting to preserve digits.
    """
    # A longer kernel helps avoid deleting the crossbar of a handwritten 7.
    horizontal_length = max(25, binary.shape[1] // 3)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_length, 1)
    )

    horizontal_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel
    )

    cleaned = cv2.subtract(binary, horizontal_lines)
    return cleaned


def prepare_candidate(binary, method_name):
    """
    Converts a binary digit into a centered 28x28 MNIST-style image.
    Foreground should be white; background should be black.
    """
    binary = (binary > 0).astype(np.uint8) * 255

    # Remove tiny isolated noise.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    cleaned = np.zeros_like(binary)

    for i in range(1, count):
        x, y, w, h, area = stats[i]

        if area >= 3:
            cleaned[labels == i] = 255

    points = cv2.findNonZero(cleaned)

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)

    # A very small crop is usually noise rather than a usable digit.
    if w < 3 or h < 5:
        return None

    digit = cleaned[y:y + h, x:x + w]

    # Resize proportionally into a 20x20 area.
    scale = min(20 / w, 20 / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)

    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2

    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized

    # Slight blur can reduce jagged edges from phone photos.
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    return {
        "name": method_name,
        "image": canvas,
        "crop": digit,
    }


def create_candidates(pil_image, ink_mode="Auto-detect"):
    """
    Build multiple MNIST-style candidates for a single handwritten digit.
    Auto mode tests common pen colours plus grayscale threshold methods.
    Keep original candidates as well as line-cleaned variants because
    aggressive line removal can damage the crossbar of a handwritten 7.
    """
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    candidates = []

    def add_candidate(mask, name, remove_lines=False):
        if mask is None:
            return
        binary = (mask > 0).astype(np.uint8) * 255
        if remove_lines:
            binary = remove_notebook_lines(binary)
        prepared = prepare_candidate(binary, name)
        if prepared is None:
            return
        for existing in candidates:
            if np.array_equal(existing["image"], prepared["image"]):
                return
        candidates.append(prepared)

    # HSV masks: the hue ranges help separate coloured ink from the paper.
    blue_mask = cv2.inRange(
        hsv, np.array([85, 35, 25]), np.array([145, 255, 255])
    )
    red_low = cv2.inRange(
        hsv, np.array([0, 45, 35]), np.array([12, 255, 255])
    )
    red_high = cv2.inRange(
        hsv, np.array([168, 45, 35]), np.array([179, 255, 255])
    )
    red_mask = cv2.bitwise_or(red_low, red_high)
    green_mask = cv2.inRange(
        hsv, np.array([35, 30, 25]), np.array([90, 255, 255])
    )
    dark_mask = cv2.inRange(enhanced, 0, 150)

    masks_by_mode = {
        "Blue ink": [(blue_mask, "Blue ink")],
        "Black ink": [(dark_mask, "Black/dark ink")],
        "Red ink": [(red_mask, "Red ink")],
        "Green ink": [(green_mask, "Green ink")],
        "Other / dark ink": [(dark_mask, "Other/dark ink")],
    }

    if ink_mode in masks_by_mode:
        for mask, name in masks_by_mode[ink_mode]:
            add_candidate(mask, f"{name} original", False)
            add_candidate(mask, f"{name} line-cleaned", True)
    else:
        # Original candidates come first so an actual 7 crossbar is retained.
        add_candidate(blue_mask, "Blue ink original", False)
        add_candidate(red_mask, "Red ink original", False)
        add_candidate(green_mask, "Green ink original", False)
        add_candidate(dark_mask, "Dark ink original", False)

        # Also try line-cleaned colour candidates, but don't rely on them alone.
        add_candidate(blue_mask, "Blue ink line-cleaned", True)
        add_candidate(red_mask, "Red ink line-cleaned", True)
        add_candidate(green_mask, "Green ink line-cleaned", True)
        add_candidate(dark_mask, "Dark ink line-cleaned", True)

        _, otsu = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        add_candidate(otsu, "Otsu threshold original", False)
        add_candidate(remove_notebook_lines(otsu), "Otsu threshold line-cleaned", False)

        # Keep the original adaptive candidates as a baseline.
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 9
        )
        add_candidate(adaptive, "Adaptive threshold original", False)
        add_candidate(
            remove_notebook_lines(adaptive),
            "Adaptive threshold line-cleaned",
            False,
        )

        # Additional candidates after reducing uneven background shadows.
        shadow_corrected = correct_shadow(gray)
        shadow_enhanced = clahe.apply(shadow_corrected)

        for block_size, offset in [(31, 7), (31, 11), (41, 9)]:
            shadow_adaptive = cv2.adaptiveThreshold(
                shadow_enhanced, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block_size, offset
            )
            label = f"Shadow-corrected adaptive {block_size}/{offset}"
            add_candidate(shadow_adaptive, label, False)
            add_candidate(
                remove_notebook_lines(shadow_adaptive),
                label + " line-cleaned",
                False,
            )

    return candidates



def predict_candidates(model, candidates):
    """Run all candidate images through the model in one batch."""
    if not candidates:
        return []

    batch = np.stack(
        [candidate["image"].astype(np.float32) / 255.0
         for candidate in candidates],
        axis=0,
    )

    if len(model.input_shape) == 4:
        batch = batch[..., np.newaxis]

    probabilities_batch = model(batch, training=False).numpy()

    results = []
    for candidate, probabilities in zip(candidates, probabilities_batch):
        results.append({
            "name": candidate["name"],
            "image": candidate["image"],
            "probabilities": probabilities,
            "digit": int(np.argmax(probabilities)),
            "confidence": float(np.max(probabilities)),
        })

    results.sort(key=lambda item: item["confidence"], reverse=True)
    return results
def show_header():
    st.markdown("""
    <div class="hero">
        <div class="section-tag">ARTIFICIAL INTELLIGENCE · COMPUTER VISION</div>
        <div class="hero-title">
            DigitVision <span class="gradient-text">AI</span>
        </div>
        <div class="hero-sub">
            Turn handwritten numbers into digital predictions.
            Upload a digit image, explore different preprocessing methods,
            and see how a neural network interprets your handwriting.
        </div>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label, value, note):
    st.markdown(f"""
    <div class="metric">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="small-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def show_footer():
    st.divider()
    st.markdown(
        '<div style="text-align:center;color:#8494b3;font-size:12px;'
        'line-height:1.9;">'
        '<b>V.R PROGRAMMER</b> · DigitVision AI<br>'
        '© 2026 V.R PROGRAMMER and the DigitVision AI project team. '
        'All rights reserved to the respective creators.'
        '</div>',
        unsafe_allow_html=True,
    )


# =========================================================
# HOME PAGE
# =========================================================

if st.session_state.page == "Home":
    show_header()

    st.markdown("### Welcome to your AI workspace")
    st.write(
        "DigitVision AI is a handwriting-recognition prototype. "
        "Start by uploading a clear photo of a single handwritten digit."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("01 · INPUT", "Your image", "Upload a handwritten number.")

    with c2:
        metric_card("02 · PROCESS", "OpenCV", "Clean, crop and center the digit.")

    with c3:
        metric_card("03 · PREDICT", "TensorFlow", "Compare model predictions.")

    st.markdown("")
    left, right = st.columns([1.2, 0.8], gap="large")

    with left:
        st.markdown("""
        <div class="panel">
            <div class="section-tag">GET STARTED</div>
            <h3>Recognize your first digit</h3>
            <p>
                Use a photo of one digit at a time. The digit should be
                clearly visible, with limited shadows and enough space
                around the writing.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "🔍  Open Digit Recognizer",
            use_container_width=True,
            type="primary",
        ):
            navigate("Recognize Digit")
            st.rerun()

    with right:
        st.markdown("""
        <div class="panel">
            <div class="section-tag">FEATURES</div>
            <h3>Explore the project</h3>
            <p>✓ Multiple image preprocessing methods</p>
            <p>✓ Blue and black pen options</p>
            <p>✓ Prediction probability chart</p>
            <p>✓ Bug report and feedback form</p>
            <p>✓ Step-by-step explanation</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Tip: For best results, crop the photo around one large digit. "
        "Notebook lines, shadows and very small writing can affect accuracy."
    )


# =========================================================
# RECOGNITION PAGE
# =========================================================

elif st.session_state.page == "Recognize Digit":
    show_header()

    st.markdown("### 🔍 Digit recognition studio")
    st.write(
        "Upload one handwritten digit. Auto-detect tries several "
        "preprocessing methods, which can help when a digit is written "
        "in a notebook."
    )

    model = get_model()

    if model is not None:
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### 1. Upload your image")

            ink_mode = st.selectbox(
                "Pen / ink mode",
                [
                    "Auto-detect", "Blue ink", "Black ink",
                    "Red ink", "Green ink", "Other / dark ink"
                ],
                help="Auto-detect tests several ink colours and threshold methods. "
                     "For coloured pen, choose its colour if automatic detection is poor.",
            )

            uploaded_file = st.file_uploader(
                "Upload PNG, JPG or JPEG",
                type=["png", "jpg", "jpeg"],
                help="Use a clear image containing one large digit.",
            )

            st.markdown(
                '<p class="small-note">Recommended: one digit, good light, '
                'minimal shadow, and the digit filling most of the photo.</p>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### 2. Original image")

            if uploaded_file is not None:
                pil_image = Image.open(uploaded_file).convert("RGB")
                st.image(
                    pil_image,
                    caption="Original uploaded image",
                    use_container_width=True,
                )
            else:
                st.markdown(
                    '<div style="padding:55px 15px;text-align:center;'
                    'border:1px dashed #465a7c;border-radius:15px;'
                    'color:#9aaccb;">'
                    '<div style="font-size:42px;">🔢</div>'
                    '<b>Your image preview will appear here</b><br>'
                    '<span class="small-note">Upload a digit to begin</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            if st.button(
                "✨  Recognize Digit",
                use_container_width=True,
                type="primary",
            ):
                try:
                    with st.spinner(
                        "Cleaning your image and testing AI predictions..."
                    ):
                        candidates = create_candidates(pil_image, ink_mode)

                        if not candidates:
                            st.session_state.last_result = None
                            st.error(
                                "No clear digit was detected. Try a closer, "
                                "brighter photo with less background."
                            )
                        else:
                            results = predict_candidates(model, candidates)

                            st.session_state.last_result = results
                            st.session_state.last_filename = uploaded_file.name
                            st.session_state.last_ink_mode = ink_mode
                            st.session_state.last_original = pil_image.copy()

                except Exception as exc:
                    st.error("Recognition failed.")
                    st.code(str(exc))

        results = st.session_state.last_result

        if results:
            st.divider()
            st.markdown("### 🎯 Recognition results")

            # Let the user compare preprocessing results.
            option_names = [
                f"{r['name']} — predicts {r['digit']} "
                f"({r['confidence']:.1%})"
                for r in results
            ]

            selected_index = st.selectbox(
                "Compare preprocessing methods",
                options=list(range(len(results))),
                format_func=lambda i: option_names[i],
                help="If the top result is wrong, try another method here.",
            )

            selected = results[selected_index]

            result_left, result_right = st.columns([0.9, 1.1], gap="large")

            with result_left:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown("#### Predicted digit")
                st.markdown(
                    f'<div style="font-size:100px;font-weight:800;'
                    f'text-align:center;line-height:1.2;'
                    f'background:linear-gradient(90deg,#55e6ff,#b09cff);'
                    f'-webkit-background-clip:text;'
                    f'-webkit-text-fill-color:transparent;">'
                    f'{selected["digit"]}</div>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div style="text-align:center;color:#c6d4ed;">'
                    f'Confidence score: '
                    f'<b>{selected["confidence"]:.2%}</b></div>',
                    unsafe_allow_html=True,
                )

                st.progress(
                    max(0.0, min(1.0, selected["confidence"]))
                )

                st.caption(
                    f"Preprocessing method: {selected['name']}"
                )
                st.image(
                    selected["image"],
                    caption="Processed 28 × 28 model input",
                    width=220,
                    clamp=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with result_right:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown("#### Prediction probabilities")

                probabilities = selected["probabilities"]
                chart_data = {
                    "Digit": [str(i) for i in range(10)],
                    "Probability": [
                        float(p) for p in probabilities
                    ],
                }

                st.bar_chart(
                    chart_data,
                    x="Digit",
                    y="Probability",
                    use_container_width=True,
                )

                st.markdown(
                    '<p class="small-note">These values show how strongly '
                    'the model favours each digit. High confidence does not '
                    'always mean the prediction is correct.</p>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # Show agreement between preprocessing methods.
            digit_counts = {
                digit: sum(1 for r in results if r["digit"] == digit)
                for digit in range(10)
            }
            vote_digit = max(digit_counts, key=digit_counts.get)
            vote_count = digit_counts[vote_digit]

            st.info(
                f"🧠 Method agreement: {vote_count}/{len(results)} methods "
                f"predict digit {vote_digit}. This indicates consistency, "
                "not guaranteed correctness."
            )

            st.markdown("#### 🧪 Compare all preprocessing methods")

            comparison = []
            for r in results:
                comparison.append({
                    "Method": r["name"],
                    "Predicted digit": r["digit"],
                    "Confidence": f"{r['confidence']:.2%}",
                })

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )

            st.warning(
                "If the uploaded digit is 7 but the model predicts another "
                "number, select another preprocessing method and compare "
                "the 28 × 28 preview. If every method is wrong, the model "
                "may need additional training on real handwritten notebook "
                "digits."
            )

            with st.expander("Report this prediction as incorrect"):
                actual_digit = st.selectbox(
                    "What was the correct digit?",
                    list(range(10)),
                    key="correct_digit",
                )

                if st.button("Prepare correction report"):
                    subject_text = "DigitVision AI - Incorrect Prediction"
                    body_text = (
                        f"DigitVision AI correction report\n\n"
                        f"Image: {st.session_state.last_filename}\n"
                        f"Predicted digit: {selected['digit']}\n"
                        f"Predicted confidence: {selected['confidence']:.2%}\n"
                        f"Preprocessing: {selected['name']}\n"
                        f"Correct digit: {actual_digit}\n\n"
                        "Please investigate this recognition error."
                    )
                    gmail_url = gmail_compose_url(subject_text, body_text)

                    st.link_button(
                        "📧 Open Gmail in your browser",
                        gmail_url,
                        use_container_width=True,
                    )

    show_footer()


# =========================================================
# HOW IT WORKS
# =========================================================

elif st.session_state.page == "How It Works":
    show_header()

    st.markdown("### ⚙️ How DigitVision AI works")
    st.write(
        "The application turns an uploaded photo into a small image "
        "that a trained neural network can process."
    )

    steps = [
        (
            "01",
            "Image upload",
            "The app reads your JPG or PNG image.",
        ),
        (
            "02",
            "Image processing",
            "OpenCV improves contrast, separates the ink from the "
            "background and attempts to remove notebook lines.",
        ),
        (
            "03",
            "Digit isolation",
            "The detected writing is cropped, resized proportionally "
            "and centered on a 28 × 28 canvas.",
        ),
        (
            "04",
            "AI prediction",
            "TensorFlow/Keras passes the processed image to the trained "
            "model, which estimates probabilities for digits 0–9.",
        ),
        (
            "05",
            "Result display",
            "The app shows the predicted digit, confidence and probability "
            "chart. Different preprocessing methods can be compared.",
        ),
    ]

    for number, title, description in steps:
        st.markdown(f"""
        <div class="panel">
            <div style="display:flex;gap:18px;align-items:flex-start;">
                <div style="font-size:26px;font-weight:800;color:#62e5ff;">
                    {number}
                </div>
                <div>
                    <h3 style="margin:0;">{title}</h3>
                    <p style="margin-top:8px;">{description}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Technologies used")

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        metric_card("LANGUAGE", "Python", "Application logic")

    with t2:
        metric_card("VISION", "OpenCV", "Image processing")

    with t3:
        metric_card("AI MODEL", "TensorFlow", "Neural network")

    with t4:
        metric_card("INTERFACE", "Streamlit", "Interactive web app")

    st.info(
        "The model currently recognizes digits 0–9. It is not a general "
        "handwriting OCR system and does not automatically read full words "
        "or sentences."
    )

    show_footer()


# =========================================================
# ABOUT THE TEAM
# =========================================================

elif st.session_state.page == "About the Team":
    show_header()

    st.markdown("### 👥 About the project")
    st.write(
        "DigitVision AI is a student project that explores computer vision, "
        "machine learning and a practical interface for handwritten-digit "
        "recognition."
    )

    st.markdown("""
    <div class="panel">
        <div class="section-tag">PROJECT IDENTITY</div>
        <h2>DigitVision AI</h2>
        <p><b>Brand:</b> V.R PROGRAMMER</p>
        <p><b>Goal:</b> Recognize handwritten digits from uploaded images.</p>
        <p><b>Core technologies:</b> Python, OpenCV, TensorFlow/Keras and Streamlit.</p>
        <p><b>Current input:</b> One handwritten digit per image.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Project capabilities")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="panel">
            <h3>What it can do</h3>
            <p>✓ Accept a digit image</p>
            <p>✓ Process ink and background</p>
            <p>✓ Predict one of ten digits</p>
            <p>✓ Display probability estimates</p>
            <p>✓ Compare preprocessing approaches</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="panel">
            <h3>Current limitations</h3>
            <p>• Handwriting styles vary</p>
            <p>• Notebook lines can affect results</p>
            <p>• Shadows can distort the image</p>
            <p>• A high score can still be incorrect</p>
            <p>• Full text recognition is not included</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Future improvements")
    st.write(
        "Possible next steps include training with real notebook photos, "
        "adding a drawing canvas, collecting correction examples, and "
        "supporting multiple digits."
    )

    show_footer()


# =========================================================
# REPORT A BUG
# =========================================================

elif st.session_state.page == "Report a Bug":
    show_header()

    st.markdown("### 🐞 Report a bug or suggest an improvement")
    st.write(
        "Help improve DigitVision AI by reporting incorrect predictions, "
        "upload problems, display issues or other bugs."
    )

    with st.form("bug_report_form"):
        category = st.selectbox(
            "Issue type",
            [
                "Incorrect digit prediction",
                "Digit not detected",
                "Upload problem",
                "Application error",
                "Interface or navigation issue",
                "Feature suggestion",
                "Other",
            ],
        )

        expected_digit = st.selectbox(
            "Correct digit, if relevant",
            ["Not applicable", "0", "1", "2", "3", "4",
             "5", "6", "7", "8", "9"],
        )

        details = st.text_area(
            "Describe the issue",
            placeholder=(
                "Example: I uploaded a handwritten 7 in blue ink, "
                "but the application predicted 4."
            ),
            height=150,
        )

        contact_name = st.text_input(
            "Your name (optional)",
            placeholder="Enter your name",
        )

        submitted = st.form_submit_button(
            "Prepare Bug Report Email",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        if not details.strip():
            st.error("Please describe the issue before preparing the report.")
        else:
            subject_text = f"DigitVision AI Bug Report: {category}"

            body_text = (
                "DigitVision AI Bug Report\n\n"
                f"Issue type: {category}\n"
                f"Correct digit: {expected_digit}\n"
                f"Name: {contact_name or 'Not provided'}\n\n"
                f"Description:\n{details}\n\n"
                f"Last uploaded image: "
                f"{st.session_state.get('last_filename') or 'Not available'}\n"
                f"App section: {st.session_state.page}\n"
            )

            gmail_url = gmail_compose_url(subject_text, body_text)

            st.success("Your report is ready to open in Gmail in your browser.")
            st.link_button(
                "📧 Open Gmail in your browser",
                gmail_url,
                use_container_width=True,
            )

            st.caption(
                "Sign in to Gmail if prompted, review the prepared message, "
                "and press Send. The app does not send emails automatically "
                "or attach your uploaded photo."
            )

    st.markdown("""
    <div class="panel">
        <div class="section-tag">SUPPORT CONTACT</div>
        <h3>V.R PROGRAMMER</h3>
        <p>Bug reports and feedback</p>
        <p><b>vashishtvrjani@gmail.com</b></p>
    </div>
    """, unsafe_allow_html=True)

    show_footer()
