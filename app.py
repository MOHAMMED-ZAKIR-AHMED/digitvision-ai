import json
import random

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# =====================================================================
#  EDIT YOUR DETAILS HERE (only these 5 lines)
# =====================================================================
YOUR_NAME = "Zakir Ahmed"
TAGLINE = "Data Science / ML Engineer"
LINKEDIN_URL = "https://www.linkedin.com/in/zakir00"
GITHUB_URL = "https://github.com/MOHAMMED-ZAKIR-AHMED"
PORTFOLIO_URL = "https://zenodev-portfolio.vercel.app/"
# =====================================================================

st.set_page_config(page_title="DigitVision AI", page_icon="🧠", layout="wide")

# ----------------------------- STYLING -------------------------------
CSS = """
<style>
.stApp {background: radial-gradient(1200px 600px at 10% -10%, #1d2650 0%, #0B1020 55%);}
.block-container {padding-top: 1.5rem; max-width: 1200px;}
.hero {padding: 26px 30px; border-radius: 22px; margin-bottom: 18px;
  background: linear-gradient(135deg, rgba(139,92,246,.25), rgba(6,182,212,.18));
  border: 1px solid rgba(255,255,255,.12);}
.hero-title {font-size: 2.6rem; font-weight: 800; letter-spacing: -1px;}
.hero-title span {background: linear-gradient(90deg,#A78BFA,#22D3EE);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.hero-sub {opacity: .85; margin-top: 4px; font-size: 1.05rem;}
.byline {margin-top: 14px; font-size: 1rem;}
.badge {display: inline-block; padding: 6px 14px; margin: 8px 8px 0 0; border-radius: 999px;
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.18);
  color: #fff !important; text-decoration: none !important; font-size: .9rem; transition: .2s;}
.badge:hover {background: linear-gradient(90deg,#8B5CF6,#06B6D4); transform: translateY(-2px);}
.section {font-size: 1.15rem; font-weight: 700; margin: 4px 0 8px 0;}
.glass {padding: 18px; border-radius: 18px; background: rgba(255,255,255,.05);
  border: 1px solid rgba(255,255,255,.10); margin-bottom: 12px;}
.digit-card {text-align: center; padding: 14px; border-radius: 20px;
  background: linear-gradient(135deg,#7C3AED,#06B6D4); box-shadow: 0 10px 30px rgba(124,58,237,.35);}
.digit-card .lbl {opacity: .9; font-size: .9rem; letter-spacing: 2px;}
.digit-card .num {font-size: 5.5rem; font-weight: 800; line-height: 1;}
.bar-wrap {margin: 8px 0;}
.bar-lbl {display: flex; justify-content: space-between; font-size: .9rem; margin-bottom: 3px;}
.bar-bg {height: 12px; border-radius: 99px; background: rgba(255,255,255,.1); overflow: hidden;}
.bar-fg {height: 100%; border-radius: 99px;}
.pipe {display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px; margin: 10px 0 4px 0;}
.pbox {flex: 1 1 130px; padding: 14px 10px; text-align: center; border-radius: 16px;
  background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.14);}
.pbox .ic {font-size: 1.8rem;}
.pbox .tt {font-weight: 700; margin-top: 4px;}
.pbox .ds {font-size: .78rem; opacity: .75; margin-top: 2px;}
.parrow {align-self: center; font-size: 1.4rem; opacity: .6;}
.lbox {flex: 1 1 110px; padding: 12px 8px; text-align: center; border-radius: 14px; color: #fff;}
.lbox .nm {font-weight: 700; font-size: .9rem;}
.lbox .sh {font-size: .78rem; opacity: .9;}
.target {text-align: center; font-size: 1.4rem; margin: 6px 0 10px 0;}
.target span {font-size: 3.2rem; font-weight: 800; padding: 0 16px; border-radius: 16px;
  background: linear-gradient(135deg,#F59E0B,#EF4444); margin-left: 8px;}
.chip {display: inline-block; padding: 4px 12px; margin: 3px; border-radius: 99px;
  background: rgba(139,92,246,.25); border: 1px solid rgba(139,92,246,.5); font-size: .85rem;}
.footer {text-align: center; opacity: .8; margin-top: 30px; padding: 18px; font-size: .95rem;
  border-top: 1px solid rgba(255,255,255,.1);}
#MainMenu, footer {visibility: hidden;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------- MODEL ---------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn.keras")


@st.cache_data
def load_accuracy():
    try:
        with open("metrics.json") as f:
            return json.load(f)["test_accuracy"]
    except Exception:
        return None


@st.cache_resource
def get_extractor():
    """Returns a model that outputs the feature maps of each Conv layer."""
    m = load_model()
    convs = [l for l in m.layers if isinstance(l, tf.keras.layers.Conv2D)]
    return tf.keras.Model(inputs=m.inputs, outputs=[l.output for l in convs])


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model. Make sure 'mnist_cnn.keras' is in the same folder as app.py.\n\n{e}")
    st.stop()

model_acc = load_accuracy()


# --------------------------- PREPROCESSING ---------------------------
def preprocess(rgb_img):
    """Canvas drawing -> MNIST style 28x28 (crop, fit in 20x20, center of mass)."""
    arr = np.array(Image.fromarray(rgb_img.astype("uint8")).convert("L"))
    if arr.max() == 0:
        return None
    rows, cols = np.any(arr > 20, axis=1), np.any(arr > 20, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    digit = arr[rmin:rmax + 1, cmin:cmax + 1]

    h, w = digit.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    digit = Image.fromarray(digit).resize((nw, nh), Image.LANCZOS)

    final = Image.new("L", (28, 28), 0)
    final.paste(digit, ((28 - nw) // 2, (28 - nh) // 2))
    a = np.array(final).astype("float32")

    # Move the center of mass to the center of the 28x28 grid (like MNIST)
    total = a.sum()
    if total > 0:
        cy = (a.sum(axis=1) * np.arange(28)).sum() / total
        cx = (a.sum(axis=0) * np.arange(28)).sum() / total
        a = np.roll(a, (int(round(13.5 - cy)), int(round(13.5 - cx))), axis=(0, 1))

    final = Image.fromarray(a.astype("uint8"))
    x = (a / 255.0).reshape(1, 28, 28, 1)
    return x, final


def predict(x):
    return model(x, training=False).numpy()[0]


def color_for(pct):
    return "#22C55E" if pct >= 90 else "#F59E0B" if pct >= 60 else "#EF4444"


def bar(label, pct, color):
    return (
        f'<div class="bar-wrap"><div class="bar-lbl"><span>{label}</span><b>{pct:.2f}%</b></div>'
        f'<div class="bar-bg"><div class="bar-fg" style="width:{pct:.1f}%;background:{color}"></div></div></div>'
    )


def feature_grid(fmap, n=16, cols=8, scale=3):
    fmap = fmap[0]
    n = min(n, fmap.shape[-1])
    tiles = []
    for i in range(n):
        m = fmap[:, :, i]
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        rgb = np.stack([m * 0.35, m * 0.85, m], axis=-1)
        rgb = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), constant_values=0.12)
        tiles.append(rgb)
    rows = [np.concatenate(tiles[r * cols:(r + 1) * cols], axis=1) for r in range(n // cols)]
    img = Image.fromarray((np.concatenate(rows, axis=0) * 255).astype("uint8"))
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def socials_html():
    return (
        f'<a class="badge" href="{LINKEDIN_URL}" target="_blank">💼 LinkedIn</a>'
        f'<a class="badge" href="{GITHUB_URL}" target="_blank">🐙 GitHub</a>'
        f'<a class="badge" href="{PORTFOLIO_URL}" target="_blank">🌐 Portfolio</a>'
    )


# --------------------------- SESSION STATE ---------------------------
defaults = {
    "main_key": 0, "ch_key": 0, "history": [], "last": None,
    "target": random.randint(0, 9), "score": 0, "tries": 0,
    "streak": 0, "best": 0, "msg": None, "celebrate": False,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ------------------------------ SIDEBAR ------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    stroke = st.slider("🖊️ Pen thickness", 10, 30, 18)
    threshold = st.slider("🎯 'Not sure' threshold (%)", 30, 95, 60)
    show_small = st.toggle("Show the model's 28×28 view", value=True)
    show_maps = st.toggle("Show 'What the CNN sees'", value=True)
    st.markdown("---")
    st.markdown("### 🕘 Prediction history")
    if st.session_state.history:
        st.markdown(
            "".join(f'<span class="chip">{d} · {c:.0f}%</span>' for d, c in st.session_state.history[-12:]),
            unsafe_allow_html=True,
        )
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("Nothing here yet. Click 📌 Save prediction after a prediction.")
    st.markdown("---")
    st.markdown(f"**Built by {YOUR_NAME}**")
    st.markdown(socials_html(), unsafe_allow_html=True)

# ------------------------------- HERO --------------------------------
st.markdown(
    f"""
<div class="hero">
<div class="hero-title">🧠 Digit<span>Vision</span> AI</div>
<div class="hero-sub">Draw any digit from 0 to 9 on the canvas. A Convolutional Neural Network recognizes it in real time and tells you how confident it is.</div>
<div class="byline">🚀 Built with ❤️ by <b>{YOUR_NAME}</b> &nbsp;·&nbsp; {TAGLINE}</div>
<div>{socials_html()}</div>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🎨 Draw & Predict", "🎮 Challenge Mode", "🔬 How it Works"])

# =====================================================================
#  TAB 1: DRAW & PREDICT
# =====================================================================
with tab1:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="section">🎨 Draw here</div>', unsafe_allow_html=True)
        canvas = st_canvas(
            fill_color="black", stroke_width=stroke, stroke_color="white",
            background_color="black", height=300, width=300,
            drawing_mode="freedraw", display_toolbar=False, return_image_data=True,
            key=f"canvas_main_{st.session_state.main_key}",
        )
        res = None
        if canvas.image_data is not None:
            res = preprocess(canvas.image_data[:, :, :3])

        probs = None
        if res is not None:
            x, small = res
            probs = predict(x)
            st.session_state.last = {"x": x, "small": small, "probs": probs}

        b1, b2 = st.columns(2)
        if b1.button("🧹 Clear canvas", width="stretch"):
            st.session_state.main_key += 1
            st.rerun()
        if b2.button("📌 Save prediction", width="stretch"):
            if probs is not None:
                st.session_state.history.append((int(np.argmax(probs)), float(probs.max()) * 100))
                st.toast("Saved to history ✅")
            else:
                st.toast("Draw something first 🙂")
        st.caption("💡 Tip: draw the digit big, with a thick stroke, in the center of the canvas.")

    with c2:
        st.markdown('<div class="section">🔮 Prediction</div>', unsafe_allow_html=True)
        if probs is None:
            st.markdown(
                '<div class="glass" style="text-align:center;padding:60px 10px;">'
                '<div style="font-size:3rem">✍️</div>Draw a digit on the canvas,<br>and the prediction will appear here live.</div>',
                unsafe_allow_html=True,
            )
        else:
            pred = int(np.argmax(probs))
            conf = float(probs[pred]) * 100
            st.markdown(
                f'<div class="digit-card"><div class="lbl">PREDICTED DIGIT</div><div class="num">{pred}</div></div>',
                unsafe_allow_html=True,
            )
            if conf >= 90:
                st.success("🔥 The model is very sure about this one!")
            elif conf >= threshold:
                st.info("🙂 The model is fairly confident.")
            else:
                st.warning("🤔 The model is confused. Try drawing a bigger, clearer digit.")

            m1, m2 = st.columns(2)
            m1.metric("Confidence (this drawing)", f"{conf:.2f}%")
            m2.metric("Model Test Accuracy", f"{model_acc * 100:.2f}%" if model_acc else "N/A")

            st.markdown('<div class="section">🏆 Top 3 guesses</div>', unsafe_allow_html=True)
            top3 = np.argsort(probs)[::-1][:3]
            st.markdown(
                "".join(bar(f"Digit {i}", float(probs[i]) * 100, color_for(float(probs[i]) * 100)) for i in top3),
                unsafe_allow_html=True,
            )

    if probs is not None:
        d1, d2 = st.columns(2, gap="large")
        with d1:
            st.markdown('<div class="section">📊 Probability of every digit</div>', unsafe_allow_html=True)
            df = pd.DataFrame({"Digit": [str(i) for i in range(10)], "Probability (%)": probs * 100})
            st.bar_chart(df.set_index("Digit"), height=260)
        with d2:
            if show_small:
                st.markdown('<div class="section">👁️ The 28×28 image the model received</div>', unsafe_allow_html=True)
                st.image(small.resize((196, 196), Image.NEAREST))
                st.caption("Your drawing is cropped, resized and centered to match the MNIST format.")

        if show_maps:
            st.markdown('<div class="section">🧩 What the CNN sees (feature maps)</div>', unsafe_allow_html=True)
            try:
                maps = get_extractor()(st.session_state.last["x"])
                f1, f2 = st.columns(2)
                f1.image(feature_grid(maps[0].numpy(), n=16, cols=8, scale=3), width="stretch")
                f1.caption("Conv Layer 1: detects edges and simple lines.")
                f2.image(feature_grid(maps[1].numpy(), n=16, cols=8, scale=5), width="stretch")
                f2.caption("Conv Layer 2: detects curves and more complex shapes.")
            except Exception as e:
                st.caption(f"Feature maps are not available in this environment ({e}).")

# =====================================================================
#  TAB 2: CHALLENGE MODE
# =====================================================================
with tab2:
    if st.session_state.celebrate:
        st.balloons()
        st.session_state.celebrate = False

    st.markdown(
        "**How to play:** draw the digit shown on screen. If the model recognizes it correctly, you earn a point. "
        "See how long a streak you can build! 🔥"
    )
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("🏅 Score", st.session_state.score)
    s2.metric("🔥 Streak", st.session_state.streak)
    s3.metric("👑 Best streak", st.session_state.best)
    rate = (st.session_state.score / st.session_state.tries * 100) if st.session_state.tries else 0
    s4.metric("🎯 Success rate", f"{rate:.0f}%")

    g1, g2 = st.columns([1, 1], gap="large")
    with g1:
        st.markdown(
            f'<div class="target">Draw this digit: <span>{st.session_state.target}</span></div>',
            unsafe_allow_html=True,
        )
        canvas2 = st_canvas(
            fill_color="black", stroke_width=stroke, stroke_color="white",
            background_color="black", height=300, width=300,
            drawing_mode="freedraw", display_toolbar=False, return_image_data=True,
            key=f"canvas_ch_{st.session_state.ch_key}",
        )
    with g2:
        st.markdown('<div class="section">🎬 Controls</div>', unsafe_allow_html=True)
        check = st.button("✅ Check my drawing", width="stretch", type="primary")
        skip = st.button("⏭️ Skip digit", width="stretch")
        reset = st.button("🔄 Reset game", width="stretch")

        if st.session_state.msg:
            kind, text = st.session_state.msg
            (st.success if kind == "ok" else st.error)(text)

    if check:
        r2 = preprocess(canvas2.image_data[:, :, :3]) if canvas2.image_data is not None else None
        if r2 is None:
            st.warning("Draw something first 🙂")
        else:
            p2 = predict(r2[0])
            got, c = int(np.argmax(p2)), float(p2.max()) * 100
            st.session_state.tries += 1
            if got == st.session_state.target:
                st.session_state.score += 1
                st.session_state.streak += 1
                st.session_state.best = max(st.session_state.best, st.session_state.streak)
                st.session_state.msg = ("ok", f"🎉 Correct! The model recognized {got} ({c:.1f}% confidence).")
                st.session_state.target = random.randint(0, 9)
                st.session_state.celebrate = True
            else:
                st.session_state.streak = 0
                st.session_state.msg = ("err", f"😅 The model read it as {got} ({c:.1f}%). Try again and draw a bit more clearly!")
            st.session_state.ch_key += 1
            st.rerun()
    if skip:
        st.session_state.target = random.randint(0, 9)
        st.session_state.msg = None
        st.session_state.ch_key += 1
        st.rerun()
    if reset:
        for k in ["score", "tries", "streak", "best"]:
            st.session_state[k] = 0
        st.session_state.msg = None
        st.session_state.ch_key += 1
        st.rerun()

# =====================================================================
#  TAB 3: HOW IT WORKS
# =====================================================================
with tab3:
    st.markdown('<div class="section">🛤️ The journey of your drawing (pipeline)</div>', unsafe_allow_html=True)
    steps = [
        ("✍️", "1. Draw", "You draw a digit on the canvas"),
        ("✂️", "2. Crop", "The empty black area around the digit is removed"),
        ("📐", "3. Resize", "The digit is fitted into 20×20 and centered in a 28×28 image"),
        ("🧠", "4. CNN", "Convolution and pooling layers detect patterns"),
        ("📊", "5. Softmax", "Converts the output into 10 probabilities (sum = 100%)"),
        ("🎯", "6. Result", "The digit with the highest probability is the prediction"),
    ]
    html = '<div class="pipe">'
    for i, (ic, t, d) in enumerate(steps):
        html += f'<div class="pbox"><div class="ic">{ic}</div><div class="tt">{t}</div><div class="ds">{d}</div></div>'
        if i < len(steps) - 1:
            html += '<div class="parrow">➜</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    st.markdown('<div class="section" style="margin-top:22px">🏗️ CNN architecture</div>', unsafe_allow_html=True)
    layers = [
        ("Input", "28×28×1", "#475569"),
        ("Conv2D · 64", "26×26×64", "#7C3AED"),
        ("MaxPool", "13×13×64", "#0891B2"),
        ("Conv2D · 32", "11×11×32", "#7C3AED"),
        ("MaxPool", "5×5×32", "#0891B2"),
        ("Flatten", "800", "#B45309"),
        ("Dense · 128", "128", "#BE185D"),
        ("Dense · 32", "32", "#BE185D"),
        ("Output", "10 (softmax)", "#15803D"),
    ]
    html = '<div class="pipe">'
    for i, (n, s, c) in enumerate(layers):
        html += f'<div class="lbox" style="background:{c}"><div class="nm">{n}</div><div class="sh">{s}</div></div>'
        if i < len(layers) - 1:
            html += '<div class="parrow">➜</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    with e1:
        st.markdown(
            """
<div class="glass">
<b>🟣 Conv2D layer</b><br>Small 3×3 filters slide across the whole image and look for edges, lines and curves.
The first layer picks up simple edges, and the second layer combines them into shapes.
</div>
<div class="glass">
<b>🔵 MaxPooling</b><br>Shrinks the image to half its size by keeping the strongest signal in each 2×2 block.
This makes the model faster and more tolerant to small shifts in the digit.
</div>
""",
            unsafe_allow_html=True,
        )
    with e2:
        st.markdown(
            """
<div class="glass">
<b>🟠 Flatten + Dense</b><br>The feature maps are flattened into a single row of 800 numbers. The dense layers combine these features to make the final decision.
</div>
<div class="glass">
<b>🟢 Softmax output</b><br>The last layer outputs 10 numbers (one for each digit from 0 to 9) that add up to 100%. The digit with the highest probability is the prediction.
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section" style="margin-top:10px">🔴 Live demo: how your last drawing looks inside the model</div>', unsafe_allow_html=True)
    last = st.session_state.last
    if last is None:
        st.info("Draw something in the 'Draw & Predict' tab first, then come back here. 🙂")
    else:
        try:
            maps = get_extractor()(last["x"])
            l1, l2, l3 = st.columns([1, 2, 2], gap="medium")
            l1.image(last["small"].resize((140, 140), Image.NEAREST), caption="Input 28×28")
            l2.image(feature_grid(maps[0].numpy(), 16, 8, 3), caption="After Conv 1 (edges)", width="stretch")
            l3.image(feature_grid(maps[1].numpy(), 16, 8, 5), caption="After Conv 2 (shapes)", width="stretch")
            top = int(np.argmax(last["probs"]))
            st.success(f"Final output: digit **{top}** with **{last['probs'][top] * 100:.2f}%** confidence.")
        except Exception as e:
            st.caption(f"Feature maps are not available ({e}).")

    with st.expander("❓ What is the difference between confidence and accuracy?"):
        st.write(
            "- **Confidence (probability)**: how sure the model is about your current drawing only.\n"
            "- **Test accuracy**: the model's overall score on the full MNIST test set (10,000 images). "
            "It stays the same for every drawing."
        )
    with st.expander("⚠️ When does the model make mistakes?"):
        st.write(
            "MNIST contains handwritten digits in a particular style. If your digit is very small, tilted or thin, "
            "or looks like a similar digit (such as 1 and 7), the confidence may drop. Draw it big and clear for the best results."
        )

# ------------------------------- FOOTER ------------------------------
st.markdown(
    f"""
<div class="footer">
Built with ❤️ by <b>{YOUR_NAME}</b> using TensorFlow, Keras and Streamlit<br>
{socials_html()}
</div>
""",
    unsafe_allow_html=True,
)