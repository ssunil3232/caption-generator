import os
import base64
from io import BytesIO
import hashlib

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI
from transformers import BlipProcessor, BlipForConditionalGeneration

load_dotenv()

# Avoid tokenizer parallelism warnings in forked Streamlit runs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

st.set_page_config(layout="wide")


def get_secret(key):
    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError):
        return None


st.markdown(
    """
    <style>
    .main {
        background-color: #F4FFFC;
    }
    h4 {
        color: #FCB929;
    }
    h3 {
        color: #000000;
    }
    .loading-overlay {
        position: fixed;
        inset: 0;
        background: rgba(244, 255, 252, 0.9);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    }
    .loading-spinner {
        width: 64px;
        height: 64px;
        border: 6px solid rgba(12, 133, 124, 0.2);
        border-top-color: rgb(12 133 124);
        border-radius: 50%;
        animation: spin 0.9s linear infinite;
    }
    .loading-text {
        margin-top: 12px;
        font-size: 20px;
        font-weight: 700;
        color: rgb(12 133 124);
        letter-spacing: 0.5px;
    }
    .generated-output {
        color: #000000;
    }
    .output-card {
        background-color: #dbf3f1;
        padding: 12px 16px;
        border-radius: 10px;
        margin: 4px 0 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    div[data-testid="stAlert"][data-testid="stAlert"][role="alert"] {
        background-color: #E50000;
        color: #ffffff;
        border: 1px solid #E50000;
    }
    div[data-testid="stAlert"] * {
        color: #ffffff !important;
    }
    button[title="View fullscreen"]{
        visibility: hidden;
    }
    label[data-testid="stWidgetLabel"] {
        p{
            font-size: 24px;
            font-weight: bold;
            color: #FCB929;
        }
    }
    div.stFileUploader > div:first-child {
    font-size: 24px;
    font-weight: bold;
}
    div[data-testid="stFileUploaderDropzone"] {
        border: 2px dashed rgb(12 133 124);
        background-color: rgb(12 133 124)!important;
    }
    div[data-testid="stFileUploaderDropzone"] > div {
        background-color: rgb(12 133 124)!important;
    }
    div[data-testid="stFileUploaderDropzone"] svg,
    div[data-testid="stFileUploaderDropzone"] svg * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    div[data-testid="stFileUploaderDropzone"] * {
        color: #ffffff !important;
    }
    div[data-testid="stTextArea"] textarea {
        text-align: left;
        padding-left: 12px;
    }
    div.stButton > button {
        background-color: rgb(12 133 124);
        border: none;
        color: #ffffff !important;
    }
    div.stButton > button:hover {
        background-color: rgb(10 112 105);
        border: none;
        color: #ffffff !important;
    }
    div.stButton > button:active,
    div.stButton > button:focus,
    div.stButton > button:focus-visible {
        border: none;
        outline: none;
        box-shadow: none;
        color: #ffffff !important;
    }
    .st-au {
        background-color: #ff4e4e !important;
    }
    .caption-output {
        font-size: 24px;
        font-weight: bold;
        margin: 0 0 6px 0;
    }
    .eval-output {
        font-size: 24px;
        font-weight: bold;
        color: #FCB929;
    }
    .stMarkdown {
        color: #000000;
    }
    #eval-output + div[data-testid="stMarkdownContainer"] {
        color: #000000;
    }
    #eval-output + div[data-testid="stMarkdownContainer"] h1,
    #eval-output + div[data-testid="stMarkdownContainer"] h2,
    #eval-output + div[data-testid="stMarkdownContainer"] h3 {
        font-size: 24px;
        font-weight: 700;
        color: #FCB929;
        margin: 4px 0 4px;
    }
    #eval-output + div[data-testid="stMarkdownContainer"] p,
    #eval-output + div[data-testid="stMarkdownContainer"] li,
    #eval-output + div[data-testid="stMarkdownContainer"] span,
    #eval-output + div[data-testid="stMarkdownContainer"] strong,
    #eval-output + div[data-testid="stMarkdownContainer"] em {
        color: #000000 !important;
    }
    #eval-output + div[data-testid="stMarkdownContainer"] p {
        margin: 4px 0;
        line-height: 1.4;
    }
    #eval-output + div[data-testid="stMarkdownContainer"] ul,
    #eval-output + div[data-testid="stMarkdownContainer"] ol {
        margin: 4px 0 6px 18px;
    }
    #eval-output + div[data-testid="stMarkdownContainer"] li {
        margin: 2px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

loading_overlay = st.empty()


def show_loading_overlay(message="Loading..."):
    loading_overlay.markdown(
        f"""
        <div class="loading-overlay">
            <div class="loading-spinner"></div>
            <div class="loading-text">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hide_loading_overlay():
    loading_overlay.empty()


DEEPSEEK_API_KEY = get_secret("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")


@st.cache_resource(show_spinner=False)
def load_caption_models():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large")
    return processor, model


if not DEEPSEEK_API_KEY:
    st.error("Missing DEEPSEEK_API_KEY. Add it to .env or Streamlit secrets.")
    st.stop()

processor, model = load_caption_models()

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",
)
img_col_left, img_col_center, img_col_right = st.columns([1, 8, 1])
with img_col_center:
    st.image("titlepage.svg", use_column_width=True)
    st.image("secondpage.svg", use_column_width=True)
# File upload handling
uploaded_file = st.file_uploader(
    "Upload your next social media post!", type=['png', 'jpg', 'jpeg'])


def generate_social_media_post(
    image_conditional_caption, image_unconditional_caption, company_info, social_media_posts
):
    prompt = f"""Generate a creative social media post based on the following inputs:\nConditional Caption: {image_conditional_caption}\nUnconditional Caption: {
        image_unconditional_caption} \n Company Information: {company_info} \n Recent Social Media Posts: {social_media_posts} \n Just give the caption that I can post on social media sites please."""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "You are a pro social media marketing caption generator. Create engaging and relevant posts.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def evaluate_social_media_post(image, caption, company_information):
    _ = image  # image not used; DeepSeek Chat is text-only
    prompt = f"""
    Given an image and its caption, evaluate the caption's effectiveness based on the following criteria:

    1. **Engagement Potential**: Consider factors such as the caption's ability to capture attention, provoke thought, or encourage interaction (likes, comments, shares). Assess whether the caption uses language that is likely to engage the target audience, including any use of humor, questions, or call-to-actions.

    2. **Alignment with Company Values**: Examine if the caption accurately reflects the company's values and branding. The company is committed to [describe company values briefly, e.g., sustainability, innovation, customer focus]. Determine if the caption supports these values, either directly through the content or indirectly through tone and approach.

    3. **Rating for the post**: Finally rate the post on a scale of 1 to 10.

    ### Image Description
    Use the conditional and unconditional captions as the image description.

    ### Caption
    {caption}

    ### Company Information
    {company_information}

    Please provide a detailed evaluation of the caption based on the above criteria, highlighting its strengths and suggesting any improvements if necessary.
    """

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {
                "role": "system",
                "content": "You are a pro social media marketing caption evaluator. Evaluate my caption.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content


if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    image_key = hashlib.md5(image_bytes).hexdigest()
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')

    with col2:
        st.image(image, caption='Uploaded Image', width=300)

    with col3:
        st.write(' ')

    # Text input for company info and social media posts
    company_info = st.text_area(
        "Company Info", value="Example - Fenty Beauty by Rihanna was created with the promise of inclusion for all women. With an unmatched offering of shades and colors for ALL skin tones, you'll never look elsewhere for your beauty staples.")
    social_media_posts = st.text_area(
        "Recent Social Media Posts",
        value=(
            "Mirror mirror on the wall who's the baddest of them all...\n"
            "In the mood for sum soft smooth skin 😉 Grab a spoon for this #CookiesNClean Face Scrub...\n"
            "The cherry on top? #GlossBombHeat in 'Hot Cherry' 🔥🍒...\n"
            "All that glitters is gold ✨creating the perfect canvas for this gold AND bold look...\n"
            "Double the gloss double the glam 💦 Are you ready to #DoubleGloss fam?..."
        ),
    )

    if st.button("Generate Caption and Evaluate"):
        show_loading_overlay("Loading...")
        try:
            # Conditional image captioning
            text = "a photograph of"
            inputs = processor(image, text, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=40)
            conditional_caption = processor.decode(out[0], skip_special_tokens=True)

            # Unconditional image captioning
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=40)
            unconditional_caption = processor.decode(out[0], skip_special_tokens=True)

            social_post = generate_social_media_post(
                conditional_caption, unconditional_caption, company_info, social_media_posts)
            post_evaluation = evaluate_social_media_post(
                image, social_post, company_info)
            st.session_state["last_image_key"] = image_key
            st.session_state["conditional_caption"] = conditional_caption
            st.session_state["unconditional_caption"] = unconditional_caption
            st.session_state["social_post"] = social_post
            st.session_state["post_evaluation"] = post_evaluation
        finally:
            hide_loading_overlay()

    if st.session_state.get("last_image_key") == image_key:
        st.markdown(
            f"<div class=\"output-card generated-output\"><div class=\"caption-output\">Generated Social Media Post:</div><div class=\"caption-body\">{st.session_state.get('social_post', '')}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class=\"eval-output\">Social Media Post Evaluation:</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div id=\"eval-output\"></div>", unsafe_allow_html=True)
        st.markdown(st.session_state.get("post_evaluation", ""))
