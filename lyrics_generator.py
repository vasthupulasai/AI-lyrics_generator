import streamlit as st
from transformers import pipeline, set_seed
from googletrans import Translator
import torch


st.set_page_config(page_title="AI Lyrics Generator", layout="centered")
st.title("🎵 AI Lyrics Generator")


with st.sidebar:
    st.header("🎛️ Settings")
    keywords = st.text_input("Enter a few keywords:", value="love, pain, hope")
    genre = st.selectbox("Genre:", ["Pop", "Hip-Hop", "Rock", "Romantic", "Sad", "Folk"])
    mood = st.selectbox("Mood:", ["Melancholic", "Uplifting", "Dark", "Peaceful", "Energetic", "Emotional"])
    language = st.selectbox("Language:", ["English", "Hindi", "Telugu", "Tamil", "Spanish"])
    scale = st.text_input("Scale (e.g., C Major, A Minor):", value="A Minor")
    tempo = st.slider("Tempo (BPM):", 60, 200, 120)


@st.cache_resource
def load_generator():
    return pipeline(
        "text-generation",
        model="EleutherAI/gpt-neo-1.3B",
        device=torch.device("cpu")
    )

generator = load_generator()
set_seed(42)


if st.button("🎤 Generate Lyrics"):
    with st.spinner("🎶 Composing lyrics... Please wait"):
        try:
            
            prompt = f"""{genre} song lyrics:
Topic: {keywords}
Mood: {mood}
Style: poetic, lyrical, rhyming
Scale: {scale}, Tempo: {tempo} BPM

[Verse 1]
"""

            st.markdown(f"**📝 Prompt:** {prompt.strip()}")

            
            result = generator(
                prompt,
                max_length=180,
                num_return_sequences=1,
                pad_token_id=50256,
                truncation=True
            )[0]["generated_text"]

            
            result = result.replace(prompt, "").strip()

        
            if language != "English":
                translator = Translator()
                translated = translator.translate(result, dest=language.lower())
                result = translated.text

            
            st.subheader("🎶 Generated Lyrics:")
            st.write(result)

        except Exception as e:
            st.error(f"❌ Error: {e}")


