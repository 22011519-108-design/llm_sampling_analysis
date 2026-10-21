import streamlit as st
import ollama
import pandas as pd
import os
from openai import OpenAI

st.set_page_config(page_title="LLM Sampling Analyzer", layout="wide")

# Hosted models (Groq) used when a GROQ_API_KEY secret is set, e.g. on Streamlit Cloud.
# Locally, without the secret, the app talks to Ollama as before.
HOSTED_MODELS = {"llama3.1": "llama-3.1-8b-instant", "qwen2.5": "qwen/qwen3.8-27b"}


def get_secret(name):
    try:
        return st.secrets.get(name) or os.environ.get(name)
    except Exception:
        return os.environ.get(name)


groq_key = get_secret("GROQ_API_KEY")
hosted_client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1") if groq_key else None

st.title("🔬 LLM Decoding Parameter & Sampling Analysis")
st.markdown("Evaluate how **Temperature**, **Top-K**, and **Top-P** impact local LLMs (Llama 3.1, Qwen 2.5).")

st.sidebar.header("Controls")
model_choice = st.sidebar.selectbox("Model", ["llama3.1", "qwen2.5"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
top_k = st.sidebar.slider("Top-K", 1, 100, 40, 1)
top_p = st.sidebar.slider("Top-P", 0.1, 1.0, 0.9, 0.05)
runs = st.sidebar.number_input("Generation Runs", min_value=1, max_value=5, value=3)

user_prompt = st.text_area("Test Prompt:", "What is the capital of France?")

if st.button("Run Live Experiment"):
    if user_prompt.strip():
        outputs = []
        if hosted_client:
            hosted_model = HOSTED_MODELS[model_choice]
            st.caption(f"Running on hosted model `{hosted_model}` via Groq. Top-K isn't supported by the hosted API, so it's ignored here.")
            try:
                with st.spinner("Generating responses..."):
                    for _ in range(runs):
                        res = hosted_client.chat.completions.create(
                            model=hosted_model,
                            messages=[{"role": "user", "content": user_prompt}],
                            temperature=temperature,
                            top_p=top_p,
                        )
                        outputs.append(res.choices[0].message.content)
            except Exception as e:
                st.error(f"The hosted model request failed: {e}")
        else:
            try:
                with st.spinner("Generating responses..."):
                    for _ in range(runs):
                        res = ollama.generate(
                            model=model_choice,
                            prompt=user_prompt,
                            options={"temperature": temperature, "top_k": top_k, "top_p": top_p}
                        )
                        outputs.append(res["response"])
            except (ConnectionError, ollama.ResponseError):
                st.warning(
                    "Live experiments need a running Ollama server with the selected model, "
                    "which isn't available in this hosted demo. Run the app locally with Ollama "
                    "to generate live responses — the benchmark results below were produced that way."
                )

        for idx, out in enumerate(outputs):
            with st.expander(f"Run {idx + 1}"):
                st.write(out)

if os.path.exists("experiment_results.csv"):
    st.markdown("---")
    st.subheader("📊 Benchmark Results Dataset")
    df = pd.read_csv("experiment_results.csv")
    st.dataframe(df)