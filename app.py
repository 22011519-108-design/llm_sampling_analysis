import streamlit as st
import ollama
import pandas as pd
import os

st.set_page_config(page_title="LLM Sampling Analyzer", layout="wide")

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