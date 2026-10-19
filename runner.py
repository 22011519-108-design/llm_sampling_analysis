import json
import os
import ollama
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODELS = ["llama3.1", "qwen2.5"]
RUNS_PER_CONFIG = 3

# Parameter combinations to test
PARAM_GRID = [
    {"temperature": 0.0, "top_k": 10, "top_p": 0.5},
    {"temperature": 0.3, "top_k": 40, "top_p": 0.9},
    {"temperature": 0.8, "top_k": 40, "top_p": 0.9},
    {"temperature": 1.2, "top_k": 100, "top_p": 1.0},
]

def calculate_diversity(responses):
    if len(responses) < 2:
        return 0.0
    vectorizer = TfidfVectorizer().fit_transform(responses)
    vectors = vectorizer.toarray()
    sim_matrix = cosine_similarity(vectors)
    np.fill_diagonal(sim_matrix, np.nan)
    return float(1.0 - np.nanmean(sim_matrix))

def run_experiments():
    if not os.path.exists("benchmark_prompts.json"):
        print("[X] Error: benchmark_prompts.json not found!")
        return

    with open("benchmark_prompts.json", "r") as f:
        benchmarks = json.load(f)

    results = []

    for model in MODELS:
        print(f"\n[+] Testing Model: {model}")
        for item in benchmarks:
            cat = item["category"]
            prompt = item["prompt"]

            for config in PARAM_GRID:
                responses = []
                print(f" -> Running [{cat}]: Temp={config['temperature']}")

                for r in range(RUNS_PER_CONFIG):
                    try:
                        response = ollama.generate(
                            model=model,
                            prompt=prompt,
                            options={
                                "temperature": config["temperature"],
                                "top_k": config["top_k"],
                                "top_p": config["top_p"]
                            }
                        )
                        responses.append(response["response"].strip())
                    except Exception as e:
                        print(f"Error with model {model}: {e}")
                        responses.append("")

                diversity_score = calculate_diversity(responses)
                exact_matches = len(set(responses)) == 1

                results.append({
                    "model": model,
                    "category": cat,
                    "prompt": prompt,
                    "temperature": config["temperature"],
                    "top_k": config["top_k"],
                    "top_p": config["top_p"],
                    "runs_output": " || ".join(responses),
                    "lexical_diversity": diversity_score,
                    "is_deterministic": exact_matches
                })

    df = pd.DataFrame(results)
    df.to_csv("experiment_results.csv", index=False)
    print("\n[✔] Done! Results saved to experiment_results.csv")

if __name__ == "__main__":
    run_experiments()