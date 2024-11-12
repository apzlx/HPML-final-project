import torch
from utils.model_loader import load_model
from baseline import run_baseline
from optimizations.mixed_precision import enable_mixed_precision
from profiling.profiler import profile_model

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    model, tokenizer = load_model(model_name, device)

    # Baseline
    input_text = "What is the meaning of life?"
    print("Running Baseline...")
    generated_text, latency = run_baseline(model, tokenizer, input_text, device)
    print(f"Generated: {generated_text}")
    print(f"Latency: {latency} seconds")

    # Mixed Precision Optimization
    print("Running Mixed Precision Optimization...")
    enable_mixed_precision()
    generated_text, latency = run_baseline(model, tokenizer, input_text, device)
    print(f"Optimized Latency: {latency} seconds")

    # Profiling
    print("Profiling Model...")
    print(profile_model(model, tokenizer, input_text, device))
