import torch
import time

def run_baseline(model, tokenizer, input_text, device="cuda"):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    start_time = time.time()
    outputs = model.generate(**inputs)
    latency = time.time() - start_time
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text, latency
