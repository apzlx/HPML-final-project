import torch
from torch.profiler import profile, record_function, ProfilerActivity

def profile_model(model, tokenizer, input_text, device="cuda", log_dir='./log', row_limit=10):
    """
    Profiles the model inference on CPU and CUDA, saving results for TensorBoard.

    Args:
        model: The PyTorch model to profile.
        tokenizer: Tokenizer for processing input text.
        input_text (str): Input text to pass through the model.
        device (str): Device to run profiling on ('cuda' or 'cpu').
        log_dir (str): Directory to save profiling logs for TensorBoard.
        row_limit (int): Number of rows to display in the profiling table.

    Returns:
        str: Profiling table summary sorted by CUDA time.
    """
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        with_stack=True,  # Enables stack tracing for better insights
        record_shapes=True  # Logs tensor shapes for deeper analysis
    ) as prof:
        with record_function("model_inference"):
            model.generate(**inputs)

    # Print and return profiling summary
    result_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit)
    print(result_table)
    return result_table
