import hashlib
from src.run import run
from tqdm import tqdm
import datetime

def generate_batch_folder_name(model_config_names: list[str]) -> str:
    """Generates a unique batch folder name based on timestamp and model names."""
    now = datetime.now()
    timestamp = now.strftime("%b%d:%a:h%H")
    model_summary = "-".join(model_config_names)[:50]  # Keep it readable, truncate if long
    hash_suffix = hashlib.sha1("".join(model_config_names).encode()).hexdigest()[:6]  # Short hash
    return f"batch_{timestamp}_{len(model_config_names)}models_{hash_suffix}"

def run_batches(
    model_config_names: list[str], preprocessing_config_names: list[str] = [], batch_folder_name = None
):
    if batch_folder_name is None:
        batch_folder_name = generate_batch_folder_name(model_config_names)

    print(f"Running batch of length {len(model_config_names)}.")
    print(f"Batch folder: {batch_folder_name}")
    print("Running the following model_configs:")
    print("\n".join(model_config_names))

    for idx, model_config_name in enumerate(
        tqdm(model_config_names, desc="Processing Models", unit="model")
    ):
        if preprocessing_config_names:
            preprocessing_config_name = preprocessing_config_names[idx]
        else:
            preprocessing_config_name = ""
        print(f"@@@@ Batch number {idx+1}: {model_config_name} @@@@")
        run(model_config_name, preprocessing_config_name, idx + 1, batch_folder_name)


if __name__ == "__main__":
    model_config_names = ["128_v0", "128_v1"]
    run_batches(model_config_names)
