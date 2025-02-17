from src.run import run
from tqdm import tqdm


def run_batches(
    model_config_names: list[str], preprocessing_config_names: list[str] = []
):
    print(f"Running batch of length {len(model_config_names)}.")
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
        run(model_config_name, preprocessing_config_name, idx + 1)


if __name__ == "__main__":
    model_config_names = ["small_v0", "small_v1"]
    run_batches(model_config_names)
