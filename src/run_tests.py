''' Tests wether a set of default model configs work when ran in a batch. '''
from src.run_batches import run_batches

if __name__ == "__main__":
    model_config_names = ["37_v0", "37_v1", "37_v2"]
    print(f"Testing models: {model_config_names}")
    run_batches(model_config_names=model_config_names, batch_folder_name="test")
    print("Batch test complete.")