import os
import subprocess

CONFIGS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
PREPROCESSING_CONFIGS_FOLDER = os.path.join(CONFIGS_FOLDER, "preprocessing")


def run_all_preprocessing():
    for filename in os.listdir(PREPROCESSING_CONFIGS_FOLDER):
        if filename.endswith(".yaml"):
            config_path = os.path.join(PREPROCESSING_CONFIGS_FOLDER, filename)
            print(f"Running preprocessing with config: {config_path}")
            subprocess.run(
                ["python", "-m", "src.preprocessing", "--name", config_path], check=True
            )


if __name__ == "__main__":
    run_all_preprocessing()
