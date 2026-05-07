import subprocess
from pathlib import Path


def main():
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "ammarnassanalhajali/pklot-dataset",
            "-p",
            str(data_dir),
            "--unzip",
        ],
        check=True,
    )

    print("Data downloaded to data/raw")


if __name__ == "__main__":
    main()
