# Parking-Spaces-Classifier

Final project for the Neural Networks course (University of Wrocław, 2026).

Authors:
Kornel Orawczak, Marcin Mularczyk, Mateusz Matyskiel, Jan Lachowski

## 🚀 Quick Setup

This project uses uv to manage dependencies and virtual environment.

### 1. Prerequisites 

- Python 3.14+
- `uv` installed (`pip install uv`)

### 2. Clone repository and install dependencies

```
git clone https://github.com/Neural-Networks-Project-2026/parking-spaces-classifier.git
cd parking-spaces-classifier
uv sync --dev
uv run pre-commit install
```

### 3. Activating venv (optional)

```
# MacOS/Linux
source .venv/bin/activate

# Windows
source .venv/Scripts/activate
```

## 📦 Adding dependencies

Production dependency:
`uv add package_name`

Dev dependency:
`uv add --dev package_name`

## 🧹 Code quality (Ruff)
We use Ruff to keep the code clean and consistent.
It automatically checks for mistakes and enforces a common style across the project.

Basic ruff commands:
- Check code:
`uv run ruff check .`

- Auto-fix issues:
`uv run ruff check . --fix`

- Format code:
`uv run ruff format .`

## 🔒 Pre-commit

We use pre-commit hooks to automatically run checks before every commit. If your commit didn’t pass the first time, Ruff will often fix the issues automatically, so you just need to stage changes again and commit once more.

## 📊 Dataset setup (Kaggle)

1. Create API token:
Kaggle → Settings → API → Create New Token

2. Set environment variable:
```
# macOS / Linux
echo 'export KAGGLE_API_TOKEN="YOUR_TOKEN_HERE"' >> ~/.zshrc
source ~/.zshrc

# Windows
setx KAGGLE_API_TOKEN "YOUR_TOKEN_HERE"
```
3. Download dataset:
`uv run fetch-data`

Data will be in:
data/raw/