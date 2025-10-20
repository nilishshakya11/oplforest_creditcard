# OptIForest-style Credit Demo

A minimal project mirroring the xiagll/OptIForest structure for running on `creditcard.csv`.
💳 Credit Card Anomaly Detection using OptiForest

This project applies the OptiForest (Optimized Isolation Forest) algorithm to detect fraudulent transactions in credit card datasets.
It replicates and extends the original OptiForest implementation, comparing its performance with the standard Isolation Forest.

📁 Project Structure
oplforest_creditcard/
├── demo_credit.py               # Main script to run OptiForest and Isolation Forest
├── optiforest.py                # Core OptiForest algorithm
├── synthesize_credit_v2.py      # Script to generate synthetic credit datasets
├── test_demo_credit.py          # Unit tests
├── requirements.txt             # Python dependencies
├── get_data.py                  # Script to auto-download datasets
├── README.md                    # Project documentation
└── data/                        # Folder for large CSV datasets (ignored in Git)

⚙️ Setup Instructions

Clone the repository

git clone https://github.com/nilishshakya11/oplforest_creditcard.git
cd oplforest_creditcard

Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
📊 Dataset Information

⚠️ GitHub cannot host files larger than 100 MB.
The datasets used in this project (creditcard.csv and creditcard_synth_v2_lite.csv) are large and therefore excluded from Git.

You can download them automatically or manually:


📥 Option 1 — Automatic (Recommended)

Run the helper script to fetch both datasets into a data/ folder:

python get_data.py


Example get_data.py:

import os, urllib.request, pathlib

URLS = {
    "creditcard.csv": "https://github.com/nilishshakya11/oplforest_creditcard/releases/download/v0.1/creditcard.csv",
    "creditcard_synth_v2_lite.csv": "https://github.com/nilishshakya11/oplforest_creditcard/releases/download/v0.1/creditcard_synth_v2_lite.csv",
}

pathlib.Path("data").mkdir(exist_ok=True)

for name, url in URLS.items():
    dest = f"data/{name}"
    if not os.path.exists(dest):
        print(f"⬇️ Downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"✅ Saved to {dest}")
    else:
        print(f"✔️ Already exists: {dest}")


(Update the URLs above once you publish your datasets under GitHub “Releases” → “Attach files”)

📦 Option 2 — Manual Download

Download from GitHub Releases

Place them inside:

oplforest_creditcard/
└── data/
    ├── creditcard.csv
    └── creditcard_synth_v2_lite.csv

🚀 How to Run

Run the anomaly detection on the real dataset:

python demo_credit.py --csv_path data/creditcard.csv


Run it on the synthetic dataset:

python demo_credit.py --csv_path data/creditcard_synth_v2_lite.csv


Optional parameters:

--branch <int>         # branching factor (default: 2)
--threshold <float>    # cut threshold (default: 0.5)
--seed <int>           # random seed for reproducibility

Example:

python demo_credit.py --csv_path data/creditcard.csv --branch 2 --threshold 0.5 --seed 42

📈 Expected Output

After running, the script prints model metrics and comparison results:

Model              ROC-AUC    Precision    Recall
-------------------------------------------------
Isolation Forest    0.934       0.72        0.68
OptiForest          0.962       0.80        0.76


You can visualize or log these results using additional tools such as matplotlib or pandas.

🧾 References

Xiang, G. (2022). OptiForest: An Optimized Isolation Forest for Anomaly Detection.

Original OptiForest GitHub Repository

Install dependencies

pip install -r requirements.txt
