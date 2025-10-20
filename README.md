# OptIForest-style Credit Demo

A minimal project mirroring the xiagll/OptIForest structure for running on `creditcard.csv`.
💳 Credit Card Anomaly Detection using OptiForest

This project applies the OptiForest (Optimized Isolation Forest) algorithm to detect fraudulent transactions in credit card datasets.
It replicates and extends the original OptiForest implementation, comparing its performance with the standard Isolation Forest.

📁 Project Structure
```plaintext
oplforest_creditcard/
├── demo_credit.py               # Main script to run OptiForest and Isolation Forest
├── optiforest.py                # Core OptiForest algorithm
├── synthesize_credit_v2.py      # Script to generate synthetic credit datasets
├── test_demo_credit.py          # Unit tests
├── requirements.txt             # Python dependencies
├── get_data.py                  # Script to auto-download datasets
├── README.md                    # Project documentation
└── data/                        # Folder for large CSV datasets (ignored in Git)
```

⚙️ Setup Instructions

Clone the repository
```plaintext
git clone https://github.com/nilishshakya11/oplforest_creditcard.git
cd oplforest_creditcard
```

Create and activate a virtual environment (recommended):
```plaintext
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```
📊 Dataset Information

⚠️ GitHub cannot host files larger than 100 MB.
The datasets used in this project (creditcard.csv and creditcard_synth_v2_lite.csv) are large and therefore excluded from Git.

You can download them manually:

— Manual Download

Download from GitHub Releases

Place them inside:
```plaintext
oplforest_creditcard/
└── data/
    ├── creditcard.csv
    └── creditcard_synth_v2_lite.csv

google drive link for data files :
```plaintext
https://drive.google.com/drive/folders/1ysGdc466Qq07Y3VINLpP8vbDsRecOmoW?usp=drive_link
```

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

```

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
```plaintext
pip install -r requirements.txt
```
