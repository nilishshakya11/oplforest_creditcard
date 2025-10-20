# Generate the synthetic CSV now by running the synthesize_credit.py script
import runpy, sys, os, pandas as pd

script = "/synthesize_credit.py"
csv_in = "/creditcard.csv"
csv_out = "/creditcard_synth.csv"

# Prepare argv and run the script
sys.argv = ["synthesize_credit.py",
            "--csv_path", csv_in,
            "--out_path", csv_out,
            "--synth_ratio", "0.01",
            "--seed", "42"]

runpy.run_path(script, run_name="__main__")

# Verify the file and show quick shape + fraud counts
df = pd.read_csv(csv_out)
shape = df.shape
fraud_counts = df["Class"].value_counts().to_dict()
shape, fraud_counts
