import pandas as pd
import numpy as np

# --- SETTINGS ---
INPUT_FILE = "creditcard.csv"          # must be in same folder
OUTPUT_FILE = "creditcard_synth_v2_lite.csv"
SYNTH_RATIO = 0.01                     # 1% synthetic anomalies
SEED = 123

# --- LOAD ORIGINAL DATA ---
print("Loading:", INPUT_FILE)
df = pd.read_csv(INPUT_FILE)
rng = np.random.default_rng(SEED)
pca_cols = [c for c in df.columns if c.startswith("V")]

# --- SAMPLE NORMAL TRANSACTIONS ---
normals = df[df["Class"] == 0]
n_synth = int(len(normals) * SYNTH_RATIO)
sampled = normals.sample(n=n_synth, random_state=SEED).reset_index(drop=True)

# --- CREATE MIXED ANOMALIES ---
pca_stds = df[pca_cols].std(ddof=0).replace(0, 1.0).values
anom_types = rng.choice(["AmountSpike", "PCAShift", "Both"], size=n_synth)

synth = sampled.copy()
synth["Synthetic"] = 1
synth["AnomalyType"] = anom_types
synth["Class"] = 1

mask_amount = (anom_types == "AmountSpike") | (anom_types == "Both")
scales = rng.uniform(5.0, 40.0, size=mask_amount.sum())
synth.loc[mask_amount, "Amount"] *= scales

mask_pca = (anom_types == "PCAShift") | (anom_types == "Both")
noise = rng.normal(0, 3, size=(mask_pca.sum(), len(pca_cols))) * pca_stds
synth.loc[mask_pca, pca_cols] = synth.loc[mask_pca, pca_cols].values + noise

if "Time" in synth.columns:
    jitter = rng.integers(60, 3600, size=len(synth))
    synth["Time"] += jitter

# --- MERGE AND SAVE ---
df["Synthetic"] = 0
df["AnomalyType"] = "None"
out = pd.concat([df, synth], ignore_index=True)
out.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Saved new dataset: {OUTPUT_FILE}")
print(f"Total rows: {len(out):,}")
print(f"Frauds (including synthetic): {(out['Class']==1).sum():,}")
