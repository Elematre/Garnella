"""
prepare_data.py
---------------
Prepares the CIL sentiment classification dataset for LoRA-XS fine-tuning.

Train split
-----------
Reads train.csv (id, sentence, label), drops `id`, and splits 90/10
stratified by label.

Output:
    ~/data/train_split.csv  (~226,800 rows)  columns: sentence, label
    ~/data/val_split.csv    (~25,200 rows)   columns: sentence, label

Test split
----------
Reads test.csv (id, sentence) — no labels since this is the Kaggle test set.
Adds a dummy label=0 column so main_glue.py can load it with the same schema
as train/val. The dummy labels are never used; predictions come from logits.
The original test.csv is kept untouched so its `id` column can be used when
building the final Kaggle submission CSV.

Output:
    ~/data/test_inference.csv  (~168,000 rows)  columns: sentence, label
"""
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_TRAIN = "/cluster/courses/cil/text-classification/data/train.csv"
RAW_TEST  = "/cluster/courses/cil/text-classification/data/test.csv"

OUT_TRAIN = "/home/mdietsche/data/train_split.csv"
OUT_VAL   = "/home/mdietsche/data/val_split.csv"
OUT_TEST  = "/home/mdietsche/data/test_inference.csv"

# --- Train / validation split ---
train_df = pd.read_csv(RAW_TRAIN)[["sentence", "label"]]

train_df, val_df = train_test_split(
    train_df, test_size=0.1, random_state=42, stratify=train_df["label"]
)

train_df.to_csv(OUT_TRAIN, index=False)
val_df.to_csv(OUT_VAL, index=False)

print(f"Train: {len(train_df)} rows")
print(f"Val:   {len(val_df)} rows")

# --- Test inference file ---
# Keep only sentence; add dummy label so main_glue.py loads it without errors.
test_df = pd.read_csv(RAW_TEST)[["sentence"]]
test_df["label"] = 0

test_df.to_csv(OUT_TEST, index=False)

print(f"Test:  {len(test_df)} rows  →  {OUT_TEST}")