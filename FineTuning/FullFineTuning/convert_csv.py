import pandas as pd

# 1. Load the original test data to get the IDs
test_df = pd.read_csv("~/data/test_inference.csv")

# 2. Load your current predictions (labels only)
preds_df = pd.read_csv("051126_1527/predictions.csv")

# 3. Combine them
# We assume the order in predictions.csv matches the order in test.csv
submission = pd.DataFrame({
    "id": test_df.index,
    "label": preds_df["label"]
})

# 4. Save the formatted version
submission.to_csv("051126_1527/final_submission.csv", index=False)

print("Formatted submission saved to submissions/final_submission.csv")