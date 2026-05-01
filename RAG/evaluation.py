import pandas as pd
from bert_score import score

df = pd.read_csv("../data/evaluation_results_sea_lion_gemma_finetuned_only.csv")
candidates = df["Gemma-SEA-LION-v4-27B Finetuned Only - 78%"].astype(str).tolist()
references = df["Ground Truth"].astype(str).tolist()

P, R, F1 = score(
    candidates,
    references,
    lang="my",  # Myanmar language
    model_type="xlm-roberta-base",  # important for Burmese
    verbose=True
)

print("Precision:", sum(P.tolist()) / len(P))
print("Recall:", sum(R.tolist()) / len(R))
print("F1:", sum(F1.tolist()) / len(F1))