import pandas as pd

df = pd.read_parquet("vogue-data/VogueRunway.parquet")

df = df[
    df["designer"].str.contains(
        "Mugler", case=False, na=False
    )
    & (df["section"] == "Collection")
    & (df["width"] >= 1000)
    & (df["height"] >= 1500)
]

print(
    df.groupby(["designer", "year", "season", "category"],
               dropna=False)
      .size()
      .sort_values(ascending=False)
      .head(20)
)

sample = df.head(100)
sample.to_csv("training_selection.csv", index=False)
