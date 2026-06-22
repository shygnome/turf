"""List decisive matches where the winner had fewer LBP/30 min than the loser."""
import ast

import pandas as pd

goals = pd.read_csv("data/wc2022_goals.csv", comment="#")
score = (
    goals.groupby(["match_id", "scoring_team"]).size()
    .unstack(fill_value=0)
    .reindex(columns=["Home", "Away"], fill_value=0)
    .rename(columns={"Home": "home_goals", "Away": "away_goals"})
)
meta = pd.read_csv("data/preprocessed/pff/fifa-wc-2022/metadata.csv")


def _has_any_around(s):
    try:
        return "Around" in ast.literal_eval(str(s))
    except Exception:
        return False


records = []
for mid in sorted(score.index):
    try:
        lm = pd.read_csv(f"output/pff/fifa-wc-2022/{mid}/pass/labeled_metadata.csv")
        ps = pd.read_csv(f"output/pff/fifa-wc-2022/{mid}/possession_summary.csv")
    except FileNotFoundError:
        continue
    lm = lm[lm["subtype"] == "success"].copy()
    ib = lm["is_line_breaking"].astype(bool)
    lm["_all"] = ib
    lm["_any"] = ib & lm["direction_per_line"].apply(_has_any_around)
    for team in ("Home", "Away"):
        t = lm[lm["team"] == team]
        col = "home_sec" if team == "Home" else "away_sec"
        poss_min = float(ps[col].sum()) / 60.0
        records.append({
            "match_id": mid, "team": team, "poss_min": poss_min,
            "all_breaks": int(t["_all"].sum()),
            "around_any": int(t["_any"].sum()),
        })

df = pd.DataFrame(records)
for col in ("all_breaks", "around_any"):
    df[f"{col}_per_30"] = df[col] / (df["poss_min"] / 30)

pivot = df.pivot(
    index="match_id",
    columns="team",
    values=[
        "all_breaks_per_30", "around_any_per_30",
        "all_breaks", "around_any", "poss_min",
    ],
)
pivot.columns = [f"{c}_{t.lower()}" for c, t in pivot.columns]
pivot = pivot.merge(score, left_index=True, right_index=True, how="inner")
pivot = pivot.merge(
    meta[["match_id", "home_team_name", "away_team_name"]].set_index("match_id"),
    left_index=True, right_index=True, how="left"
)

rows = []
for mid, row in pivot.iterrows():
    hg, ag = int(row["home_goals"]), int(row["away_goals"])
    if hg == ag:
        continue
    winner = "home" if hg > ag else "away"
    loser  = "away" if hg > ag else "home"
    wr = float(row[f"all_breaks_per_30_{winner}"])
    lr = float(row[f"all_breaks_per_30_{loser}"])
    if wr < lr:
        rows.append({
            "match_id":       mid,
            "home":           row["home_team_name"],
            "away":           row["away_team_name"],
            "score":          f"{hg}-{ag}",
            "winner":         winner.upper(),
            "W_lbp":          int(row[f"all_breaks_{winner}"]),
            "L_lbp":          int(row[f"all_breaks_{loser}"]),
            "W_poss_min":     round(float(row[f"poss_min_{winner}"]), 1),
            "L_poss_min":     round(float(row[f"poss_min_{loser}"]), 1),
            "W_per30":        round(wr, 2),
            "L_per30":        round(lr, 2),
            "diff":           round(wr - lr, 2),
        })

out = pd.DataFrame(rows).sort_values("diff")
n_decisive = sum(
    1 for _, r in pivot.iterrows()
    if int(r["home_goals"]) != int(r["away_goals"])
)
n_fewer = len(out)
print(f"Winner had FEWER LBP/30 than loser: {n_fewer} / {n_decisive} decisive matches\n")  # noqa: E501
pd.set_option("display.max_colwidth", 22)
pd.set_option("display.width", 130)
print(out.to_string(index=False))
