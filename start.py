import os
import subprocess
from datetime import datetime, timedelta
from collections import Counter
import pandas as pd

from quant_data_core import load_results_dataframe

SCRIPTS = [
    "deepseek_scr1.py",
    "deepseek_scr2.py",
    "deepseek_scr3.py",
    "deepseek_scr4.py",
    "deepseek_scr5.py",
    "deepseek_scr6.py",
    "deepseek_scr7.py",
    "deepseek_scr8.py",
    "deepseek_scr9.py",
]

SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]
TOP_N_LIST = [3, 5, 7, 10, 15, 20, 22, 25, 30, 35, 40]
MASTER_XLSX = os.path.join("predictions", "timemachine_scr1_9_predictions.xlsx")


def run_child_script(script_path: str) -> dict:
    """Run a child script and parse its stdout for slot predictions."""
    result = subprocess.run(["py", "-3.12", script_path], capture_output=True, text=True, check=False)
    stdout_lines = result.stdout.splitlines()
    preds = {}

    for line in stdout_lines:
        if ":" not in line:
            continue
        try:
            slot_part, nums_part = line.split(":", 1)
            slot = slot_part.strip().upper()
            if slot not in SLOTS:
                continue
            nums = [n.strip() for n in nums_part.split(",") if n.strip().isdigit()]
            preds[slot] = [int(n) for n in nums]
        except Exception:
            continue

    if result.returncode != 0:
        print(f"⚠️  Script {script_path} exited with code {result.returncode}")
        if result.stderr:
            print(result.stderr)

    return preds


def build_actual_results(df_full: pd.DataFrame):
    actual = {}
    for _, row in df_full.iterrows():
        d = row["DATE"].date()
        for slot in SLOTS:
            value = row.get(slot)
            if pd.notna(value):
                actual[(d, slot)] = int(value)
    return actual


def prepare_dates(df_full: pd.DataFrame):
    min_date = df_full["DATE"].min().date()
    max_date = df_full["DATE"].max().date()

    print(f"Available DATE range: {min_date} to {max_date}")
    start_str = input("Enter backtest start date (DD-MM-YYYY): ").strip()
    start_date = datetime.strptime(start_str, "%d-%m-%Y").date()

    if start_date <= min_date:
        raise ValueError(f"Start date must be after {min_date}")
    if start_date > max_date:
        raise ValueError(f"Start date cannot exceed {max_date}")

    target_dates = []
    curr = start_date
    while curr <= max_date:
        target_dates.append(curr)
        curr += timedelta(days=1)
    return target_dates, start_date, max_date


def collect_predictions(target_dates, actual):
    all_predictions = {}

    for target_date in target_dates:
        history_end = target_date - timedelta(days=1)
        os.environ["PREDICTOR_CUTOFF_DATE"] = history_end.strftime("%Y-%m-%d")
        os.environ["PP_RUN_MODE"] = "backtest"

        print(f"\n=== Generating predictions for {target_date} using data through {history_end} ===")

        for idx, script in enumerate(SCRIPTS, start=1):
            preds = run_child_script(script)
            for slot in SLOTS:
                nums = preds.get(slot, [])
                nums_as_strings = [str(n).zfill(2) for n in nums]
                key = (target_date, slot)
                all_predictions.setdefault(key, {})
                all_predictions[key][f"SCR{idx}"] = nums_as_strings

    os.environ.pop("PREDICTOR_CUTOFF_DATE", None)
    os.environ.pop("PP_RUN_MODE", None)

    return all_predictions


def save_master_excel(all_predictions):
    os.makedirs(os.path.dirname(MASTER_XLSX), exist_ok=True)

    rows = []
    for (d, slot), scripts_dict in all_predictions.items():
        row = {
            "DATE": d.strftime("%Y-%m-%d"),
            "SLOT": slot,
        }
        for idx in range(1, len(SCRIPTS) + 1):
            vals = scripts_dict.get(f"SCR{idx}", [])
            row[f"SCR{idx}"] = ", ".join(vals)
        rows.append(row)

    df_new = pd.DataFrame(rows)

    if not os.path.exists(MASTER_XLSX):
        with pd.ExcelWriter(MASTER_XLSX) as writer:
            df_new.to_excel(writer, sheet_name="raw_predictions", index=False)
        return df_new

    try:
        df_old = pd.read_excel(MASTER_XLSX, sheet_name="raw_predictions")
    except Exception:
        df_old = pd.DataFrame()

    df_old["KEY"] = df_old.get("DATE", "") + "|" + df_old.get("SLOT", "")
    df_new["KEY"] = df_new["DATE"] + "|" + df_new["SLOT"]

    df_old = df_old[~df_old["KEY"].isin(df_new["KEY"])] if not df_old.empty else pd.DataFrame()
    combined = pd.concat([df_old, df_new], ignore_index=True)
    combined = combined.drop(columns=["KEY"], errors="ignore")
    combined = combined.sort_values(["DATE", "SLOT"]).reset_index(drop=True)

    with pd.ExcelWriter(MASTER_XLSX) as writer:
        combined.to_excel(writer, sheet_name="raw_predictions", index=False)

    return combined


def calculate_roi(all_predictions, actual):
    total_stake = {n: 0 for n in TOP_N_LIST}
    total_return = {n: 0 for n in TOP_N_LIST}

    for key, scripts_dict in all_predictions.items():
        date, slot = key
        if (date, slot) not in actual:
            continue

        votes = Counter()
        for nums in scripts_dict.values():
            for s in nums:
                s2 = str(s).zfill(2)
                votes[s2] += 1

        if not votes:
            continue

        actual_num = str(actual[(date, slot)]).zfill(2)

        for n in TOP_N_LIST:
            top_list = [num for num, _ in votes.most_common(n)]
            stake = len(top_list)
            hit = actual_num in top_list
            total_stake[n] += stake
            total_return[n] += 90 if hit else 0

    summary_rows = []
    for n in TOP_N_LIST:
        stake = total_stake[n]
        ret = total_return[n]
        profit = ret - stake
        roi = (profit / stake * 100) if stake else 0.0
        summary_rows.append({
            "top_n": n,
            "total_stake": stake,
            "total_return": ret,
            "profit": profit,
            "roi_pct": roi,
        })

    return pd.DataFrame(summary_rows)


def main():
    os.environ.pop("PREDICTOR_CUTOFF_DATE", None)
    os.environ.pop("PP_RUN_MODE", None)

    df_full = load_results_dataframe()
    if df_full.empty:
        print("❌ No results data available.")
        return

    actual = build_actual_results(df_full)
    target_dates, start_date, max_date = prepare_dates(df_full)

    all_predictions = collect_predictions(target_dates, actual)

    combined_raw = save_master_excel(all_predictions)

    roi_df = calculate_roi(all_predictions, actual)

    # Persist ROI summary
    with pd.ExcelWriter(MASTER_XLSX, mode="a", if_sheet_exists="replace") as writer:
        combined_raw.to_excel(writer, sheet_name="raw_predictions", index=False)
        roi_df.to_excel(writer, sheet_name="roi_summary", index=False)

    best_roi_row = roi_df.loc[roi_df["roi_pct"].idxmax()]
    best_profit_row = roi_df.loc[roi_df["profit"].idxmax()]

    print("\n=== ENSEMBLE ROI SUMMARY (SCR1–SCR9) ===")
    print(f"Window: {start_date} to {max_date} (all 4 slots)")
    for _, row in roi_df.iterrows():
        print(
            f"Top{int(row['top_n']):<2}: stake={row['total_stake']}, "
            f"return={row['total_return']}, profit={row['profit']}, ROI={row['roi_pct']:.2f}%"
        )

    print(
        f"\nMax ROI   : Top{int(best_roi_row['top_n'])} "
        f"(ROI={best_roi_row['roi_pct']:.2f}%, profit={best_roi_row['profit']}, stake={best_roi_row['total_stake']})"
    )
    print(
        f"Max profit: Top{int(best_profit_row['top_n'])} "
        f"(profit={best_profit_row['profit']}, ROI={best_profit_row['roi_pct']:.2f}%, stake={best_profit_row['total_stake']})"
    )


if __name__ == "__main__":
    main()
