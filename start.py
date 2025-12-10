import os
import re
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

    pattern = re.compile(r"\b(FRBD|GZBD|GALI|DSWR)\b\s*[:\-]\s*(.*)", re.IGNORECASE)

    for line in stdout_lines:
        try:
            match = pattern.search(line)
            if not match:
                continue
            slot = match.group(1).upper()
            nums_part = match.group(2)
            nums = [n.strip() for n in re.split(r"[,\s]+", nums_part) if n.strip().isdigit()]
            preds[slot] = [int(n) for n in nums]
        except Exception:
            continue

    if result.returncode != 0:
        print(f"WARNING: Script {script_path} exited with code {result.returncode}")
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
    start_str = input("Enter backtest start date (DD-MM-YYYY, blank = auto): ").strip()

    if not start_str:
        start_date = min_date + timedelta(days=1)

        if os.path.exists(MASTER_XLSX):
            try:
                df_master = pd.read_excel(MASTER_XLSX, sheet_name="raw_predictions")
                if "DATE" in df_master.columns and not df_master.empty:
                    df_master["DATE"] = pd.to_datetime(df_master["DATE"], errors="coerce")
                    last_pred_date = df_master["DATE"].max()
                    if pd.notna(last_pred_date):
                        last_pred_date = last_pred_date.date()
                        if min_date <= last_pred_date < max_date:
                            start_date = last_pred_date + timedelta(days=1)
                        elif last_pred_date >= max_date:
                            print("INFO: All dates already processed; nothing to do.")
                            return [], None, max_date
            except Exception:
                pass
    else:
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


def save_master_excel(all_predictions, slot_metrics, roi_rows=None):
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
        metrics = slot_metrics.get((d, slot), {})
        row.update(metrics)
        rows.append(row)

    df_new = pd.DataFrame(rows)

    roi_cols = [f"top_{n}_roi" for n in TOP_N_LIST]
    desired_cols = (
        ["DATE", "SLOT"]
        + [f"SCR{idx}" for idx in range(1, len(SCRIPTS) + 1)]
        + ["final"]
        + roi_cols
    )
    df_new = df_new.reindex(columns=[c for c in desired_cols if c in df_new.columns])

    if not os.path.exists(MASTER_XLSX):
        with pd.ExcelWriter(MASTER_XLSX) as writer:
            df_new.to_excel(writer, sheet_name="raw_predictions", index=False)
            if roi_rows is not None:
                pd.DataFrame(roi_rows).to_excel(writer, sheet_name="roi_summary", index=False)
        return df_new

    old = pd.read_excel(MASTER_XLSX, sheet_name="raw_predictions")

    old["KEY"] = old["DATE"].astype(str) + "|" + old["SLOT"].astype(str)
    df_new["KEY"] = df_new["DATE"].astype(str) + "|" + df_new["SLOT"].astype(str)

    old = old[~old["KEY"].isin(df_new["KEY"])]
    combined = pd.concat([old, df_new], ignore_index=True)
    combined = combined.drop(columns=["KEY"])
    combined = combined.sort_values(["DATE", "SLOT"]).reset_index(drop=True)

    with pd.ExcelWriter(MASTER_XLSX) as writer:
        combined.to_excel(writer, sheet_name="raw_predictions", index=False)
        if roi_rows is not None:
            pd.DataFrame(roi_rows).to_excel(writer, sheet_name="roi_summary", index=False)

    return combined


def calculate_roi(all_predictions, actual):
    """
    Returns:
      roi_rows    – aggregate summary per Top-N
      slot_metrics – dict[(date, slot)] = {
           "final": "...",
           "top_3_roi": ...,
           "top_5_roi": ...,
           ...
       }
    """

    slot_metrics = {}
    total_stake = {n: 0 for n in TOP_N_LIST}
    total_return = {n: 0 for n in TOP_N_LIST}

    for (date, slot), scripts_dict in all_predictions.items():
        if (date, slot) not in actual:
            continue

        votes = Counter()
        for nums in scripts_dict.values():
            for s in nums:
                votes[str(s).zfill(2)] += 1

        if not votes:
            continue

        actual_num = str(actual[(date, slot)]).zfill(2)

        ordered = [num for num, _ in votes.most_common()]
        per_slot = {"final": ", ".join(ordered)}

        for n in TOP_N_LIST:
            top_list = ordered[:n]
            if not top_list:
                stake = 0
                ret = 0
                roi_pct = 0.0
            else:
                stake = len(top_list)
                hit = actual_num in top_list
                ret = 90 if hit else 0
                roi_pct = (ret - stake) / stake * 100.0

            total_stake[n] += stake
            total_return[n] += ret

            per_slot[f"top_{n}_roi"] = roi_pct

        slot_metrics[(date, slot)] = per_slot

    roi_rows = []
    for n in TOP_N_LIST:
        stake = total_stake[n]
        ret = total_return[n]
        profit = ret - stake
        roi_pct = (profit / stake * 100) if stake else 0.0
        roi_rows.append({
            "top_n": n,
            "total_stake": stake,
            "total_return": ret,
            "profit": profit,
            "roi_pct": roi_pct,
        })

    return roi_rows, slot_metrics


def main():
    os.environ.pop("PREDICTOR_CUTOFF_DATE", None)
    os.environ.pop("PP_RUN_MODE", None)

    df_full = load_results_dataframe()
    if df_full.empty:
        print("ERROR: No results data available.")
        return

    actual = build_actual_results(df_full)
    target_dates, start_date, max_date = prepare_dates(df_full)

    if not target_dates:
        print("INFO: Nothing to process.")
        return

    all_predictions = collect_predictions(target_dates, actual)

    roi_rows, slot_metrics = calculate_roi(all_predictions, actual)

    save_master_excel(all_predictions, slot_metrics, roi_rows)
    best_roi_row = max(roi_rows, key=lambda r: r["roi_pct"]) if roi_rows else None
    best_profit_row = max(roi_rows, key=lambda r: r["profit"]) if roi_rows else None

    print("\n=== ENSEMBLE ROI SUMMARY (SCR1-SCR9) ===")
    print(f"Window: {start_date} to {max_date} (all 4 slots)")
    for row in roi_rows:
        print(
            f"Top{int(row['top_n']):<2}: stake={row['total_stake']}, "
            f"return={row['total_return']}, profit={row['profit']}, ROI={row['roi_pct']:.2f}%"
        )

    if best_roi_row:
        print(
            f"\nBest ROI   : Top{int(best_roi_row['top_n'])} "
            f"(ROI={best_roi_row['roi_pct']:.2f}%, profit={best_roi_row['profit']}, stake={best_roi_row['total_stake']})"
        )
    if best_profit_row:
        print(
            f"Best profit: Top{int(best_profit_row['top_n'])} "
            f"(profit={best_profit_row['profit']}, ROI={best_profit_row['roi_pct']:.2f}%, stake={best_profit_row['total_stake']})"
        )


if __name__ == "__main__":
    main()
