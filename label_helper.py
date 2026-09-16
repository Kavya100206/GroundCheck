#!/usr/bin/env python3
"""
label_helper.py — Golden Set Labelling Helper (Phase 1b)

Usage:
    python3 label_helper.py

Behaviour:
  - Resumes automatically from the first row where gold_intent is blank.
  - Writes to CSV immediately after each completed row (no data loss on quit).
  - Contamination warnings:
      * playback_issue 'other' >= 4 of 34  -> pauses, asks whether to continue
      * overall 'other' > 15% of labelled  -> prints warning (does not stop)
  - Heuristic bucket and sampling stratum are hidden during labelling.
  - Type BACK at any prompt to undo and re-label the previous row.
  - Type q at any prompt to save and quit.
"""

import os
import sys
import pandas as pd

CSV_PATH = "/Users/kavya/Desktop/Groundcheck/golden_set_to_label.csv"

VALID_INTENTS = {
    "p": "playback_issue",
    "a": "account_login",
    "b": "premium_billing",
    "t": "app_technical",
    "d": "device_platform",
    "o": "other",
    "playback_issue":  "playback_issue",
    "account_login":   "account_login",
    "premium_billing": "premium_billing",
    "app_technical":   "app_technical",
    "device_platform": "device_platform",
    "other":           "other",
}

VALID_BOOL = {
    "y": "true",  "yes": "true",  "true": "true",
    "n": "false", "no": "false", "false": "false",
}

VALID_DEC = {
    "a": "auto_handle", "auto": "auto_handle", "auto_handle": "auto_handle",
    "e": "escalate",    "esc":  "escalate",    "escalate":    "escalate",
}

VALID_DIFF = {
    "e": "easy", "easy": "easy",
    "h": "hard", "hard": "hard",
}

PLAYBACK_OTHER_THRESHOLD    = 4  # >= 4 of 34 (>10%)
OVERALL_OTHER_PCT_THRESHOLD = 0.15
ARCHIVED_OTHER_COUNT        = 5  # From initial gs_0000-0003, 0009 labelled as 'other'
ARCHIVED_TOTAL_COUNT        = 6  # 5 other + 1 device_platform from initial pass

USE_COLOR = sys.stdout.isatty()

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def BOLD(t):   return _c("1",     t)
def RED(t):    return _c("1;31",  t)
def GREEN(t):  return _c("1;32",  t)
def YELLOW(t): return _c("1;33",  t)
def CYAN(t):   return _c("1;36",  t)
def DIM(t):    return _c("2",     t)


def clear():
    os.system("clear" if os.name == "posix" else "cls")


def prompt(label, hint="", allow_back=True):
    suffix = f"  {DIM(hint)}" if hint else ""
    while True:
        try:
            val = input(f"\n  {YELLOW(label)}{suffix}\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nQuitting — progress saved.")
            sys.exit(0)
        if val.lower() == "q":
            print("\nQuitting — progress saved.")
            sys.exit(0)
        if allow_back and val.upper() == "BACK":
            return "__BACK__"
        if val:
            return val
        print(f"  {RED('Enter a value (or q to quit).')}")


def load_csv():
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    for col in ["gold_intent", "has_visible_resolution",
                "gold_decision", "gold_decision_reason", "difficulty_tier"]:
        if col not in df.columns:
            df[col] = ""
    return df


def save_csv(df):
    df.to_csv(CSV_PATH, index=False)


def stats_bar(df, i):
    labelled        = int((df["gold_intent"] != "").sum())
    tot_labelled    = labelled + ARCHIVED_TOTAL_COUNT
    total           = len(df)
    active_oth      = int((df["gold_intent"] == "other").sum())
    tot_oth         = active_oth + ARCHIVED_OTHER_COUNT
    pb_oth          = int(((df["sampling_stratum"] == "playback_issue") &
                           (df["gold_intent"] == "other")).sum())
    oth_pct         = f"{tot_oth/tot_labelled:.0%}" if tot_labelled else "—"
    return (f"{GREEN(f'{labelled}/{total}')}  |  "
            f"overall other (inc. 5 archived): {tot_oth} ({oth_pct})  |  "
            f"playback other: {pb_oth}/34  |  "
            f"row: {i+1}")


def show_row(row, i, df):
    clear()
    print(BOLD("=" * 72))
    print(stats_bar(df, i))
    print(BOLD("=" * 72))
    print()
    print(BOLD("CUSTOMER TWEET:"))
    print(f"  {CYAN(str(row['customer_text']))}")
    print()
    reply = str(row.get("brand_reply_text", "")).strip()
    if reply and reply not in ("nan", ""):
        print(BOLD("BRAND REPLY:"))
        # Word-wrap at 70 chars
        words = reply.split()
        line, lines = "", []
        for w in words:
            if len(line) + len(w) + 1 > 68:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)
        for l in lines:
            print(f"  {l}")
        vis = str(row.get("has_brand_reply_visible_resolution", "")).lower()
        if vis == "true":
            print(f"\n  {DIM('resolution hint:')} {GREEN('visible')}")
        elif vis == "false":
            print(f"\n  {DIM('resolution hint:')} {RED('DM-redirect / no visible resolution')}")
    else:
        print(DIM("  [No brand reply captured]"))
    print()
    print(DIM(f"  id: {row['id']}  |  BACK to undo prev row  |  q to quit"))
    print()


def check_contamination(df, pb_oth_before, labelled_before):
    oth_mask = df["gold_intent"] == "other"
    pb_oth   = int(((df["sampling_stratum"] == "playback_issue") & oth_mask).sum())
    tot_oth  = int(oth_mask.sum()) + ARCHIVED_OTHER_COUNT
    labelled = int((df["gold_intent"] != "").sum()) + ARCHIVED_TOTAL_COUNT

    if pb_oth >= PLAYBACK_OTHER_THRESHOLD and pb_oth > pb_oth_before:
        print()
        print(RED(f"  REPLACEMENT BATCH WATCH: {pb_oth}/34 active playback_issue rows = 'other' (>={PLAYBACK_OTHER_THRESHOLD}, >10%)."))
        print(RED("  Note: In this replacement batch, firing signals cross-intent ambiguity"))
        print(RED("  or difficulty tier edge-cases rather than keyword miscalibration."))
        print()
        print("  Options:  1 = continue anyway   2 = stop and report")
        choice = input("  > ").strip()
        if choice == "2":
            print(f"\n{BOLD('Stopped. playback_issue other count: ')} {pb_oth}/34 (plus {ARCHIVED_OTHER_COUNT} archived)")
            print("Report this number to the user before continuing.")
            sys.exit(0)

    if labelled > 10:
        oth_pct = tot_oth / labelled
        prev_pct = (tot_oth - 1) / max(labelled - 1, 1)
        if oth_pct > OVERALL_OTHER_PCT_THRESHOLD and prev_pct <= OVERALL_OTHER_PCT_THRESHOLD:
            print()
            print(RED(f"  WARNING: Overall 'other' rate crossed 15% "
                      f"({tot_oth}/{labelled} = {oth_pct:.0%}, including {ARCHIVED_OTHER_COUNT} archived rows)."))
            print(RED("  Flag this to the user — heuristic-bucket reliability may be weak across multiple intents."))
            input("  Press Enter to continue... ")



def label_row(row):
    # 1. gold_intent
    val = prompt(
        "gold_intent",
        "[p]layback  [a]ccount_login  [b]illing  [t]ech  [d]evice  [o]ther",
    )
    if val == "__BACK__": return "__BACK__"
    if val.lower() not in VALID_INTENTS:
        print(RED("  Invalid — use p/a/b/t/d/o or the full intent name."))
        return label_row(row)
    gold_intent = VALID_INTENTS[val.lower()]

    # 2. has_visible_resolution
    val = prompt(
        "has_visible_resolution",
        "[y]es / [n]o — real troubleshooting steps in brand reply?",
    )
    if val == "__BACK__": return "__BACK__"
    if val.lower() not in VALID_BOOL:
        print(RED("  Invalid — enter y or n."))
        return label_row(row)
    has_vis = VALID_BOOL[val.lower()]

    # 3. gold_decision
    val = prompt(
        "gold_decision",
        "[a]uto_handle  /  [e]scalate  (default to e if unsure)",
    )
    if val == "__BACK__": return "__BACK__"
    if val.lower() not in VALID_DEC:
        print(RED("  Invalid — enter a (auto) or e (escalate)."))
        return label_row(row)
    gold_dec = VALID_DEC[val.lower()]

    # 4. gold_decision_reason
    val = prompt(
        "gold_decision_reason",
        'e.g. "billing dispute — hard gate" / "simple app reinstall"',
    )
    if val == "__BACK__": return "__BACK__"
    reason = val

    # 5. difficulty_tier
    val = prompt(
        "difficulty_tier",
        "[e]asy  /  [h]ard  (hard = took >1 moment to decide intent or escalation)",
    )
    if val == "__BACK__": return "__BACK__"
    if val.lower() not in VALID_DIFF:
        print(RED("  Invalid — enter e (easy) or h (hard)."))
        return label_row(row)
    diff = VALID_DIFF[val.lower()]

    return {
        "gold_intent":            gold_intent,
        "has_visible_resolution": has_vis,
        "gold_decision":          gold_dec,
        "gold_decision_reason":   reason,
        "difficulty_tier":        diff,
    }


def print_summary(df):
    clear()
    print(BOLD("=" * 72))
    print(BOLD("  LABELLING COMPLETE"))
    print(BOLD("=" * 72))
    labelled = int((df["gold_intent"] != "").sum())
    tot_labelled = labelled + ARCHIVED_TOTAL_COUNT
    tot_oth = int((df["gold_intent"] == "other").sum()) + ARCHIVED_OTHER_COUNT
    print(f"\n  Active golden set labelled: {labelled}/{len(df)}")
    print(f"  Total labelled (inc. {ARCHIVED_TOTAL_COUNT} archived): {tot_labelled}")
    print(f"  Overall 'other' count: {tot_oth} ({tot_oth/tot_labelled:.1%})\n")
    print("  Active set intent distribution:")
    print(df["gold_intent"].value_counts().to_string())
    print("\n  Active set decision distribution:")
    print(df["gold_decision"].value_counts().to_string())
    print("\n  Active set difficulty distribution:")
    print(df["difficulty_tier"].value_counts().to_string())
    other_df = df[df["gold_intent"] == "other"]
    if len(other_df):
        print("\n  Active set 'other' by heuristic bucket:")
        print(other_df["sampling_stratum"].value_counts().to_string())
    print(f"\n  {GREEN('Saved: ' + CSV_PATH)}\n")


def main():
    if not os.path.exists(CSV_PATH):
        print(RED(f"ERROR: File not found: {CSV_PATH}"))
        sys.exit(1)

    df = load_csv()
    unlabelled = df[df["gold_intent"] == ""]

    if unlabelled.empty:
        print(GREEN("All rows already labelled."))
        print_summary(df)
        return

    start_idx = unlabelled.index[0]
    done = int((df["gold_intent"] != "").sum())

    if start_idx > 0:
        print(GREEN(f"Resuming from row {start_idx + 1} / {len(df)}  ({done} done)."))
        input("  Press Enter to start... ")

    i = start_idx
    while i < len(df):
        row = df.iloc[i]
        pb_oth_before  = int(((df["sampling_stratum"] == "playback_issue") &
                               (df["gold_intent"] == "other")).sum())
        labelled_before = int((df["gold_intent"] != "").sum())

        show_row(row, i, df)
        result = label_row(row)

        if result == "__BACK__":
            if i == 0:
                print(RED("  Already at row 1 — cannot go back."))
                input("  Press Enter... ")
                continue
            prev = i - 1
            for col in ["gold_intent", "has_visible_resolution",
                        "gold_decision", "gold_decision_reason", "difficulty_tier"]:
                df.at[prev, col] = ""
            save_csv(df)
            i = prev
            continue

        for col, val in result.items():
            df.at[i, col] = val
        save_csv(df)

        check_contamination(df, pb_oth_before, labelled_before)
        i += 1

    print_summary(df)


if __name__ == "__main__":
    main()
