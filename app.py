import re
import pdfplumber
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Racing Strategy Matrix", page_icon="🏇", layout="wide")


def parse_sa_racecoast_pdf(uploaded_file) -> pd.DataFrame:
    runners = []
    current_race = None
    race_distance = 1200
    speed_map = {}
    rest_map = {}

    race_header_re = re.compile(r"RACE\s+(\d+)\s*-\s*(?:Abt/)?(\d+)m", re.IGNORECASE)
    runner_header_re = re.compile(
        r"^(?P<No>\d{1,2})\s+"
        r"(?P<Horse>[A-Z0-9\'\-\’]+(?:\s+[A-Z0-9\'\-\’]+)*?)\s+"
        r"(?P<Gear>[ASLF](?:\s+[BCPXHJKNZte])*)\s+"
        r"(?P<Draw>\d{1,2}|RESERVE\s*\d)\s+"
        r"(?P<Weight>\d{2}(?:\.\d)?)"
        r"(?:\s+\((?P<MR>\d+)\))?"
        r"(?:\s+\[(?P<LastMR>\d+)\])?$"
    )

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = [line.strip() for line in text.splitlines() if line.strip()]

            page_race_match = None
            for line in lines[:6]:
                m = race_header_re.search(line)
                if m:
                    page_race_match = m
                    break
            if not page_race_match:
                continue

            current_race = int(page_race_match.group(1))
            race_distance = int(page_race_match.group(2))

            full_page_text = " ".join(lines)
            speed_block = re.search(r"SPEED RATINGS:\s*(.*?)(?:BEST AVERAGE:|DAYS RESTED:)", full_page_text)
            if speed_block:
                for item in speed_block.group(1).split(";"):
                    sm = re.search(r"([A-Za-z\s\'\-\’]+?)\s+(-?\d+\.\d+)", item.strip())
                    if sm:
                        speed_map[sm.group(1).strip().upper()] = float(sm.group(2))

            rest_block = re.search(r"DAYS RESTED:\s*(.*?)(?:\b1\s+[A-Z]{2,})", full_page_text)
            if rest_block:
                for item in rest_block.group(1).split(";"):
                    rm = re.search(r"([A-Za-z\s\'\-\’]+?)\s*\((\d+)\s*days\)", item.strip(), re.IGNORECASE)
                    if rm:
                        rest_map[rm.group(1).strip().upper()] = int(rm.group(2))

            current_runner = None
            for line in lines:
                rm = runner_header_re.match(line)
                if rm:
                    if current_runner:
                        runners.append(current_runner)

                    horse_name = rm.group("Horse").strip().upper()
                    draw_raw = rm.group("Draw")
                    draw_num = int(draw_raw) if draw_raw.isdigit() else 99
                    mr_val = int(rm.group("MR")) if rm.group("MR") else 0

                    current_runner = {
                        "Race": current_race,
                        "Distance": race_distance,
                        "No": int(rm.group("No")),
                        "Horse": horse_name,
                        "Draw": draw_num,
                        "Weight": float(rm.group("Weight")),
                        "MR": mr_val,
                        "Runs": 0,
                        "Wins": 0,
                        "Places": 0,
                        "Stakes": 0,
                        "Speed_Rating": speed_map.get(horse_name, np.nan),
                        "Days_Rested": rest_map.get(horse_name, 21),
                        "Last_3_Pos": [],
                        "Last_Times": []
                    }
                    continue

                if current_runner:
                    stats_m = re.search(
                        r"Runs:\s*(\d+)\s+Wins:\s*(\d+)\s+Places:\s*(\d+)\s+Stakes Won:\s*R([\d,]+)",
                        line
                    )
                    if stats_m:
                        current_runner["Runs"] = int(stats_m.group(1))
                        current_runner["Wins"] = int(stats_m.group(2))
                        current_runner["Places"] = int(stats_m.group(3))
                        current_runner["Stakes"] = int(stats_m.group(4).replace(",", ""))
                        continue

                    run_m = re.match(
                        r"^[A-Z]\s+\d+\s+\d{2}\.\d{2}\.\d{2}\s+[A-Z]\s+\S+\s+(\d{3,4})\s+.*?\b(\d{1,2})\s+(\d+\.\d{2})\s+.*?\s+(\d{2,3}\.\d{2})\s+\d+/",
                        line
                    )
                    if run_m:
                        dist_ran = int(run_m.group(1))
                        finish_pos = int(run_m.group(2))
                        time_sec = float(run_m.group(4))
                        current_runner["Last_3_Pos"].append(finish_pos)
                        if time_sec > 0:
                            current_runner["Last_Times"].append(dist_ran / time_sec)

            if current_runner:
                runners.append(current_runner)

    df = pd.DataFrame(runners)
    if df.empty:
        return df

    df["Avg_Speed_MPS"] = df["Last_Times"].apply(lambda x: np.mean(x[-3:]) if len(x) > 0 else np.nan)
    df["Avg_Speed_MPS"] = df["Avg_Speed_MPS"].fillna(df.groupby("Race")["Avg_Speed_MPS"].transform("mean")).fillna(16.0)
    df["Speed_Metric"] = df["Speed_Rating"].fillna(
        df.groupby("Race")["Avg_Speed_MPS"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-5))
    ).fillna(0.0)

    df["Earnings_Per_Run"] = np.where(df["Runs"] > 0, df["Stakes"] / df["Runs"], 0)
    df["Win_Place_Rate"] = np.where(df["Runs"] > 0, (df["Wins"] * 2 + df["Places"]) / df["Runs"], 0)
    df["Recent_Finish_Avg"] = df["Last_3_Pos"].apply(lambda x: np.mean(x[-3:]) if len(x) > 0 else 8.0)

    return df.drop(columns=["Last_3_Pos", "Last_Times"])


def parse_hkjc_pdf(uploaded_file) -> pd.DataFrame:
    all_runners = []
    current_race = 1

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            rm = re.search(r"Race\s+(\d+)", text)
            if rm:
                current_race = int(rm.group(1))

            for table in page.extract_tables():
                for row in table:
                    c = [cell.replace("\n", " ").strip() if cell else "" for cell in row]
                    if len(c) >= 9 and c.isdigit() and 1 <= int(c) <= 14:
                        last_6 = [int(x) for x in c.split("/") if x.isdigit()]
                        all_runners.append({
                            "Race": current_race,
                            "No": int(c),
                            "Last_6_Runs": c,
                            "Recent_Finish_Avg": np.mean(last_6) if last_6 else 8.0,
                            "Horse": c,
                            "Weight": int(c) if c.isdigit() else 125,
                            "Jockey": c,
                            "Draw": int(c) if c.isdigit() else 7,
                            "Trainer": c,
                            "MR": int(c) if c.isdigit() else 50,
                        })
    return pd.DataFrame(all_runners)


def score_runners(df: pd.DataFrame, weights: dict, jurisdiction: str) -> pd.DataFrame:
    scored = df.copy()

    def min_max(series):
        if series.max() == series.min():
            return pd.Series(0.5, index=series.index)
        return (series - series.min()) / (series.max() - series.min())

    for race_id, group in scored.groupby("Race"):
        idx = group.index
        draw_score = 1.0 - min_max(group["Draw"].clip(upper=16))
        form_score = 1.0 - min_max(group["Recent_Finish_Avg"].clip(upper=14))

        if jurisdiction == "SA":
            class_raw = group["MR"].replace(0, group["MR"].median()) + (group["Earnings_Per_Run"] / 5000.0)
            speed_score = min_max(group["Speed_Metric"])
            pace_score = min_max(group["Win_Place_Rate"]) * (1.0 - min_max(group["Days_Rested"].clip(upper=120)) * 0.3)
        else:
            class_raw = group["MR"]
            speed_score = min_max(group["MR"] - (group["Weight"] - 120) * 0.5)
            pace_score = 1.0 - min_max(group["Weight"])

        class_score = min_max(class_raw)

        total = (
            weights["speed"] * speed_score +
            weights["class"] * class_score +
            weights["pace"] * pace_score +
            weights["draw"] * draw_score +
            weights["form"] * form_score
        ) * 100

        scored.loc[idx, "Matrix_Score"] = total.round(1)

    return scored.sort_values(["Race", "Matrix_Score"], ascending=[True, False])


tab_sa, tab_hk, tab_uploads = st.tabs(["🇿🇦 South African", "🇭🇰 Hong Kong", "📂 Uploads"])

with tab_uploads:
    st.header("📂 Race Data Uploads")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🇿🇦 South Africa (Race Coast)")
        sa_pdf = st.file_uploader("Upload Full Computerform PDF", type=["pdf"], key="sa_pdf")
        if sa_pdf is not None:
            sa_df = parse_sa_racecoast_pdf(sa_pdf)
            if not sa_df.empty:
                st.session_state["sa_df"] = sa_df
                st.success(f"Loaded {len(sa_df)} runners across {sa_df['Race'].nunique()} races! Tap '🇿🇦 South African' at the top.")
                st.dataframe(sa_df[sa_df["Race"] == 1], use_container_width=True)
            else:
                st.warning("No standard Race Coast runner rows found in this PDF.")

        sa_csv = st.file_uploader("Or Upload SA CSV / Excel", type=["csv", "xlsx", "xls"], key="sa_csv")
        if sa_csv is not None:
            sa_df = pd.read_csv(sa_csv) if sa_csv.name.endswith(".csv") else pd.read_excel(sa_csv)
            st.session_state["sa_df"] = sa_df
            st.success(f"Loaded {len(sa_df)} rows from spreadsheet!")

    with col2:
        st.subheader("🇭🇰 Hong Kong (HKJC)")
        hk_pdf = st.file_uploader("Upload HKJC Race Form PDF (All Races)", type=["pdf"], key="hk_pdf")
        if hk_pdf is not None:
            hk_df = parse_hkjc_pdf(hk_pdf)
            if not hk_df.empty:
                st.session_state["hk_df"] = hk_df
