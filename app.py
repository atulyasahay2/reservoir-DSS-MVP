import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Reservoir DSS MVP", layout="wide")

# -----------------------------
# Helpers: mock inputs + "black-box-like" schedule generator
# -----------------------------
def make_timeseries(start_ts: pd.Timestamp, hours: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start_ts, periods=hours, freq="H")
    # Mock: daily pattern + noise
    base = 120 + 30*np.sin(np.linspace(0, 3*np.pi, hours))
    noise = rng.normal(0, 6, size=hours)
    return idx, np.clip(base + noise, 0, None)

def generate_inputs(horizon_hours: int, seed: int = 7):
    start = pd.Timestamp.now().floor("H")
    idx, s1 = make_timeseries(start, horizon_hours, seed=seed)
    _, r1 = make_timeseries(start, horizon_hours, seed=seed+1)
    _, r2 = make_timeseries(start, horizon_hours, seed=seed+2)
    _, d1 = make_timeseries(start, horizon_hours, seed=seed+3)
    _, d2 = make_timeseries(start, horizon_hours, seed=seed+4)

    # Commitments: minimum environmental flow (S4) + downstream (S5)
    s4_min = np.full(horizon_hours, 40.0)
    s5_min = np.full(horizon_hours, 35.0)

    df = pd.DataFrame({
        "datetime": idx,
        "river_inflow_S1": s1,
        "upstream_release_R1": r1,
        "upstream_release_R2": r2,
        "demand_D1": d1 * 0.55,
        "demand_D2": d2 * 0.45,
        "commitment_S4_min": s4_min,
        "commitment_S5_min": s5_min
    }).set_index("datetime")

    return df

def pseudo_black_box_schedule(inputs: pd.DataFrame,
                              storage_start: float,
                              storage_min: float,
                              storage_max: float,
                              max_release_per_hour: float,
                              scenario_multipliers: dict):
    """
    NOTE: This is NOT the optimization model (out of scope). It's a simple heuristic
    to create a working MVP UI and end-to-end flow.
    """
    df = inputs.copy()

    # Apply scenario multipliers
    df["river_inflow_S1"] *= scenario_multipliers["river"]
    df["upstream_release_R1"] *= scenario_multipliers["upstream"]
    df["upstream_release_R2"] *= scenario_multipliers["upstream"]
    df["demand_D1"] *= scenario_multipliers["demand"]
    df["demand_D2"] *= scenario_multipliers["demand"]
    df["commitment_S4_min"] *= scenario_multipliers["commitments"]
    df["commitment_S5_min"] *= scenario_multipliers["commitments"]

    storage = storage_start
    rows = []

    for ts, r in df.iterrows():
        inflow = float(r["river_inflow_S1"] + r["upstream_release_R1"] + r["upstream_release_R2"])

        # Available water = inflow + usable storage above min
        usable_storage = max(0.0, storage - storage_min)
        available = inflow + usable_storage

        # First satisfy minimum commitments
        s4 = float(r["commitment_S4_min"])
        s5 = float(r["commitment_S5_min"])

        # Then satisfy demands (weighted fairness if not enough)
        d1 = float(r["demand_D1"])
        d2 = float(r["demand_D2"])

        base_need = s4 + s5 + d1 + d2

        # Cap on total releases per hour
        total_release_cap = max_release_per_hour

        # If not enough available, scale demands (keep commitments)
        if available < (s4 + s5):
            # extreme shortage: still attempt commitments proportionally
            scale = available / (s4 + s5 + 1e-6)
            s4_out = s4 * scale
            s5_out = s5 * scale
            d1_out = 0.0
            d2_out = 0.0
        else:
            remaining = available - (s4 + s5)
            demand_sum = d1 + d2
            if remaining >= demand_sum:
                d1_out, d2_out = d1, d2
            else:
                # fairness: allocate proportionally
                if demand_sum < 1e-6:
                    d1_out = d2_out = 0.0
                else:
                    d1_out = remaining * (d1 / demand_sum)
                    d2_out = remaining * (d2 / demand_sum)
            s4_out, s5_out = s4, s5

        total_release = s4_out + s5_out + d1_out + d2_out

        # Enforce max release per hour by scaling everything proportionally
        if total_release > total_release_cap:
            scale = total_release_cap / total_release
            s4_out *= scale
            s5_out *= scale
            d1_out *= scale
            d2_out *= scale
            total_release = total_release_cap

        # Update storage: add inflow, subtract release; clamp to max, compute spillage
        storage_next = storage + inflow - total_release
        spillage = 0.0
        if storage_next > storage_max:
            spillage = storage_next - storage_max
            storage_next = storage_max

        # Constraint flags
        violated_min = storage_next < storage_min
        if violated_min:
            storage_next = storage_min  # clamp for MVP display

        rows.append({
            "datetime": ts,
            "inflow_total": inflow,
            "release_D1": d1_out,
            "release_D2": d2_out,
            "release_S4_river": s4_out,
            "release_S5_downstream": s5_out,
            "release_total": total_release,
            "storage_end": storage_next,
            "spillage": spillage,
            "violation_storage_min": bool(violated_min),
        })

        storage = storage_next

    out = pd.DataFrame(rows).set_index("datetime")
    return out

def kpis(inputs: pd.DataFrame, schedule: pd.DataFrame):
    d1 = inputs["demand_D1"].sum()
    d2 = inputs["demand_D2"].sum()
    dem = d1 + d2
    dem_met = schedule["release_D1"].sum() + schedule["release_D2"].sum()
    dem_pct = 100.0 * dem_met / (dem + 1e-6)

    com = inputs["commitment_S4_min"].sum() + inputs["commitment_S5_min"].sum()
    com_met = schedule["release_S4_river"].sum() + schedule["release_S5_downstream"].sum()
    com_pct = 100.0 * com_met / (com + 1e-6)

    spill = schedule["spillage"].sum()
    violations = int(schedule["violation_storage_min"].sum())

    return {
        "Demand met %": dem_pct,
        "Commitments met %": com_pct,
        "Total spillage": spill,
        "Storage-min violations": violations
    }

# -----------------------------
# UI
# -----------------------------
st.title("Reservoir Operation DSS — MVP (UI Prototype)")

with st.sidebar:
    st.header("Run controls")
    horizon = st.selectbox("Horizon", ["24 hours", "72 hours", "5 days (120h)"], index=2)
    horizon_hours = {"24 hours": 24, "72 hours": 72, "5 days (120h)": 120}[horizon]

    st.subheader("Scenario multipliers (What-if)")
    upstream_pct = st.slider("Upstream releases", -30, 30, 0)
    river_pct = st.slider("River inflow", -30, 30, 0)
    demand_pct = st.slider("Demands", -30, 30, 0)
    commit_pct = st.slider("Commitments", -30, 30, 0)

    st.subheader("Reservoir constraints (MVP)")
    storage_start = st.number_input("Storage start", value=5000.0, step=100.0)
    storage_min = st.number_input("Storage min", value=2000.0, step=100.0)
    storage_max = st.number_input("Storage max", value=9000.0, step=100.0)
    max_release = st.number_input("Max release per hour", value=600.0, step=50.0)

    run = st.button("Run (mock optimization)")

# session state
if "inputs" not in st.session_state or run:
    st.session_state["inputs"] = generate_inputs(horizon_hours)
if "schedule" not in st.session_state or run:
    mult = {
        "upstream": 1 + upstream_pct/100.0,
        "river": 1 + river_pct/100.0,
        "demand": 1 + demand_pct/100.0,
        "commitments": 1 + commit_pct/100.0
    }
    st.session_state["mult"] = mult
    st.session_state["schedule"] = pseudo_black_box_schedule(
        st.session_state["inputs"],
        storage_start=storage_start,
        storage_min=storage_min,
        storage_max=storage_max,
        max_release_per_hour=max_release,
        scenario_multipliers=mult
    )

inputs = st.session_state["inputs"]
schedule = st.session_state["schedule"]

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Basin Map", "Schedule", "Publish", "MIS"])

with tab1:
    st.subheader("Single-screen snapshot")
    k = kpis(inputs, schedule)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demand met %", f'{k["Demand met %"]:.1f}')
    c2.metric("Commitments met %", f'{k["Commitments met %"]:.1f}')
    c3.metric("Total spillage", f'{k["Total spillage"]:.1f}')
    c4.metric("Storage-min violations", f'{k["Storage-min violations"]}')

    st.caption("Inputs (current + forecast) and recommended hourly schedule summary.")
    colA, colB = st.columns(2)
    with colA:
        st.write("**Inputs (sample)**")
        st.dataframe(inputs.head(12), use_container_width=True)
    with colB:
        st.write("**Schedule (sample)**")
        st.dataframe(schedule.head(12), use_container_width=True)

with tab2:
    st.subheader("Basin Map (MVP)")
    st.info("MVP limitation: clickable graph can be simulated by selecting an element from dropdown.")
    element = st.selectbox("Select element", ["upstream_release_R1", "upstream_release_R2", "river_inflow_S1", "demand_D1", "demand_D2", "commitment_S4_min", "commitment_S5_min"])
    st.line_chart(inputs[[element]])

with tab3:
    st.subheader("Hourly Release Schedule")
    st.write("Planned releases by branch (mock output).")
    st.line_chart(schedule[["release_D1", "release_D2", "release_S4_river", "release_S5_downstream"]])
    st.dataframe(schedule, use_container_width=True)
    st.download_button("Download schedule (CSV)", schedule.reset_index().to_csv(index=False).encode("utf-8"), file_name="release_schedule.csv")

with tab4:
    st.subheader("Publish & Disseminate (mock)")
    st.write("Choose recipients and generate a shareable message (for WhatsApp/SMS/email).")
    a = st.multiselect("Audience", ["Field officers", "Farmers groups", "Downstream operator", "State control room"], default=["Field officers"])
    ch = st.multiselect("Channels", ["SMS", "WhatsApp", "Email", "API"], default=["WhatsApp"])
    note = st.text_area("Operator note", "Please follow the attached hourly release schedule.")
    if st.button("Generate message"):
        msg = f"""Reservoir Release Schedule (next {len(schedule)} hours)
Audience: {', '.join(a) if a else '—'}
Channel(s): {', '.join(ch) if ch else '—'}

Key KPIs:
- Demand met: {kpis(inputs, schedule)['Demand met %']:.1f}%
- Commitments met: {kpis(inputs, schedule)['Commitments met %']:.1f}%
- Spillage: {kpis(inputs, schedule)['Total spillage']:.1f}

Note: {note}
"""
        st.code(msg)
        st.success("Message generated (MVP).")

with tab5:
    st.subheader("MIS Dashboard (MVP)")
    # Build a simple fact table
    fact = schedule.copy()
    fact["demand_D1"] = inputs["demand_D1"]
    fact["demand_D2"] = inputs["demand_D2"]
    fact["commit_S4"] = inputs["commitment_S4_min"]
    fact["commit_S5"] = inputs["commitment_S5_min"]
    fact["demand_met_D1_pct"] = 100.0 * fact["release_D1"] / (fact["demand_D1"] + 1e-6)
    fact["demand_met_D2_pct"] = 100.0 * fact["release_D2"] / (fact["demand_D2"] + 1e-6)
    fact["commit_met_S4_pct"] = 100.0 * fact["release_S4_river"] / (fact["commit_S4"] + 1e-6)
    fact["commit_met_S5_pct"] = 100.0 * fact["release_S5_downstream"] / (fact["commit_S5"] + 1e-6)

    st.write("**Hourly fact table (sample fields for MIS)**")
    st.dataframe(fact.reset_index().head(24), use_container_width=True)
    st.download_button("Download MIS fact table (CSV)", fact.reset_index().to_csv(index=False).encode("utf-8"), file_name="mis_fact_table.csv")
