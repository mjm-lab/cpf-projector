import math
from datetime import date, datetime
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# ==============================
# Aesthetics & Page Config
# ==============================
st.set_page_config(
    page_title="CPF Projector",
    page_icon="🧮",
    layout="wide"
)

st.markdown(
    """
    <style>
    .metric-card {
        border-radius: 16px;
        padding: 16px 18px;
        border: 1px solid rgba(0,0,0,0.08);
        background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .small-muted { color: #6b7280; font-size: 12px; }
    .section-title { font-weight: 700; font-size: 20px; margin-top: 8px; }
    .pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        background: #ecfeff;
        color: #0369a1;
        font-size: 12px;
        margin-left: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================
# Assumptions / Policy Tables
# ==============================
BASE_INT = {"OA": 0.025, "SA": 0.04, "MA": 0.04, "RA": 0.04}
ERS_FACTOR_DEFAULT = 2.0  # default factor: 2×FRS

EXTRA_BELOW_55 = {"pool": 60000.0, "oa_cap": 20000.0, "tier1_rate": 0.01}
EXTRA_55_PLUS = {
    "tier1_amount": 30000.0, "tier1_rate": 0.02,
    "tier2_amount": 30000.0, "tier2_rate": 0.01,
    "oa_cap": 20000.0
}

def ow_ceiling_monthly(year: int):
    if year <= 2024:
        return 6800.0
    elif year == 2025:
        return 7400.0
    else:  # 2026+
        return 8000.0

ANNUAL_TW_CEILING = 102000.0

FRS_KNOWN = {2025: 213000.0, 2026: 220400.0, 2027: 228200.0}
FRS_growth_pct_default = 0.035

BHS_KNOWN = {2025: 75500.0}
BHS_growth_pct_default = 0.05

# ---- Cohort BHS (fixed at 65) ----
COHORT_BHS_BY_YEAR65 = {
    2025: 75500.0, 2024: 71500.0, 2023: 68500.0, 2022: 66000.0,
    2021: 63000.0, 2020: 60000.0, 2019: 57200.0, 2018: 54500.0,
    2017: 52000.0, 2016: 49800.0
}

# ==============================
# CPF LIFE payout coefficients (monthly)
# payout = a + b * RA_at_65 (when start age = 65)
# ==============================
CPF_LIFE_COEFFS = {
    "M": {
        "Standard_65":   {"a": 72.29412442, "b": 0.005182694},
        "Escalating_65": {"a": 58.33903251, "b": 0.004085393},
        "Basic_65":      {"a": 66.1522067,  "b": 0.004734576},
    },
    "F": {
        "Standard_65":   {"a": 68.51952457, "b": 0.004825065},
        "Escalating_65": {"a": 54.50151582, "b": 0.003718627},
        "Basic_65":      {"a": 66.44720836, "b": 0.004566059},
    },
}
CPF_LIFE_DEFERRAL_PER_YEAR = 0.07  # ~7% more payout per year of deferral beyond 65
ESCALATING_RATE = 0.02             # +2% per year escalation after start
BASIC_PREMIUM_FRAC = 0.10          # fraction of RA paid as premium at start for Basic (approx)

# Defaults so project() can be reused
m_topup_month = 1
topup_OA = topup_SA_RA = topup_MA = 0.0
# Removed: yearly MA/OA deduction globals (simplified)

# ==============================
# Helper functions
# ==============================
def _label_year_age(y, yearly_df):
    try:
        age = int(yearly_df.set_index("Year").loc[y, "Age_end"])
        return f"{y} (Age {age})"
    except KeyError:
        return str(y)

def age_at(dob: date, year: int, month: int):
    d = date(year, month, 28)  # approx month-end
    return d.year - dob.year - ((d.month, d.day) < (dob.month, d.day))

def get_alloc_for_age(age):
    for row in [
        {"min_age": 0,  "max_age": 35, "total": 0.37,  "OA": 0.23,  "SA_RA": 0.06,  "MA": 0.08},
        {"min_age": 35, "max_age": 45, "total": 0.37,  "OA": 0.21,  "SA_RA": 0.07,  "MA": 0.09},
        {"min_age": 45, "max_age": 50, "total": 0.37,  "OA": 0.19,  "SA_RA": 0.08,  "MA": 0.10},
        {"min_age": 50, "max_age": 55, "total": 0.37,  "OA": 0.15,  "SA_RA": 0.115, "MA": 0.105},
        {"min_age": 55, "max_age": 60, "total": 0.325, "OA": 0.12,  "SA_RA": 0.10,  "MA": 0.105},
        {"min_age": 60, "max_age": 65, "total": 0.235, "OA": 0.035, "SA_RA": 0.095, "MA": 0.105},
        {"min_age": 65, "max_age": 70, "total": 0.165, "OA": 0.01,  "SA_RA": 0.085, "MA": 0.07},
        {"min_age": 70, "max_age": 200,"total": 0.125, "OA": 0.005, "SA_RA": 0.07,  "MA": 0.05},
    ]:
        if (age >= row["min_age"]) and (age < row["max_age"]):
            return row
    return {"total":0.0,"OA":0.0,"SA_RA":0.0,"MA":0.0}

def get_frs_for_cohort(year55: int, frs_growth_pct: float):
    if year55 in FRS_KNOWN:
        return FRS_KNOWN[year55]
    last_year = max(FRS_KNOWN.keys())
    frs = FRS_KNOWN[last_year]
    for _ in range(last_year+1, year55+1):
        frs *= (1 + frs_growth_pct)
    return round(frs, 2)

def get_frs_for_year(year: int, frs_growth_pct: float) -> float:
    if year in FRS_KNOWN:
        return FRS_KNOWN[year]
    last_year = max(FRS_KNOWN.keys())
    frs = FRS_KNOWN[last_year]
    for _ in range(last_year + 1, year + 1):
        frs *= (1 + frs_growth_pct)
    return round(frs, 2)

def get_bhs_for_year_with_cohort(dob: date, year: int, bhs_growth_pct: float):
    """Before 65: prevailing national BHS.
       From 65th-birthday year onward: fixed cohort BHS at age 65.
    """
    year65 = dob.year + 65
    if year < year65:
        if year in BHS_KNOWN:
            return BHS_KNOWN[year]
        bhs = BHS_KNOWN[2025]
        for _ in range(2026, year + 1):
            bhs *= (1 + bhs_growth_pct)
        return round(bhs, 2)
    else:
        if year65 in BHS_KNOWN:
            cohort_bhs = BHS_KNOWN[year65]
        elif year65 >= 2026:
            cohort_bhs = BHS_KNOWN[2025]
            for _ in range(2026, year65 + 1):
                cohort_bhs *= (1 + bhs_growth_pct)
        else:
            if year65 >= 2024:
                cohort_bhs = COHORT_BHS_BY_YEAR65[2024]
            elif year65 >= 2023:
                cohort_bhs = COHORT_BHS_BY_YEAR65[2023]
            elif year65 >= 2022:
                cohort_bhs = COHORT_BHS_BY_YEAR65[2022]
            elif year65 >= 2021:
                cohort_bhs = COHORT_BHS_BY_YEAR65[2021]
            elif year65 >= 2020:
                cohort_bhs = COHORT_BHS_BY_YEAR65[2020]
            elif year65 >= 2019:
                cohort_bhs = COHORT_BHS_BY_YEAR65[2019]
            elif year65 >= 2018:
                cohort_bhs = COHORT_BHS_BY_YEAR65[2018]
            elif year65 >= 2017:
                cohort_bhs = COHORT_BHS_BY_YEAR65[2017]
            else:
                cohort_bhs = COHORT_BHS_BY_YEAR65[2016]
        return round(cohort_bhs, 2)

# ---- MediShield Life premium (GST inc.) by ANB ----
def get_mshl_premium_by_anb(anb: int) -> float:
    bands = [
        ((1,20), 200), ((21,30), 295), ((31,40), 503), ((41,50), 637),
        ((51,60), 903), ((61,65), 1131), ((66,70), 1326), ((71,73), 1643),
        ((74,75), 1816), ((76,78), 2027), ((79,80), 2187), ((81,83), 2303),
        ((84,85), 2616), ((86,88), 2785), ((89,90), 2785), ((91,200), 2826)
    ]
    for (lo, hi), prem in bands:
        if lo <= anb <= hi:
            return float(prem)
    return 0.0

# -------- Extra interest BEFORE CPF LIFE starts --------
def compute_extra_interest_distribution(age, oa, sa, ma, ra):
    ei = {"OA": 0.0, "SA": 0.0, "MA": 0.0, "RA": 0.0}
    if age < 55:
        remaining = EXTRA_BELOW_55["pool"]
        take_oa = min(remaining, min(oa, EXTRA_BELOW_55["oa_cap"]))
        if take_oa > 0:
            ei["OA"] += take_oa * EXTRA_BELOW_55["tier1_rate"]
            remaining -= take_oa
        take_sa = min(remaining, sa)
        if take_sa > 0:
            ei["SA"] += take_sa * EXTRA_BELOW_55["tier1_rate"]
            remaining -= take_sa
        take_ma = min(remaining, ma)
        if take_ma > 0:
            ei["MA"] += take_ma * EXTRA_BELOW_55["tier1_rate"]
            remaining -= take_ma
    else:
        t1 = 30000.0; t2 = 30000.0
        oa_cap = 20000.0
        r1 = min(t1, ra); t1 -= r1
        o1 = min(t1, min(oa, oa_cap)); t1 -= o1
        s1 = min(t1, sa); t1 -= s1
        m1 = min(t1, ma); t1 -= m1
        ei["RA"] += r1 * 0.02; ei["OA"] += o1 * 0.02; ei["SA"] += s1 * 0.02; ei["MA"] += m1 * 0.02

        r2 = min(t2, max(0.0, ra - r1)); t2 -= r2
        o2 = min(t2, max(0.0, min(oa, oa_cap) - o1)); t2 -= o2
        s2 = min(t2, max(0.0, sa - s1)); t2 -= s2
        m2 = min(t2, max(0.0, ma - m1)); t2 -= m2
        ei["RA"] += r2 * 0.01; ei["OA"] += o2 * 0.01; ei["SA"] += s2 * 0.01; ei["MA"] += m2 * 0.01
    return ei

# -------- Extra interest AFTER CPF LIFE starts --------
def compute_extra_interest_distribution_after_cpf_life(age, oa, sa, ma, ra, premium_remaining):
    if age < 55:
        return compute_extra_interest_distribution(age, oa, sa, ma, ra)
    tier1 = min(ra, 30000.0) * 0.02
    tier2 = min(max(ra - 30000.0, 0.0), 30000.0) * 0.01
    return {"OA": 0.0, "SA": 0.0, "MA": 0.0, "RA": tier1 + tier2}

def spill_from_ma(age, ma_end, bhs, sa, oa, ra, frs_for_cohort):
    if ma_end <= bhs:
        return ma_end, sa, oa, ra
    excess = ma_end - bhs
    ma_end = bhs
    if age < 55:
        space_sa = max(0.0, frs_for_cohort - sa)
        to_sa = min(excess, space_sa); sa += to_sa; excess -= to_sa
        oa += excess; excess = 0.0
    else:
        space_ra = max(0.0, frs_for_cohort - ra)
        to_ra = min(excess, space_ra); ra += to_ra; excess -= to_ra
        oa += excess; excess = 0.0
    return ma_end, sa, oa, ra

def transfer_to_ra_at_55(age_this_month, sa, oa, ra, transfer_target, cohort_frs):
    if age_this_month < 55:
        return sa, oa, ra, 0.0

    moved_capital = 0.0
    total_pre = oa + sa

    # Case 1: below FRS
    if total_pre < cohort_frs:
        keep = min(5000.0, total_pre)
        if oa < keep:
            need = keep - oa
            take = min(sa, need)
            sa -= take
            oa += take
        if sa > 0:
            ra += sa
            moved_capital += sa
            sa = 0.0
        excess_oa = max(0.0, oa - keep)
        if excess_oa > 0:
            ra += excess_oa
            moved_capital += excess_oa
            oa -= excess_oa
        return sa, oa, ra, moved_capital

    # Case 2: at/above FRS
    needed = max(0.0, transfer_target - ra)
    if needed <= 0:
        oa += sa
        sa = 0.0
        return sa, oa, ra, moved_capital

    take_sa = min(needed, sa)
    ra += take_sa
    sa -= take_sa
    moved_capital += take_sa
    needed -= take_sa

    if needed > 0:
        take_oa = min(needed, oa)
        ra += take_oa
        oa -= take_oa
        moved_capital += take_oa

    oa += sa
    sa = 0.0
    return sa, oa, ra, moved_capital


# ==============================
# IP helpers (CSV schema)
# ==============================
REQUIRED_IP_COLS = ["Insurer", "Ward Class", "Plan Type", "Plan Name", "Age", "Premium in MA", "Premium in Cash"]

def try_load_ip_csv(uploaded_file):
    """
    Try to load from uploaded file first, else from local ./ip_premiums.csv, else /mnt/data/ip_premiums.csv.
    Returns (df or None, error_message or None)
    """
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df, None
        except Exception as e:
            return None, f"Could not read the uploaded CSV: {e}"
    for p in ["ip_premiums.csv", "/mnt/data/ip_premiums.csv"]:
        try:
            df = pd.read_csv(p)
            return df, None
        except Exception:
            continue
    return None, "ip_premiums.csv not found (search paths: app folder or /mnt/data). Upload it to enable IP."

def validate_ip_df(df):
    missing = [c for c in REQUIRED_IP_COLS if c not in df.columns]
    if missing:
        return f"ip_premiums.csv is missing required columns: {missing}"
    return None

def _ip_lookup_amounts(ip_df, insurer, ward_class, plan_name, plan_type, anb):
    """Return (ma_amount, cash_amount) for given selection and ANB. 0 if not found."""
    if ip_df is None or any(x in (None, "", "(None)") for x in [insurer, ward_class, plan_name]):
        return 0.0, 0.0
    df = ip_df[
        (ip_df["Insurer"] == insurer) &
        (ip_df["Ward Class"] == ward_class) &
        (ip_df["Plan Type"] == plan_type) &
        (ip_df["Plan Name"] == plan_name) &
        (ip_df["Age"].astype(int) == int(anb))
    ]
    if df.empty:
        return 0.0, 0.0
    row = df.iloc[0]
    ma = float(row.get("Premium in MA", 0.0))
    cash = float(row.get("Premium in Cash", 0.0))
    return ma, cash


# ==============================
# Core projection
# ==============================
def project(
    name: str,
    dob_str: str,
    gender: str,
    start_year: int,
    years: int,
    monthly_income: float,
    annual_bonus: float,
    salary_growth_pct: float,
    bonus_growth_pct: float,
    opening_balances: dict,
    frs_growth_pct: float,
    bhs_growth_pct: float,
    ers_factor: float = ERS_FACTOR_DEFAULT,
    retirement_age: int = 65,
    # CPF LIFE controls
    include_cpf_life: bool = True,
    cpf_life_plan: str = "Standard",  # "Standard","Escalating","Basic"
    payout_start_age: int = 65,
    # Top-up stop controls
    topup_stop_option: str = "No limit",
    topup_years_limit: int = 0,
    topup_stop_age: int = 120,
    # Long-term care insurance (deduct from MA)
    include_ltci: bool = False,
    ltci_ma_premium: float = 0.0,
    ltci_pay_until_age: int = 67,
    ltci_month: int = 1,
    # Integrated Shield Plan
    ip_enabled: bool = False,
    ip_df: pd.DataFrame | None = None,
    ip_insurer: str | None = None,
    ip_ward: str | None = None,
    ip_base_plan: str | None = "(None)",
    ip_rider: str | None = "(None)",
    insurance_month: int = 1,   # <-- single month for MSHL & IP
    # OA withdrawals (55+)
    withdraw_oa_enabled: bool = False,
    withdraw_oa_monthly_amount: float = 0.0,
    withdraw_oa_start_age: int = 55,
    withdraw_oa_end_age: int = 120,
    # Housing loan (OA monthly) — start is implicit (now); stop by end age
    house_enabled: bool = False,
    house_monthly_amount: float = 0.0,
    house_end_age: int = 120,
):
    dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
    bal = opening_balances.copy()
    monthly_rows = []
    year55 = dob.year + 55
    cohort_frs = get_frs_for_cohort(year55, frs_growth_pct)
    cohort_ers = cohort_frs * ers_factor
    ra_transfer_target = cohort_ers

    # Track capital and CPF LIFE
    ra_capital = bal.get("RA", 0.0)
    prev_bal_for_interest = bal.copy()

    # CPF LIFE derived values
    ra_at_65_value = None
    premium_pool = 0.0
    ra_savings_for_basic = 0.0
    cpf_life_started = False
    monthly_start_payout = None
    psa_month = dob.month  # CPF LIFE start in birth month of chosen age

    # OA run-out trackers
    oa_runs_out_age = None
    oa_runs_out_year = None
    oa_runs_out_month = None

    house_runs_out_age = None
    house_runs_out_year = None
    house_runs_out_month = None

    start_year_sched = None

    for year in range(start_year, start_year + years):
        yr_index = year - start_year
        monthly_income_y = monthly_income * ((1 + salary_growth_pct) ** yr_index)
        annual_bonus_y = annual_bonus * ((1 + bonus_growth_pct) ** yr_index)

        bhs_this_year = get_bhs_for_year_with_cohort(dob, year, bhs_growth_pct)
        ow_cap = ow_ceiling_monthly(year)

        prevailing_frs_year = get_frs_for_year(year, frs_growth_pct)
        prevailing_ers_year = prevailing_frs_year * ers_factor

        ow_subject_per_mo = min(monthly_income_y, ow_cap)
        ow_used_ytd = 0.0  # for bonus cap

        for month in range(1, 13):
            age = age_at(dob, year, month)
            alloc = get_alloc_for_age(age)

            # Snapshot RA at exact age-65 birth month (before any CPF LIFE premium deduction)
            if (age == 65) and (month == dob.month) and (ra_at_65_value is None):
                ra_at_65_value = bal["RA"]

            # RA transfer at 55 birth month
            if age >= 55 and bal["SA"] > 0:
                if (year > year55) or (year == year55 and month >= dob.month):
                    bal["SA"], bal["OA"], bal["RA"], moved = transfer_to_ra_at_55(
                        age, bal["SA"], bal["OA"], bal["RA"], ra_transfer_target, cohort_frs
                    )
                    ra_capital += moved

            # CPF LIFE: initialise payouts & deduct premium at start month
            if include_cpf_life and (age == payout_start_age) and (month == psa_month) and not cpf_life_started:
                if ra_at_65_value is None:
                    ra_at_65_value = bal["RA"]
                coeff_key = "M" if gender == "M" else "F"
                plan_map = {"Standard":"Standard_65","Escalating":"Escalating_65","Basic":"Basic_65"}
                coeff = CPF_LIFE_COEFFS[coeff_key][plan_map[cpf_life_plan]]
                payout_65 = coeff["a"] + coeff["b"] * ra_at_65_value
                monthly_start_payout = payout_65 * ((1 + CPF_LIFE_DEFERRAL_PER_YEAR) ** (payout_start_age - 65))
                # Premium deduction
                if cpf_life_plan in ("Standard","Escalating"):
                    premium = bal["RA"]
                    premium_pool += premium
                    bal["RA"] = 0.0
                else:  # Basic
                    premium = bal["RA"] * BASIC_PREMIUM_FRAC
                    premium_pool += premium
                    bal["RA"] -= premium
                    ra_savings_for_basic = bal["RA"]
                cpf_life_started = True
                start_year_sched = year

            # --- Monthly OW contributions ---
            working = age < retirement_age
            if working:
                to_MA = ow_subject_per_mo * alloc["MA"]
                to_SA_RA = ow_subject_per_mo * alloc["SA_RA"]
                to_OA = ow_subject_per_mo * alloc["OA"]
            else:
                to_MA = to_SA_RA = to_OA = 0.0

            if age >= 55:
                space_ra = max(0.0, cohort_frs - bal["RA"])
                to_RA = min(to_SA_RA, space_ra)
                to_OA += (to_SA_RA - to_RA)
                to_SA = 0.0
            else:
                to_SA = to_SA_RA
                to_RA = 0.0

            bal["MA"] += to_MA
            bal["OA"] += to_OA
            bal["SA"] += to_SA
            bal["RA"] += to_RA
            if age >= 55:
                ra_capital += to_RA  # contributions to RA are capital

            income_used_this_month = ow_subject_per_mo if working else 0.0
            ow_used_ytd += income_used_this_month

            # --- Annual bonus in December (AW) ---
            aw_used_this_dec = 0.0
            if month == 12 and working and (annual_bonus_y > 0):
                aw_ceiling_rem = max(0.0, ANNUAL_TW_CEILING - ow_used_ytd)
                aw_subject = min(annual_bonus_y, aw_ceiling_rem)
                aw_used_this_dec = aw_subject
                to_MA_aw = aw_subject * alloc["MA"]
                to_SA_RA_aw = aw_subject * alloc["SA_RA"]
                to_OA_aw = aw_subject * alloc["OA"]
                if age >= 55:
                    space_ra2 = max(0.0, cohort_frs - bal["RA"])
                    to_RA_aw = min(to_SA_RA_aw, space_ra2)
                    to_OA_aw += (to_SA_RA_aw - to_RA_aw)
                    to_SA_aw = 0.0
                else:
                    to_SA_aw = to_SA_RA_aw
                    to_RA_aw = 0.0
                bal["MA"] += to_MA_aw
                bal["OA"] += to_OA_aw
                bal["SA"] += to_SA_aw
                bal["RA"] += to_RA_aw
                if age >= 55:
                    ra_capital += to_RA_aw

            # --- MediShield Life premium (Age Next Birthday) ---
            mshl_paid_this_month = 0.0
            mshl_nominal_this_month = 0.0
            if month == insurance_month:
                anb = age + 1
                prem = get_mshl_premium_by_anb(anb)
                mshl_nominal_this_month = prem
                pay = min(bal["MA"], prem)
                bal["MA"] -= pay
                mshl_paid_this_month = pay

            # --- Long-term care insurance premium (from MA) ---
            ltci_paid_this_month = 0.0
            if include_ltci and (month == ltci_month) and (age <= ltci_pay_until_age):
                ltci_paid_this_month = min(bal["MA"], float(ltci_ma_premium))
                bal["MA"] -= ltci_paid_this_month

            # --- Integrated Shield Plan premiums (CSV) ---
            ip_base_ma_paid = 0.0
            ip_base_cash = 0.0
            ip_rider_cash = 0.0
            ip_base_ma_nominal = 0.0
            ip_base_cash_nominal = 0.0
            ip_rider_cash_nominal = 0.0

            if ip_enabled and (ip_df is not None) and (month == insurance_month):
                anb_ip = age + 1

                # Base plan (MA + Cash)
                if ip_base_plan and ip_base_plan != "(None)":
                    base_ma, base_cash = _ip_lookup_amounts(
                        ip_df, ip_insurer, ip_ward, ip_base_plan, "Base", anb_ip
                    )
                    ip_base_ma_nominal = base_ma
                    ip_base_cash_nominal = base_cash

                    if base_ma > 0:
                        pay_ma = min(bal["MA"], base_ma)
                        bal["MA"] -= pay_ma
                        ip_base_ma_paid = pay_ma
                        ip_base_cash += max(0.0, base_ma - pay_ma)  # MA shortfall paid in cash
                    ip_base_cash += base_cash  # plan's cash part

                # Rider (cash only)
                if ip_rider and ip_rider != "(None)":
                    _, rider_cash = _ip_lookup_amounts(
                        ip_df, ip_insurer, ip_ward, ip_rider, "Rider", anb_ip
                    )
                    ip_rider_cash_nominal = rider_cash
                    ip_rider_cash += rider_cash

            # --- Top-ups (once a year, chosen month) ---
            topup_oa_applied = 0.0
            topup_sa_applied = 0.0
            topup_ra_applied = 0.0
            topup_ma_applied = 0.0

            if month == m_topup_month:
                allow_topup = True
                if topup_stop_option == "After N years":
                    if (year - start_year) >= int(topup_years_limit):
                        allow_topup = False
                elif topup_stop_option == "After age X":
                    if age > int(topup_stop_age):
                        allow_topup = False

                if allow_topup:
                    if float(topup_OA) > 0.0:
                        add = float(topup_OA)
                        bal["OA"] += add
                        topup_oa_applied = add

                    if float(topup_MA) > 0.0:
                        room_ma = max(0.0, bhs_this_year - bal["MA"])
                        add = min(float(topup_MA), room_ma)
                        if add > 0:
                            bal["MA"] += add
                            topup_ma_applied = add

                    if float(topup_SA_RA) > 0.0:
                        if age < 55:
                            room_sa = max(0.0, cohort_frs - bal["SA"])
                            add = min(float(topup_SA_RA), room_sa)
                            if add > 0:
                                bal["SA"] += add
                                topup_sa_applied = add
                        else:
                            room_ra_capital = max(0.0, prevailing_ers_year - ra_capital)
                            add = min(float(topup_SA_RA), room_ra_capital)
                            if add > 0:
                                bal["RA"] += add
                                ra_capital += add
                                topup_ra_applied = add

            # --- Monthly OA withdrawal (55+) ---
            oa_withdrawal_paid = 0.0
            if (
                withdraw_oa_enabled
                and (age >= max(55, int(withdraw_oa_start_age)))
                and (age <= int(withdraw_oa_end_age))
                and float(withdraw_oa_monthly_amount) > 0.0
            ):
                amt = float(withdraw_oa_monthly_amount)
                pay = min(bal["OA"], amt)
                bal["OA"] -= pay
                oa_withdrawal_paid = pay

                if (oa_runs_out_age is None) and (bal["OA"] <= 1e-6):
                    oa_runs_out_age = int(age)
                    oa_runs_out_year = int(year)
                    oa_runs_out_month = int(month)

            # --- Monthly housing loan deduction from OA (simple, up to end age) ---
            house_paid_this_month = 0.0
            if (
                house_enabled
                and (age <= int(house_end_age))
                and float(house_monthly_amount) > 0.0
            ):
                pay = min(bal["OA"], float(house_monthly_amount))
                bal["OA"] -= pay
                house_paid_this_month = pay

                # First time OA hits zero during housing period
                if (house_runs_out_age is None) and (bal["OA"] <= 1e-6):
                    house_runs_out_age = int(age)
                    house_runs_out_year = int(year)
                    house_runs_out_month = int(month)

            # --- Monthly interest (on previous month-end) ---
            base_int_OA = prev_bal_for_interest.get("OA", 0.0) * (BASE_INT["OA"] / 12.0)
            base_int_SA = prev_bal_for_interest.get("SA", 0.0) * (BASE_INT["SA"] / 12.0)
            base_int_MA = prev_bal_for_interest.get("MA", 0.0) * (BASE_INT["MA"] / 12.0)

            if cpf_life_started and cpf_life_plan in ("Standard", "Escalating"):
                base_int_RA = 0.0
            else:
                base_int_RA = prev_bal_for_interest.get("RA", 0.0) * (BASE_INT["RA"] / 12.0)

            bal["OA"] += base_int_OA
            if age < 55:
                bal["SA"] += base_int_SA
            else:
                bal["RA"] += base_int_SA
            bal["MA"] += base_int_MA
            bal["RA"] += base_int_RA

            if cpf_life_started:
                ei = compute_extra_interest_distribution_after_cpf_life(
                    age,
                    prev_bal_for_interest.get("OA", 0.0),
                    prev_bal_for_interest.get("SA", 0.0),
                    prev_bal_for_interest.get("MA", 0.0),
                    bal["RA"],
                    premium_remaining=premium_pool
                )
            else:
                ei = compute_extra_interest_distribution(
                    age,
                    prev_bal_for_interest.get("OA", 0.0),
                    prev_bal_for_interest.get("SA", 0.0),
                    prev_bal_for_interest.get("MA", 0.0),
                    prev_bal_for_interest.get("RA", 0.0),
                )

            # Route extra interest
            if age < 55:
                bal["SA"] += ei["OA"] / 12.0
                bal["SA"] += ei["SA"] / 12.0
                bal["MA"] += ei["MA"] / 12.0
            else:
                bal["RA"] += ei["OA"] / 12.0
                if not cpf_life_started:
                    bal["MA"] += ei["MA"] / 12.0
            bal["RA"] += ei["RA"] / 12.0

            # --- CPF LIFE payouts (Basic only draws from RA savings) ---
            monthly_cpf_payout = 0.0
            if include_cpf_life and cpf_life_started:
                years_since_start = (year - start_year_sched) if start_year_sched is not None else 0
                current_monthly_payout = monthly_start_payout
                if cpf_life_plan == "Escalating" and years_since_start > 0:
                    current_monthly_payout *= ((1 + ESCALATING_RATE) ** years_since_start)
                monthly_cpf_payout = current_monthly_payout
                if cpf_life_plan == "Basic":
                    draw = min(bal["RA"], monthly_cpf_payout)
                    bal["RA"] -= draw

            # --- Enforce BHS and spillovers ---
            if cpf_life_started:
                if bal["MA"] > bhs_this_year:
                    excess = bal["MA"] - bhs_this_year
                    bal["MA"] = bhs_this_year
                    bal["OA"] += excess
            else:
                ra_before = bal["RA"]
                bal["MA"], bal["SA"], bal["OA"], bal["RA"] = spill_from_ma(
                    age, bal["MA"], bhs_this_year, bal["SA"], bal["OA"], bal["RA"], cohort_frs
                )
                ra_spill = max(0.0, bal["RA"] - ra_before); ra_capital += ra_spill

            # Ensure SA closed after 55
            if age >= 55 and bal["SA"] > 0:
                bal["RA"] += bal["SA"]; bal["SA"] = 0.0

            # --- Save monthly row ---
            monthly_rows.append({
                "Year": year, "Month": month, "Age": age,
                "BHS": bhs_this_year, "FRS_cohort": cohort_frs, "ERS_cohort": cohort_ers,
                "RA_target55_multiple": ers_factor, "OW_cap": ow_cap,
                "Income_used": income_used_this_month, "Bonus_used_dec": aw_used_this_dec,
                "OA": bal["OA"], "SA": bal["SA"], "MA": bal["MA"], "RA": bal["RA"],
                "BaseInt_OA": base_int_OA, "BaseInt_SA": base_int_SA, "BaseInt_MA": base_int_MA, "BaseInt_RA": base_int_RA,
                "ExtraInt_OA": ei["OA"]/12.0, "ExtraInt_SA": ei["SA"]/12.0, "ExtraInt_MA": ei["MA"]/12.0, "ExtraInt_RA": ei["RA"]/12.0,
                "RA_capital": ra_capital, "Prevailing_ERS": prevailing_ers_year,
                "CPF_LIFE_started": int(cpf_life_started), "CPF_LIFE_monthly_payout": monthly_cpf_payout,

                # Insurance flows
                "MSHL_Premium_Paid": mshl_paid_this_month,
                "MSHL_Premium_Nominal": mshl_nominal_this_month,
                "LTCI_MA_Premium_Paid": ltci_paid_this_month,

                "IP_Base_MA_Paid": ip_base_ma_paid,
                "IP_Base_Cash": ip_base_cash,
                "IP_Rider_Cash": ip_rider_cash,

                # Nominal (for stacked chart by intended source)
                "IP_Base_MA_Nominal": ip_base_ma_nominal,
                "IP_Base_Cash_Nominal": ip_base_cash_nominal,
                "IP_Rider_Cash_Nominal": ip_rider_cash_nominal,

                # Top-ups actually applied this month
                "Topup_OA_Applied": topup_oa_applied,
                "Topup_SA_Applied": topup_sa_applied,
                "Topup_RA_Applied": topup_ra_applied,
                "Topup_MA_Applied": topup_ma_applied,

                # Withdrawals / Housing
                "OA_Withdrawal_Paid": oa_withdrawal_paid,
                "Housing_OA_Paid": house_paid_this_month,
            })

            prev_bal_for_interest = {"OA": bal["OA"], "SA": bal["SA"], "MA": bal["MA"], "RA": bal["RA"]}

    monthly_df = pd.DataFrame(monthly_rows)

    # ----- Build yearly roll-up -----
    yearly = []
    for y, grp in monthly_df.groupby("Year"):
        end_row = grp.sort_values("Month").iloc[-1]
        total_base_int = grp[["BaseInt_OA", "BaseInt_SA", "BaseInt_MA", "BaseInt_RA"]].sum().sum()
        total_extra_int = grp[["ExtraInt_OA", "ExtraInt_SA", "ExtraInt_MA", "ExtraInt_RA"]].sum().sum()
        yearly.append({
            "Year": y, "Age_end": int(end_row["Age"]),
            "End_OA": end_row["OA"], "End_SA": end_row["SA"], "End_MA": end_row["MA"], "End_RA": end_row["RA"],
            "RA_capital_end": float(end_row["RA_capital"]), "Prevailing_ERS": float(end_row["Prevailing_ERS"]),
            "Total_Base_Interest": total_base_int, "Total_Extra_Interest": total_extra_int,
            "OW_subject_total": grp["Income_used"].sum(), "AW_subject_total": grp["Bonus_used_dec"].sum(),
            "CPF_LIFE_Annual_Payout": grp["CPF_LIFE_monthly_payout"].sum(),

            # Insurance annuals (actual paid)
            "MSHL_Annual": grp["MSHL_Premium_Paid"].sum(),
            "LTCI_Premium_Annual_MA": grp["LTCI_MA_Premium_Paid"].sum(),
            "IP_Base_MA_Annual": grp["IP_Base_MA_Paid"].sum(),
            "IP_Base_Cash_Annual": grp["IP_Base_Cash"].sum(),
            "IP_Rider_Cash_Annual": grp["IP_Rider_Cash"].sum(),

            # Nominal (for chart)
            "MSHL_Nominal": grp["MSHL_Premium_Nominal"].sum(),
            "IP_Base_MA_Nominal_Annual": grp["IP_Base_MA_Nominal"].sum(),
            "IP_Base_Cash_Nominal_Annual": grp["IP_Base_Cash_Nominal"].sum(),
            "IP_Rider_Cash_Nominal_Annual": grp["IP_Rider_Cash_Nominal"].sum(),

            # Top-up annual totals
            "Topup_OA_Annual": grp["Topup_OA_Applied"].sum(),
            "Topup_SA_Annual": grp["Topup_SA_Applied"].sum(),
            "Topup_RA_Annual": grp["Topup_RA_Applied"].sum(),
            "Topup_MA_Annual": grp["Topup_MA_Applied"].sum(),

            # OA withdrawal + Housing annual totals
            "OA_Withdrawal_Annual": grp["OA_Withdrawal_Paid"].sum(),
            "Housing_OA_Annual": grp["Housing_OA_Paid"].sum(),
        })
    yearly_df = pd.DataFrame(yearly)

    # ----- CPF LIFE payout schedule & bequest track -----
    cpf_life_df = None
    bequest_df = None
    if include_cpf_life and ('monthly_start_payout' in locals()) and (monthly_start_payout is not None):
        # Determine start year calendar
        psa_rows = monthly_df[(monthly_df["Age"] == payout_start_age) & (monthly_df["Month"] == psa_month)]
        if not psa_rows.empty:
            start_year_sched = int(psa_rows.iloc[0]["Year"])
        else:
            tmp = yearly_df[yearly_df["Age_end"] >= payout_start_age]
            start_year_sched = int(tmp.iloc[0]["Year"]) if not tmp.empty else int(yearly_df["Year"].min())

        sched = []
        beq_rows = []

        beq_premium = float(premium_pool)  # premium (no credited interest)
        beq_ra_savings = float(ra_savings_for_basic if cpf_life_plan == "Basic" else 0.0)

        def _annual_ra_interest(balance: float) -> float:
            if balance <= 0:
                return 0.0
            base = balance * 0.04
            tier1 = min(balance, 30000.0) * 0.02
            tier2 = min(max(balance - 30000.0, 0.0), 30000.0) * 0.01
            return base + tier1 + tier2

        last_year = int(yearly_df["Year"].max())

        for y in range(start_year_sched, last_year + 1):
            years_since = y - start_year_sched
            monthly = monthly_start_payout
            if cpf_life_plan == "Escalating":
                monthly *= ((1 + ESCALATING_RATE) ** max(0, years_since))
            annual = monthly * 12.0

            sched.append({"Year": y, "Monthly_Payout": monthly, "Annual_Payout": annual})

            if cpf_life_plan == "Basic":
                beq_ra_savings += _annual_ra_interest(beq_ra_savings)
                draw_from_ra = min(beq_ra_savings, annual)
                beq_ra_savings -= draw_from_ra
                from_pool = max(0.0, annual - draw_from_ra)
                beq_premium = max(0.0, beq_premium - from_pool)
            else:
                beq_premium = max(0.0, beq_premium - annual)

            beq_rows.append({
                "Year": y,
                "Bequest_Remaining": beq_premium + beq_ra_savings,
                "Unused_Premium": beq_premium,
                "RA_Savings_Remaining": beq_ra_savings
            })

        cpf_life_df = pd.DataFrame(sched)
        bequest_df = pd.DataFrame(beq_rows)

    # meta & OA/housing run-out info
    meta = {
        "monthly_start_payout": monthly_start_payout,
        "oa_withdrawal_enabled": bool(withdraw_oa_enabled and withdraw_oa_monthly_amount > 0),
        "oa_runs_out_age": oa_runs_out_age,
        "oa_runs_out_year": oa_runs_out_year,
        "oa_runs_out_month": oa_runs_out_month,
        "oa_withdrawal_amount": float(withdraw_oa_monthly_amount),
        "oa_withdrawal_start_age": int(withdraw_oa_start_age),
        "oa_withdrawal_end_age": int(withdraw_oa_end_age),

        "house_enabled": bool(house_enabled and house_monthly_amount > 0),
        "house_end_age": int(house_end_age),
        "house_runs_out_age": house_runs_out_age,
        "house_runs_out_year": house_runs_out_year,
        "house_runs_out_month": house_runs_out_month,
        "house_amount": float(house_monthly_amount),
    }
    return monthly_df, yearly_df, cohort_frs, cohort_ers, meta, cpf_life_df, bequest_df


# ==============================
# Sidebar Inputs
# ==============================
with st.sidebar:
    st.header("Inputs")
    name = st.text_input("Name", value="Member")
    dob = st.date_input("Date of birth", value=date(1980,1,1), min_value=date(1900,1,1), max_value=date.today(), format="DD-MM-YYYY")
    gender = st.selectbox("Gender", ["M", "F"], index=1)

    start_year = st.number_input("Start year", min_value=2000, max_value=2100, value=date.today().year, step=1)
    years = st.slider("Years to project", min_value=5, max_value=100, value=60, step=1)

    monthly_income = st.number_input("Monthly income (gross)", min_value=0.0, value=6000.0, step=100.0, format="%.2f")
    annual_bonus = st.number_input("Annual bonus (gross)", min_value=0.0, value=6000.0, step=500.0, format="%.2f")
    salary_growth_pct = st.number_input("Salary growth % p.a.", min_value=0.0, max_value=0.20, value=0.03, step=0.01, format="%.2f")
    bonus_growth_pct = st.number_input("Bonus growth % p.a.", min_value=0.0, max_value=0.20, value=0.03, step=0.01, format="%.2f")
    retirement_age = st.number_input("Retirement age (stop working contributions)", min_value=40, max_value=80, value=60, step=1, help="From this birthday onward, monthly salary and bonus contributions stop.")

    st.subheader("Opening balances")
    col1, col2 = st.columns(2)
    with col1:
        opening_OA = st.number_input("OA", min_value=0.0, value=30000.0, step=100.0, format="%.2f")
        opening_SA = st.number_input("SA", min_value=0.0, value=100000.0, step=100.0, format="%.2f")
    with col2:
        opening_MA = st.number_input("MA", min_value=0.0, value=75000.0, step=100.0, format="%.2f")
        opening_RA = st.number_input("RA", min_value=0.0, value=0.0, step=100.0, format="%.2f")

    with st.expander("Advanced assumptions", expanded=False):
        frs_growth_pct = st.number_input("FRS growth after last known year (default 3.5%)", min_value=0.0, max_value=0.10, value=FRS_growth_pct_default, step=0.005, format="%.3f")
        bhs_growth_pct = st.number_input("BHS growth after 2025 (default 5%)", min_value=0.0, max_value=0.10, value=BHS_growth_pct_default, step=0.005, format="%.3f")
        st.caption("You can adjust FRS/BHS growth to reflect future policy changes.")
        ers_factor = st.number_input("Desired RA opening amount (×FRS)", min_value=1.0, max_value=2.0, value=1.0, step=0.05, help="Target RA amount at age 55 as a multiple of FRS (1× to 2×).")
        st.caption("Choose your desired RA opening target between 1× and 2× FRS.")

with st.expander("Top-ups", expanded=False):
    st.caption("Amounts apply once every calendar year in the selected month. SA top-ups only when SA < FRS (cohort). RA top-ups allowed up to prevailing ERS, based on capital (excludes RA interest).")
    colA, colB = st.columns(2)
    with colA:
        m_topup_month = st.selectbox("Month to apply top-ups", list(range(1,13)), index=0, help="1=Jan ... 12=Dec")
        topup_OA = st.number_input("Top-up to OA (yearly)", min_value=0.0, value=0.0, step=100.0)
        topup_SA_RA = st.number_input("Top-up to SA (if <55) / RA (if ≥55) (yearly)", min_value=0.0, value=0.0, step=100.0)
        topup_MA = st.number_input("Top-up to MA (yearly)", min_value=0.0, value=0.0, step=100.0)

    st.markdown("---")
    topup_stop_option = st.selectbox("Top-up stop option", ["No limit", "After N years", "After age X"], index=0)
    colY, colZ = st.columns(2)
    with colY:
        if topup_stop_option == "After N years":
            topup_years_limit = st.number_input("Number of years to top-up", min_value=1, max_value=60, value=10, step=1)
        else:
            topup_years_limit = 0
    with colZ:
        if topup_stop_option == "After age X":
            topup_stop_age = st.number_input("Stop top-ups from this age (no top-ups when age > this)", min_value=30, max_value=100, value=65, step=1)
        else:
            topup_stop_age = 120

with st.expander("Long-term care insurance (paid from MA)", expanded=False):
    include_ltci = st.checkbox("I have long-term care insurance and pay premiums from MA", value=False)
    col_lt1, col_lt2, col_lt3 = st.columns(3)
    with col_lt1:
        ltci_month = st.selectbox("Month to deduct", list(range(1,13)), index=0, disabled=not include_ltci)
    with col_lt2:
        ltci_ma_premium = st.number_input("MA used per year (S$)", min_value=0.0, value=0.0, step=10.0, disabled=not include_ltci)
    with col_lt3:
        ltci_pay_until_age = st.number_input("Premium payable up to age", min_value=30, max_value=120, value=67, step=1, disabled=not include_ltci)

# ---- Integrated Shield Plan (IP) + MSHL month unified ----
with st.expander("Health Insurance (MSHL + Integrated Shield)", expanded=False):
    insurance_month = st.selectbox("Month to deduct health insurance premiums (MSHL & IP)", list(range(1,13)), index=0)

    ip_enabled = st.checkbox("Include Integrated Shield Plan premiums", value=False)
    ip_upload = st.file_uploader("Upload IP premiums CSV", type=["csv"], help="Headers required: Insurer, Ward Class, Plan Type, Plan Name, Age, Premium in MA, Premium in Cash")
    ip_df, ip_load_msg = (None, None)
    ip_insurer = ip_ward = ip_base_plan = ip_rider = None

    if ip_enabled:
        ip_df, ip_load_msg = try_load_ip_csv(ip_upload)
        if ip_df is None:
            st.warning(ip_load_msg)
        else:
            err = validate_ip_df(ip_df)
            if err:
                st.error(err)
                ip_df = None
            else:
                try:
                    ip_df["Age"] = ip_df["Age"].astype(int)
                except Exception:
                    st.error("Column 'Age' must be integer values.")
                    ip_df = None

        if ip_df is not None:
            insurers = sorted(ip_df["Insurer"].dropna().unique().tolist())
            ip_insurer = st.selectbox("Insurer", insurers, index=0 if insurers else 0)

            wards = sorted(ip_df[ip_df["Insurer"] == ip_insurer]["Ward Class"].dropna().unique().tolist())
            ip_ward = st.selectbox("Ward Class", wards, index=0 if wards else 0)

            base_opts = sorted(
                ip_df[
                    (ip_df["Insurer"] == ip_insurer) &
                    (ip_df["Ward Class"] == ip_ward) &
                    (ip_df["Plan Type"] == "Base")
                ]["Plan Name"].dropna().unique().tolist()
            )
            base_opts = ["(None)"] + base_opts
            ip_base_plan = st.selectbox("Base Plan", base_opts, index=0)

            rider_opts = sorted(
                ip_df[
                    (ip_df["Insurer"] == ip_insurer) &
                    (ip_df["Ward Class"] == ip_ward) &
                    (ip_df["Plan Type"] == "Rider")
                ]["Plan Name"].dropna().unique().tolist()
            )
            rider_opts = ["(None)"] + rider_opts
            ip_rider = st.selectbox("Rider", rider_opts, index=0)

            
# ---- Housing loan (OA monthly) ----
with st.expander("Housing loan (OA monthly)", expanded=False):
    house_enabled = st.checkbox("Deduct housing repayment from OA every month", value=False)
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        house_monthly_amount = st.number_input("Monthly OA deduction (S$)", min_value=0.0, value=0.0, step=50.0, format="%.2f", disabled=not house_enabled)
    with col_h2:
        house_end_age = st.number_input("End age (stop housing deduction after this age)", min_value=18, max_value=100, value=60, step=1, disabled=not house_enabled)

# ---- OA Withdrawal (55+) ----
with st.expander("OA Withdrawal (55+)", expanded=False):
    withdraw_oa_enabled = st.checkbox("Enable monthly OA withdrawal", value=False)
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        withdraw_oa_monthly_amount = st.number_input("Monthly OA withdrawal (S$)", min_value=0.0, value=0.0, step=50.0, format="%.2f", disabled=not withdraw_oa_enabled)
    with col_w2:
        withdraw_oa_start_age = st.number_input("Start age (≥55)", min_value=55, max_value=120, value=60, step=1, disabled=not withdraw_oa_enabled)
    with col_w3:
        withdraw_oa_end_age = st.number_input("End age", min_value=55, max_value=120, value=90, step=1, disabled=not withdraw_oa_enabled)


with st.expander("CPF LIFE (payouts)", expanded=False):
    include_cpf_life = st.checkbox("Include CPF LIFE payouts in projection", value=True)
    cpf_life_plan = st.selectbox("Plan", ["Standard","Escalating","Basic"], index=0)
    payout_start_age = st.number_input("Payout start age (65–70)", min_value=65, max_value=70, value=65, step=1)

run_btn = st.button("Run Projection", type="primary", use_container_width=True)

# ==============================
# Main
# ==============================
st.title("CPF Projector")
st.markdown("Project CPF balances year-by-year with clear charts and downloadable tables.")
st.markdown('<div class="small-muted">Assumes current CPF rules in 2025; edit assumptions in the sidebar.</div>', unsafe_allow_html=True)

if run_btn:
    opening_balances = {"OA": opening_OA, "SA": opening_SA, "MA": opening_MA, "RA": opening_RA}
    monthly_df, yearly_df, cohort_frs, cohort_ers, meta, cpf_life_df, bequest_df = project(
        name=name,
        dob_str=dob.strftime("%Y-%m-%d"),
        gender=gender,
        start_year=int(start_year),
        years=int(years),
        monthly_income=float(monthly_income),
        annual_bonus=float(annual_bonus),
        salary_growth_pct=float(salary_growth_pct),
        bonus_growth_pct=float(bonus_growth_pct),
        opening_balances=opening_balances,
        frs_growth_pct=float(frs_growth_pct),
        bhs_growth_pct=float(bhs_growth_pct),
        ers_factor=float(ers_factor),
        retirement_age=int(retirement_age),
        include_cpf_life=bool(include_cpf_life),
        cpf_life_plan=str(cpf_life_plan),
        payout_start_age=int(payout_start_age),
        topup_stop_option=str(topup_stop_option),
        topup_years_limit=int(topup_years_limit),
        topup_stop_age=int(topup_stop_age),
        # LTCI
        include_ltci=bool(include_ltci),
        ltci_ma_premium=float(ltci_ma_premium),
        ltci_pay_until_age=int(ltci_pay_until_age),
        ltci_month=int(ltci_month),
        # Insurance (MSHL + IP unified month)
        ip_enabled=bool(ip_enabled),
        ip_df=ip_df if ip_enabled else None,
        ip_insurer=ip_insurer,
        ip_ward=ip_ward,
        ip_base_plan=ip_base_plan,
        ip_rider=ip_rider,
        insurance_month=int(insurance_month),
        # OA Withdrawal
        withdraw_oa_enabled=bool(withdraw_oa_enabled),
        withdraw_oa_monthly_amount=float(withdraw_oa_monthly_amount),
        withdraw_oa_start_age=int(withdraw_oa_start_age),
        withdraw_oa_end_age=int(withdraw_oa_end_age),
        # Housing (OA monthly)
        house_enabled=bool(house_enabled),
        house_monthly_amount=float(house_monthly_amount),
        house_end_age=int(house_end_age),
    )

    # Banner ribbons
    ribbons = []
    if meta.get("house_enabled"):
        ribbons.append(
            f"Housing deduction ACTIVE — ${meta['house_amount']:,.0f}/mo, "
            f"till Age {meta['house_end_age']}"
        )
    
    if meta.get("oa_withdrawal_enabled"):
        ribbons.append(
            f"OA withdrawal ACTIVE — ${meta['oa_withdrawal_amount']:,.0f}/mo, "
            f"Age {meta['oa_withdrawal_start_age']}–{meta['oa_withdrawal_end_age']}"
        )
    
    if ribbons:
        # NEW — space-separated, wrapped in a div
        html_ribbons = " ".join([f"<span class='pill'>{r}</span>" for r in ribbons])
        st.markdown(f"<div class='ribbon-row'>{html_ribbons}</div>", unsafe_allow_html=True)


    # Warnings
    if meta.get("house_enabled") and (meta.get("house_runs_out_age") is not None) and (meta["house_runs_out_age"] < meta["house_end_age"]):
        st.warning(
            f"OA runs out at age {meta['house_runs_out_age']} "
            f"(Year {meta['house_runs_out_year']}) before your housing end age {meta['house_end_age']}."
        )

    
    if meta.get("oa_withdrawal_enabled") and (meta.get("oa_runs_out_age") is not None):
        st.warning(
            f"OA runs out at age {meta['oa_runs_out_age']} "
            f"(Year {meta['oa_runs_out_year']}) under your OA withdrawal settings."
        )
    
    # KPI Cards
    end_row = yearly_df.sort_values("Year").iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "OA (final)", end_row["End_OA"]),
        (c2, "SA (final)", end_row["End_SA"]),
        (c3, "MA (final)", end_row["End_MA"]),
        (c4, "RA (final)", end_row["End_RA"]),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="small-muted">{label}</div><div style="font-size:24px;font-weight:700;">${val:,.0f}</div></div>', unsafe_allow_html=True)

    # Yearly stacked balances with FRS line
    st.markdown("### Yearly Balances (Stacked)")
    yearly_long = yearly_df.melt(id_vars=['Year','Age_end'], value_vars=['End_OA','End_SA','End_MA','End_RA'], var_name='Account', value_name='Balance')
    yearly_long['Account'] = yearly_long['Account'].replace({'End_OA':'OA','End_SA':'SA','End_MA':'MA','End_RA':'RA'})
    yearly_long['YearAge'] = yearly_long.apply(lambda r: f"{int(r['Year'])} (Age {int(r['Age_end'])})", axis=1)
    stacked = alt.Chart(yearly_long).mark_bar().encode(
        x=alt.X('YearAge:O', title='Year (Age)', sort=None),
        y=alt.Y('sum(Balance):Q', title='Balance (S$)'),
        color=alt.Color('Account:N', legend=alt.Legend(title='Account')),
        tooltip=['Year','Age_end','Account','Balance']
    ).properties(height=360)
    frs_rule = alt.Chart(pd.DataFrame({'y': [cohort_frs]})).mark_rule(strokeDash=[6,3]).encode(
        y='y:Q', tooltip=[alt.Tooltip('y:Q', title='Cohort FRS')]
    )
    st.altair_chart(stacked + frs_rule, use_container_width=True)

    # CPF LIFE payouts & bequest
    if include_cpf_life and cpf_life_df is not None:
        st.markdown("### CPF LIFE Payouts")
        start_monthly = meta.get("monthly_start_payout", None)
        if start_monthly is not None:
            st.markdown(
                f"**Plan:** {cpf_life_plan} &nbsp;&nbsp; "
                f"**Start age:** {payout_start_age} &nbsp;&nbsp; "
                f"**Monthly payout at start:** ${start_monthly:,.0f} "
            )

        _cpf_life_table = cpf_life_df.copy()
        _cpf_life_table["Year (Age)"] = _cpf_life_table["Year"].map(lambda y: _label_year_age(y, yearly_df))
        st.dataframe(
            _cpf_life_table[["Year (Age)", "Monthly_Payout", "Annual_Payout"]]
                .style.format({"Monthly_Payout": "{:,.0f}", "Annual_Payout": "{:,.0f}"}),
            use_container_width=True, height=260
        )

    if include_cpf_life and bequest_df is not None:
        st.markdown("### CPF LIFE Bequest (Estimated)")
        _bequest_table = bequest_df.copy()
        _bequest_table["Year (Age)"] = _bequest_table["Year"].map(lambda y: _label_year_age(y, yearly_df))
        st.dataframe(
            _bequest_table[["Year (Age)", "Bequest_Remaining", "Unused_Premium", "RA_Savings_Remaining"]]
                .style.format({
                    "Bequest_Remaining": "{:,.0f}",
                    "Unused_Premium": "{:,.0f}",
                    "RA_Savings_Remaining": "{:,.0f}",
                }),
            use_container_width=True, height=260
        )

        # Bequest + Monthly payout chart
        st.markdown("### Bequest & Monthly Payout Over Time")
        _bequest_plot = bequest_df.copy()
        _bequest_plot["YearAge"] = _bequest_plot["Year"].map(lambda y: _label_year_age(y, yearly_df))
        bequest_long = _bequest_plot.melt(
            id_vars=["Year", "YearAge"],
            value_vars=["Bequest_Remaining", "RA_Savings_Remaining", "Unused_Premium"],
            var_name="Component",
            value_name="Amount",
        )
        _payout_plot = cpf_life_df.copy()
        _payout_plot["YearAge"] = _payout_plot["Year"].map(lambda y: _label_year_age(y, yearly_df))
        _payout_plot["Component"] = "Monthly_Payout"

        legend_domain = ["Bequest_Remaining", "RA_Savings_Remaining", "Unused_Premium", "Monthly_Payout"]
        legend_range  = ["#1f77b4", "#9ecae1", "#d62728", "#6b7280"]

        base_x = alt.X("YearAge:O", title="Year (Age)", sort=None, axis=alt.Axis(labelAngle=-40))
        shared_color = alt.Color(
            "Component:N",
            title="Component",
            scale=alt.Scale(domain=legend_domain, range=legend_range),
        )

        bequest_lines = alt.Chart(bequest_long).mark_line(point=True).encode(
            x=base_x,
            y=alt.Y("Amount:Q", title="Bequest (S$)"),
            color=shared_color,
            tooltip=[
                alt.Tooltip("Year:Q", title="Year"),
                alt.Tooltip("YearAge:N", title="Year (Age)"),
                alt.Tooltip("Component:N", title="Component"),
                alt.Tooltip("Amount:Q", title="Bequest", format=",.0f"),
            ],
        )

        payout_line = alt.Chart(_payout_plot).mark_line(point=True, strokeDash=[4, 2]).encode(
            x=base_x,
            y=alt.Y("Monthly_Payout:Q", axis=alt.Axis(title="Monthly Payout (S$)", orient="right")),
            color=shared_color,
            tooltip=[
                alt.Tooltip("Year:Q", title="Year"),
                alt.Tooltip("YearAge:N", title="Year (Age)"),
                alt.Tooltip("Monthly_Payout:Q", title="Monthly payout", format=",.0f"),
                alt.Tooltip("Annual_Payout:Q", title="Annual payout", format=",.0f"),
            ],
        )

        dual_axis_chart = alt.layer(bequest_lines, payout_line).resolve_scale(y='independent').properties(height=320)
        st.altair_chart(dual_axis_chart, use_container_width=True)

    # Health Insurance Premiums (stacked intended sources)
    if ("MSHL_Nominal" in yearly_df.columns) and (
        yearly_df[["MSHL_Nominal",
                   "IP_Base_MA_Nominal_Annual",
                   "IP_Base_Cash_Nominal_Annual",
                   "IP_Rider_Cash_Nominal_Annual"]].sum(numeric_only=True).sum() > 0
    ):
        st.markdown("### Health Insurance Premiums (By Intended Funding Source)")
        plot_df = yearly_df[["Year", "Age_end",
                             "MSHL_Nominal",
                             "IP_Base_MA_Nominal_Annual",
                             "IP_Base_Cash_Nominal_Annual",
                             "IP_Rider_Cash_Nominal_Annual"]].copy()
        plot_df["YearAge"] = plot_df["Year"].map(lambda y: _label_year_age(y, yearly_df))

        long = plot_df.melt(
            id_vars=["Year", "Age_end", "YearAge"],
            value_vars=["MSHL_Nominal", "IP_Base_MA_Nominal_Annual", "IP_Base_Cash_Nominal_Annual", "IP_Rider_Cash_Nominal_Annual"],
            var_name="Component", value_name="Amount"
        )

        name_map = {
            "MSHL_Nominal": "MediShield Life (MA)",
            "IP_Base_MA_Nominal_Annual": "IP Base (MA)",
            "IP_Base_Cash_Nominal_Annual": "IP Base (Cash)",
            "IP_Rider_Cash_Nominal_Annual": "IP Rider (Cash)",
        }
        order_map = {
            "MediShield Life (MA)": 0,
            "IP Base (MA)": 1,
            "IP Base (Cash)": 2,
            "IP Rider (Cash)": 3,
        }
        long["Component"] = long["Component"].map(name_map)
        long["ComponentOrder"] = long["Component"].map(order_map)

        ip_chart = alt.Chart(long).mark_bar().encode(
            x=alt.X("YearAge:O", title="Year (Age)", sort=None, axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("sum(Amount):Q", title="Premium (S$)"),
            color=alt.Color("Component:N", title="Component",
                            scale=alt.Scale(domain=list(order_map.keys()))),
            order=alt.Order("ComponentOrder:Q"),
            tooltip=[
                alt.Tooltip("Year:Q", title="Year"),
                alt.Tooltip("YearAge:N", title="Year (Age)"),
                alt.Tooltip("Component:N", title="Component"),
                alt.Tooltip("Amount:Q", title="Amount", format=",.0f"),
            ],
        ).properties(height=300)
        st.altair_chart(ip_chart, use_container_width=True)

    # Yearly table
    st.markdown("### Yearly Summary Table")
    yearly_df['Year (Age)'] = yearly_df.apply(lambda r: f"{int(r['Year'])} (Age {int(r['Age_end'])})", axis=1)
    cols = ['Year (Age)', 'Year', 'Age_end', 'End_OA','End_SA','End_MA','End_RA',
            'RA_capital_end','Prevailing_ERS','Total_Base_Interest','Total_Extra_Interest',
            'OW_subject_total','AW_subject_total','CPF_LIFE_Annual_Payout',
            'MSHL_Annual','LTCI_Premium_Annual_MA','IP_Base_MA_Annual','IP_Base_Cash_Annual','IP_Rider_Cash_Annual',
            'Housing_OA_Annual',
            'Topup_OA_Annual','Topup_SA_Annual','Topup_RA_Annual','Topup_MA_Annual',
            'OA_Withdrawal_Annual']
    yearly_display = yearly_df[[c for c in cols if c in yearly_df.columns]].copy()
    st.dataframe(
        yearly_display.style.format({
            'End_OA':'{:,.0f}', 'End_SA':'{:,.0f}', 'End_MA':'{:,.0f}', 'End_RA':'{:,.0f}',
            'RA_capital_end':'{:,.0f}', 'Prevailing_ERS':'{:,.0f}',
            'Total_Base_Interest':'{:,.0f}', 'Total_Extra_Interest':'{:,.0f}',
            'OW_subject_total':'{:,.0f}', 'AW_subject_total':'{:,.0f}',
            'CPF_LIFE_Annual_Payout':'{:,.0f}',
            'MSHL_Annual':'{:,.0f}', 'LTCI_Premium_Annual_MA':'{:,.0f}',
            'IP_Base_MA_Annual':'{:,.0f}', 'IP_Base_Cash_Annual':'{:,.0f}', 'IP_Rider_Cash_Annual':'{:,.0f}',
            'Housing_OA_Annual':'{:,.0f}',
            'Topup_OA_Annual':'{:,.0f}', 'Topup_SA_Annual':'{:,.0f}', 'Topup_RA_Annual':'{:,.0f}', 'Topup_MA_Annual':'{:,.0f}',
            'OA_Withdrawal_Annual':'{:,.0f}',
        }),
        use_container_width=True, height=360
    )

    # Monthly detail
    st.markdown("### Monthly Detail (optional)")
    with st.expander("Show monthly breakdown"):
        st.dataframe(monthly_df, use_container_width=True, height=420)

    # Notes
    cohort_bhs = get_bhs_for_year_with_cohort(dob, dob.year + 65, bhs_growth_pct)
    notes_html = [
        f"  Cohort FRS (fixed at age 55): <b>${cohort_frs:,.0f}</b>.",
        f"  Desired RA opening balance at 55 years old (&times;{ers_factor:.2f}): <b>${cohort_frs*ers_factor:,.0f}</b>.",
        f"  Cohort BHS (fixed at age 65): <b>${cohort_bhs:,.0f}</b>."
    ]
    if include_cpf_life and (cpf_life_df is not None) and meta.get("monthly_start_payout"):
        start_monthly = meta["monthly_start_payout"]
        notes_html.append(
            f"  Starting CPF LIFE monthly payout (nominal at start): "
            f"<b>${start_monthly:,.0f}</b>"
        )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:16px; line-height:1.6; color:#111;'>" +
        "<br/>".join(notes_html) +
        "</div>",
        unsafe_allow_html=True
    )

else:
    st.info("Set your inputs in the sidebar and click **Run Projection**.")

