import io
from dataclasses import dataclass
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

HOURS_PER_YEAR = 8760
DEFAULT_YEAR = 2025  # année non bissextile

@dataclass
class SimulationInputs:
    batt_power_mw: float
    batt_energy_mwh: float
    pv_dc_mw: float
    productible_kwh_per_kwp: float
    pv_losses_pct: float
    plant_availability_pct: float
    eta_charge: float
    eta_discharge: float
    pv_price: np.ndarray
    batt_sell_price: np.ndarray
    grid_buy_price: np.ndarray
    solar_profile: np.ndarray  # ici: production PV nette horaire en MWh
    nightly_bess_revenue_eur: float
    soc_steps: int
    initial_soc_mwh: float
    final_soc_mwh: float
    grid_export_limit_mw: float
    cycle_cost_eur_per_mwh: float
    charge_quantile: float
    discharge_quantile: float
    max_cycles_per_year: float

def _validate_array_length(arr: np.ndarray, name: str, expected_len: int = HOURS_PER_YEAR) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if len(arr) != expected_len:
        raise ValueError(f"{name} doit contenir exactement {expected_len} valeurs. Reçu: {len(arr)}.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contient des valeurs non numériques ou infinies.")
    return arr

def _read_single_column_csv(uploaded_file, expected_len: int = HOURS_PER_YEAR) -> np.ndarray:
    if uploaded_file is None:
        raise ValueError("Aucun fichier CSV fourni.")
    # Lecture robuste : accepte CSV simple, avec séparateur virgule ou point-virgule
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        df = pd.read_csv(uploaded_file)

    if df.shape[1] == 0:
        raise ValueError("Le CSV est vide.")

    first_col = df.iloc[:, 0].astype(str).str.strip()
    # Accepte les décimales avec virgule
    first_col = first_col.str.replace(",", ".", regex=False)
    series = pd.to_numeric(first_col, errors="coerce").dropna()
    if len(series) != expected_len:
        raise ValueError(
            f"Le CSV doit contenir exactement {expected_len} lignes numériques dans la première colonne. "
            f"Reçu: {len(series)}."
        )

    arr = series.to_numpy(dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Le CSV contient des valeurs non finies.")
    return arr
    
def _make_flat_curve(value: float, expected_len: int = HOURS_PER_YEAR) -> np.ndarray:
    if value is None:
        raise ValueError("La valeur moyenne annuelle n'a pas été renseignée.")
    return np.full(expected_len, float(value), dtype=float)

def build_standard_france_solar_profile() -> np.ndarray:
    """
    Génère une courbe solaire standard 8760h, relative, puis normalisée à 1 sur l'année.
    Ce n'est pas une météo réelle, mais une forme France standard raisonnable.
    """
    idx = pd.date_range(f"{DEFAULT_YEAR}-01-01 00:00:00", periods=HOURS_PER_YEAR, freq="h")
    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy()
    seasonal = 0.18 + 0.82 * (0.5 + 0.5 * np.sin(2 * np.pi * (doy - 81) / 365.0))
    daylight_hours = 8.0 + 8.0 * (0.5 + 0.5 * np.sin(2 * np.pi * (doy - 81) / 365.0))
    sunrise = 12.0 - daylight_hours / 2.0
    sunset = 12.0 + daylight_hours / 2.0
    shape = np.zeros(HOURS_PER_YEAR, dtype=float)
    for i in range(HOURS_PER_YEAR):
        if sunrise[i] <= hour[i] <= sunset[i]:
            x = (hour[i] - sunrise[i]) / max(sunset[i] - sunrise[i], 1e-9)
            shape[i] = (np.sin(np.pi * x) ** 1.6) * seasonal[i]
    total = shape.sum()
    if total <= 0:
        raise ValueError("Impossible de générer une courbe solaire standard valide.")
    return shape / total

def build_pv_generation_mwh(
    solar_profile_relative: np.ndarray,
    pv_dc_mw: float,
    productible_kwh_per_kwp: float,
    pv_losses_pct: float,
    plant_availability_pct: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    relative = _validate_array_length(solar_profile_relative, "Le profil solaire")
    relative = np.maximum(relative, 0.0)
    if relative.sum() <= 0:
        raise ValueError("Le profil solaire doit avoir une somme strictement positive.")
    if pv_dc_mw < 0 or productible_kwh_per_kwp < 0:
        raise ValueError("La puissance PV et le productible doivent être positifs.")
    if not (0 <= pv_losses_pct <= 100):
        raise ValueError("Les pertes PV doivent être entre 0 et 100 %.")
    if not (0 <= plant_availability_pct <= 100):
        raise ValueError("La disponibilité doit être entre 0 et 100 %.")

    # MWc * kWh/kWc/an = MWh/an
    annual_dc_mwh = pv_dc_mw * productible_kwh_per_kwp
    net_factor = (1.0 - pv_losses_pct / 100.0) * (plant_availability_pct / 100.0)
    annual_net_mwh = annual_dc_mwh * net_factor
    relative = relative / relative.sum()
    hourly_net_mwh = annual_net_mwh * relative

    stats = {
        "annual_dc_mwh": float(annual_dc_mwh),
        "annual_net_mwh": float(annual_net_mwh),
        "annual_losses_mwh": float(max(annual_dc_mwh - annual_net_mwh, 0.0)),
    }
    return hourly_net_mwh, stats

def optimize_dispatch_dp(inputs: SimulationInputs) -> Dict[str, np.ndarray]:
    pv = _validate_array_length(inputs.solar_profile, "La production PV nette horaire")
    pv = np.maximum(pv, 0.0)
    pv_price = _validate_array_length(inputs.pv_price, "Le prix PV")
    batt_sell = _validate_array_length(inputs.batt_sell_price, "Le prix de vente batterie")
    grid_buy = _validate_array_length(inputs.grid_buy_price, "Le prix d'achat réseau")
    charge_threshold = np.percentile(grid_buy, inputs.charge_quantile)
    discharge_threshold = np.percentile(batt_sell, inputs.discharge_quantile)
    max_total_discharge = inputs.max_cycles_per_year * inputs.batt_energy_mwh

    if np.any(~np.isfinite(pv)) or np.any(~np.isfinite(pv_price)) or np.any(~np.isfinite(batt_sell)) or np.any(~np.isfinite(grid_buy)):
        raise ValueError("Une ou plusieurs séries contiennent des valeurs invalides.")
    if inputs.batt_power_mw < 0 or inputs.batt_energy_mwh < 0:
        raise ValueError("La puissance et la capacité batterie doivent être positives.")
    if inputs.eta_charge <= 0 or inputs.eta_charge > 1:
        raise ValueError("Le rendement de charge doit être compris entre 0 et 1.")
    if inputs.eta_discharge <= 0 or inputs.eta_discharge > 1:
        raise ValueError("Le rendement de décharge doit être compris entre 0 et 1.")
    if inputs.initial_soc_mwh < 0 or inputs.final_soc_mwh < 0:
        raise ValueError("Les SOC initial et final doivent être positifs.")
    if inputs.initial_soc_mwh > inputs.batt_energy_mwh:
        raise ValueError("Le SOC initial ne peut pas dépasser la capacité batterie.")
    if inputs.final_soc_mwh > inputs.batt_energy_mwh:
        raise ValueError("Le SOC final ne peut pas dépasser la capacité batterie.")
        
    T = len(pv)

    if T != HOURS_PER_YEAR:
        raise ValueError("Toutes les séries doivent contenir 8760 heures.")

    soc_steps = int(max(21, inputs.soc_steps))
    soc_grid = np.linspace(0.0, inputs.batt_energy_mwh, soc_steps)
    cycle_budget = max_total_discharge

    def nearest_state_index(value: float) -> int:
        value = min(max(value, 0.0), inputs.batt_energy_mwh)
        return int(np.argmin(np.abs(soc_grid - value)))

    init_idx = nearest_state_index(inputs.initial_soc_mwh)
    final_idx = nearest_state_index(inputs.final_soc_mwh)

    # Variation max de SOC sur 1h
    DT = 1.0  # heure
    charge_soc_max = inputs.batt_power_mw * inputs.eta_charge * DT
    discharge_soc_max = inputs.batt_power_mw * DT / inputs.eta_discharge

    transitions = []
    for i, soc in enumerate(soc_grid):
        j_min = np.searchsorted(soc_grid, max(0.0, soc - discharge_soc_max), side="left")
        j_max = np.searchsorted(soc_grid, min(inputs.batt_energy_mwh, soc + charge_soc_max), side="right") - 1
        transitions.append(np.arange(j_min, j_max + 1, dtype=int))

    neg_inf = -1e30
    value_next = np.full(soc_steps, neg_inf, dtype=float)
    value_next[final_idx] = 0.0
    policy_next = np.full((T, soc_steps), -1, dtype=np.int16 if soc_steps < 32000 else np.int32)

    # Backward DP
    for t in range(T - 1, -1, -1):
        value_now = np.full(soc_steps, neg_inf, dtype=float)
        pv_t = pv[t]
        pv_price_t = pv_price[t]
        batt_sell_t = batt_sell[t]
        grid_buy_t = grid_buy[t]

        # Cas de base : tout le PV est vendu directement
        pv_base_revenue = pv_t * pv_price_t

        for i in range(soc_steps):
            best_val = neg_inf
            best_j = -1
            soc_i = soc_grid[i]

            for j in transitions[i]:
                delta_soc = soc_grid[j] - soc_i
                reward = pv_base_revenue
                
                # --- Calcul des flux candidats ---
                pv_direct_candidate = pv_t
                pv_to_batt = 0.0
                grid_charge = 0.0
                discharge_candidate = 0.0
                cycle_penalty = 0.0

                if delta_soc > 1e-12:
                    # Charge batterie
                    charge_input = delta_soc / inputs.eta_charge
                    pv_to_batt = min(charge_input, pv_t)
                    grid_charge = max(charge_input - pv_to_batt, 0.0)
                    pv_direct_candidate = pv_t - pv_to_batt

                    # 🚫 FILTRE QUANTILE CHARGE
                    if grid_charge > 1e-9 and grid_buy_t > charge_threshold:
                        continue
                    
                    # no negative spread charging (anti-arbitrage)
                    MIN_SPREAD = 10  # €/MWh
                    if pv_t < charge_input and (batt_sell_t - grid_buy_t) < MIN_SPREAD:
                        continue
                
                elif delta_soc < -1e-12:
                    # Décharge batterie
                    discharge_candidate = (-delta_soc) * inputs.eta_discharge
                    pv_direct_candidate = pv_t
                    
                    # 🚫 FILTRE QUANTILE DECHARGE
                    if discharge_candidate > 1e-9 and batt_sell_t < discharge_threshold:
                        continue

                # --- CONTRAINTE GRID ---
                total_export = pv_direct_candidate + discharge_candidate

                if total_export > inputs.grid_export_limit_mw:
                    excess = total_export - inputs.grid_export_limit_mw

                    # Curtail PV en priorité
                    reduction_pv = min(excess, pv_direct_candidate)
                    pv_direct_candidate -= reduction_pv
                    excess -= reduction_pv

                    # Puis réduire la décharge si nécessaire
                    if excess > 0:
                        discharge_candidate = max(discharge_candidate - excess, 0.0)

                    cycle_penalty = 0.0

                    if discharge_candidate > 0:
                        throughput = abs(delta_soc)
                        cycle_penalty = (throughput / inputs.batt_energy_mwh) * inputs.cycle_cost_eur_per_mwh

                # --- Calcul du reward ---
                reward = pv_direct_candidate * pv_price_t
                
                if delta_soc > 1e-12:
                    reward -= grid_charge * grid_buy_t
                
                elif delta_soc < -1e-12:
                    reward += discharge_candidate * batt_sell_t
                    reward -= cycle_penalty
                
                else:
                    reward = pv_direct_candidate * pv_price_t + discharge_candidate * batt_sell_t
                    reward -= cycle_penalty
                    
                total_val = reward + value_next[j]
                if total_val > best_val:
                    best_val = total_val
                    best_j = int(j)

            value_now[i] = best_val
            policy_next[t, i] = best_j

        value_next = value_now

    if np.all(value_next == neg_inf):
        raise RuntimeError("DP failed: all states unreachable")

    # Forward reconstruction
    soc = np.zeros(T + 1, dtype=float)
    soc[0] = soc_grid[init_idx]
    state = init_idx

    pv_direct = np.zeros(T, dtype=float)
    pv_to_batt = np.zeros(T, dtype=float)
    grid_charge = np.zeros(T, dtype=float)
    discharge = np.zeros(T, dtype=float)
    batt_sale_revenue = np.zeros(T, dtype=float)
    grid_charge_cost = np.zeros(T, dtype=float)
    pv_direct_revenue = np.zeros(T, dtype=float)

    for t in range(T):
        next_state = int(policy_next[t, state])
        if next_state < 0:
            raise RuntimeError(f"Policy failure at t={t}, state={state}, value={value_next[state]}")

        delta_soc = soc_grid[next_state] - soc_grid[state]
        soc[t + 1] = soc_grid[next_state]

        pv_direct_candidate = pv[t]
        pv_to_batt[t] = 0.0
        grid_charge[t] = 0.0
        discharge[t] = 0.0

        if delta_soc > 1e-12:
            charge_input = delta_soc / inputs.eta_charge
            pv_to_batt[t] = min(charge_input, pv[t])
            grid_charge[t] = max(charge_input - pv_to_batt[t], 0.0)
            pv_direct_candidate = pv[t] - pv_to_batt[t]

        elif delta_soc < -1e-12:
            discharge[t] = (-delta_soc) * inputs.eta_discharge
            pv_direct_candidate = pv[t]

        # --- CONTRAINTE GRID ---
        total_export = pv_direct_candidate + discharge[t]

        if total_export > inputs.grid_export_limit_mw:
            excess = total_export - inputs.grid_export_limit_mw

            reduction_pv = min(excess, pv_direct_candidate)
            pv_direct_candidate -= reduction_pv
            excess -= reduction_pv

            if excess > 0:
                discharge[t] = max(discharge[t] - excess, 0.0)

        pv_direct[t] = max(pv_direct_candidate, 0.0)

        pv_direct_revenue[t] = pv_direct[t] * pv_price[t]
        batt_sale_revenue[t] = discharge[t] * batt_sell[t]
        grid_charge_cost[t] = grid_charge[t] * grid_buy[t]

        state = next_state

    total_direct_pv_revenue = float(pv_direct_revenue.sum())
    total_batt_sale_revenue = float(batt_sale_revenue.sum())
    total_grid_charge_cost = float(grid_charge_cost.sum())
    nightly_revenue_total = float(inputs.nightly_bess_revenue_eur * (T // 24))
    total_revenue = total_direct_pv_revenue + total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total

    result = {
        "soc": soc,
        "pv_direct": pv_direct,
        "pv_to_batt": pv_to_batt,
        "grid_charge": grid_charge,
        "discharge": discharge,
        "pv_direct_revenue": pv_direct_revenue,
        "batt_sale_revenue": batt_sale_revenue,
        "grid_charge_cost": grid_charge_cost,
        "total_direct_pv_revenue": np.array([total_direct_pv_revenue]),
        "total_batt_sale_revenue": np.array([total_batt_sale_revenue]),
        "total_grid_charge_cost": np.array([total_grid_charge_cost]),
        "nightly_revenue_total": np.array([nightly_revenue_total]),
        "total_revenue": np.array([total_revenue]),
        "equivalent_cycles": np.array([discharge.sum() / max(inputs.batt_energy_mwh, 1e-12)]),
        "energy_sold_total_mwh": np.array([pv_direct.sum() + discharge.sum()]),
        "energy_shifted_mwh": np.array([discharge.sum()]),
        "pv_direct_sold_mwh": np.array([pv_direct.sum()]),
    }
    return result

def build_summary_table(result: Dict[str, np.ndarray], pv_stats: Dict[str, float]) -> pd.DataFrame:
    rows = [
        ("Revenu total", float(result["total_revenue"][0]), "EUR"),
        ("Revenu PV direct", float(result["total_direct_pv_revenue"][0]), "EUR"),
        ("Revenu vente batterie", float(result["total_batt_sale_revenue"][0]), "EUR"),
        ("Coût charge réseau batterie", float(result["total_grid_charge_cost"][0]), "EUR"),
        ("Revenu services système de nuit", float(result["nightly_revenue_total"][0]), "EUR"),
        ("Énergie totale vendue", float(result["energy_sold_total_mwh"][0]), "MWh"),
        ("Énergie shiftée par batterie", float(result["energy_shifted_mwh"][0]), "MWh"),
        ("Énergie PV vendue directement", float(result["pv_direct_sold_mwh"][0]), "MWh"),
        ("Cycles équivalents batterie", float(result["equivalent_cycles"][0]), "cycles/an"),
        ("Production PV théorique brute", float(pv_stats["annual_dc_mwh"]), "MWh"),
        ("Production PV nette valorisable", float(pv_stats["annual_net_mwh"]), "MWh"),
        ("Énergie PV perdue (pertes + disponibilité)", float(pv_stats["annual_losses_mwh"]), "MWh"),
    ]
    return pd.DataFrame(rows, columns=["Indicateur", "Valeur", "Unité"])

def monthly_dataframe(result: Dict[str, np.ndarray]) -> pd.DataFrame:
    idx = pd.date_range(f"{DEFAULT_YEAR}-01-01 00:00:00", periods=HOURS_PER_YEAR, freq="h")
    df = pd.DataFrame({
        "datetime": idx,
        "pv_direct_revenue": result["pv_direct_revenue"],
        "batt_sale_revenue": result["batt_sale_revenue"],
        "grid_charge_cost": result["grid_charge_cost"],
        "pv_direct_mwh": result["pv_direct"],
        "shifted_mwh": result["discharge"],
        "grid_charge_mwh": result["grid_charge"],
        "pv_to_batt_mwh": result["pv_to_batt"],
    })
    df["month"] = df["datetime"].dt.strftime("%Y-%m")
    monthly = df.groupby("month", as_index=False).sum(numeric_only=True)
    monthly["net_revenue"] = monthly["pv_direct_revenue"] + monthly["batt_sale_revenue"] - monthly["grid_charge_cost"]
    return monthly

def to_excel_bytes(summary_df: pd.DataFrame, monthly_df: pd.DataFrame, hourly_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Synthese", index=False)
            monthly_df.to_excel(writer, sheet_name="Mensuel", index=False)
            hourly_df.to_excel(writer, sheet_name="Horaire", index=False)
    except ImportError:
        raise ImportError("Le package openpyxl n'est pas installé. Ajoute 'openpyxl' dans requirements.txt.")
    return output.getvalue()

def app():
    st.set_page_config(page_title="Évaluation revenus projet hybride PV + BESS", layout="wide")
    st.title("Évaluation des revenus d'un projet hybride PV + batterie")
    st.caption("Simulation 8760h avec optimisation économique annuelle de la batterie.")

    with st.expander("Hypothèses structurantes", expanded=False):
        st.markdown(
            """
            - Simulation **horaire sur 8760h**.
            - La batterie peut **charger depuis le PV et/ou depuis le réseau**.
            - Le moteur choisit la meilleure valorisation économique entre vente immédiate du PV, stockage PV et charge réseau.
            - Les **revenus de services système la nuit** sont ajoutés comme un **revenu fixe par nuit**, sans contrainte de capacité ni de SOC.
            - Le profil solaire standard est une **forme France standardisée**. Un CSV 8760 peut la remplacer.
            - L'optimisation utilise une **programmation dynamique discrétisée sur le SOC**.
            """
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        batt_power_mw = st.number_input("Puissance batterie utile (MW)", min_value=0.0, value=50.0, step=1.0)
        batt_energy_mwh = st.number_input("Capacité batterie utile (MWh)", min_value=0.0, value=100.0, step=1.0)
        pv_dc_mw = st.number_input("Puissance PV DC (MWc)", min_value=0.0, value=100.0, step=1.0)
        productible = st.number_input("Productible PV (kWh/kWc/an)", min_value=0.0, value=1200.0, step=10.0)
        grid_export_limit_mw = st.number_input("Limite injection réseau (MW)", min_value=0.0, value=pv_dc_mw, step=1.0)
        cycle_cost = st.number_input("Coût de cycle batterie (EUR/MWh)", value=5.0)
        charge_quantile = st.slider("Quantile charge (%)", 0, 50, 20)
        discharge_quantile = st.slider("Quantile décharge (%)", 0, 50, 20)
        max_cycles = st.number_input("Cycles max / an", value=300.0)

    with col2:
        pv_losses_pct = st.number_input("Pertes système PV (%)", min_value=0.0, max_value=100.0, value=8.0, step=0.5)
        availability_pct = st.number_input("Disponibilité globale centrale (%)", min_value=0.0, max_value=100.0, value=98.0, step=0.1)
        eta_charge = st.number_input("Rendement de charge batterie (%)", min_value=1.0, max_value=100.0, value=95.0, step=0.5) / 100.0
        eta_discharge = st.number_input("Rendement de décharge batterie (%)", min_value=1.0, max_value=100.0, value=95.0, step=0.5) / 100.0

    with col3:
        nightly_bess_revenue = st.number_input("Revenu services système nuit (EUR/nuit)", min_value=0.0, value=0.0, step=10.0)
        soc_steps = st.slider("Nombre de pas de SOC pour l'optimisation", min_value=21, max_value=201, value=101, step=10)
        initial_soc = st.number_input("SOC initial batterie (MWh)", min_value=0.0, value=0.0, step=1.0)
        final_soc = st.number_input("SOC final cible batterie (MWh)", min_value=0.0, value=0.0, step=1.0)

    st.subheader("Courbe solaire 8760h")
    solar_mode = st.radio(
        "Source du profil solaire",
        ["Courbe standard France", "Upload CSV 8760"],
        horizontal=True,
    )

    solar_upload = None
    uploaded_solar_is_relative = True

    if solar_mode == "Upload CSV 8760":
        solar_upload = st.file_uploader(
            "Upload du profil solaire CSV (8760 lignes, première colonne numérique)",
            type=["csv"],
            key="solar_csv",
        )
        uploaded_solar_is_relative = st.checkbox(
            "Le CSV uploadé est un profil relatif à normaliser sur le productible annuel (sinon : MWh nets horaires absolus)",
            value=True,
        )

    st.subheader("Prix PV")
    pv_price_mode = st.radio("Source du prix de vente du PV", ["Prix moyen annuel", "Upload CSV 8760"], horizontal=True)
    pv_price_value = None
    pv_price_upload = None

    if pv_price_mode == "Prix moyen annuel":
        pv_price_value = st.number_input("Prix moyen PV (EUR/MWh)", value=55.0, step=1.0)
    else:
        pv_price_upload = st.file_uploader("Upload prix PV CSV (8760 lignes)", type=["csv"], key="pv_price")

    st.subheader("Prix vente batterie")
    batt_sell_mode = st.radio("Source du prix de vente de l'énergie shiftée", ["Prix moyen annuel", "Upload CSV 8760"], horizontal=True)
    batt_sell_value = None
    batt_sell_upload = None

    if batt_sell_mode == "Prix moyen annuel":
        batt_sell_value = st.number_input("Prix moyen vente batterie (EUR/MWh)", value=90.0, step=1.0)
    else:
        batt_sell_upload = st.file_uploader("Upload prix vente batterie CSV (8760 lignes)", type=["csv"], key="batt_sell")

    st.subheader("Prix d'achat réseau pour charge batterie")
    grid_mode = st.radio(
        "Source du prix d'achat réseau",
        ["Identique au prix vente batterie", "Prix moyen annuel", "Upload CSV 8760"],
        horizontal=True,
    )
    grid_buy_value = None
    grid_buy_upload = None

    if grid_mode == "Prix moyen annuel":
        grid_buy_value = st.number_input("Prix moyen achat réseau (EUR/MWh)", value=55.0, step=1.0)
    elif grid_mode == "Upload CSV 8760":
        grid_buy_upload = st.file_uploader("Upload prix achat réseau CSV (8760 lignes)", type=["csv"], key="grid_buy")

    st.markdown("---")
    run = st.button("Lancer la simulation", type="primary")

    with st.expander("Format des CSV attendus", expanded=False):
        st.markdown(
            """
            - **8760 lignes**, une heure par ligne, **première colonne numérique**.
            - Pas besoin d'en-tête spécifique.
            - Les décimales avec **point ou virgule** sont acceptées.
            - Pour le solaire uploadé :
              - soit **profil relatif** renormalisé sur le productible annuel,
              - soit **MWh nets horaires absolus** si la case correspondante est décochée.
            """
        )

    if not run:
        return

    try:
        if batt_energy_mwh < batt_power_mw and batt_energy_mwh > 0:
            st.warning("Attention : la capacité batterie est inférieure à 1h de puissance. C'est possible, mais atypique.")

        if initial_soc > batt_energy_mwh:
            st.error("Le SOC initial ne peut pas dépasser la capacité batterie.")
            return

        if final_soc > batt_energy_mwh:
            st.error("Le SOC final ne peut pas dépasser la capacité batterie.")
            return

        # Profil solaire
        if solar_mode == "Courbe standard France":
            solar_relative = build_standard_france_solar_profile()
            pv_hourly_mwh, pv_stats = build_pv_generation_mwh(
                solar_relative,
                pv_dc_mw,
                productible,
                pv_losses_pct,
                availability_pct,
            )
        else:
            if solar_upload is None:
                st.error("Merci d'uploader un CSV solaire 8760.")
                return

            uploaded = _read_single_column_csv(solar_upload)

            if uploaded_solar_is_relative:
                pv_hourly_mwh, pv_stats = build_pv_generation_mwh(
                    uploaded,
                    pv_dc_mw,
                    productible,
                    pv_losses_pct,
                    availability_pct,
                )
            else:
                pv_hourly_mwh = np.maximum(uploaded, 0.0)
                annual_net = float(pv_hourly_mwh.sum())
                annual_dc = float(pv_dc_mw * productible)
                pv_stats = {
                    "annual_dc_mwh": annual_dc,
                    "annual_net_mwh": annual_net,
                    "annual_losses_mwh": float(max(annual_dc - annual_net, 0.0)),
                }

        # Courbes de prix
        pv_price_curve = (
            _make_flat_curve(pv_price_value)
            if pv_price_mode == "Prix moyen annuel"
            else _read_single_column_csv(pv_price_upload)
        )

        batt_sell_curve = (
            _make_flat_curve(batt_sell_value)
            if batt_sell_mode == "Prix moyen annuel"
            else _read_single_column_csv(batt_sell_upload)
        )

        if grid_mode == "Identique au prix vente batterie":
            grid_buy_curve = batt_sell_curve.copy()
        elif grid_mode == "Prix moyen annuel":
            grid_buy_curve = _make_flat_curve(grid_buy_value)
        else:
            grid_buy_curve = _read_single_column_csv(grid_buy_upload)

        sim_inputs = SimulationInputs(
            batt_power_mw=batt_power_mw,
            batt_energy_mwh=batt_energy_mwh,
            pv_dc_mw=pv_dc_mw,
            productible_kwh_per_kwp=productible,
            pv_losses_pct=pv_losses_pct,
            plant_availability_pct=availability_pct,
            eta_charge=eta_charge,
            eta_discharge=eta_discharge,
            pv_price=pv_price_curve,
            batt_sell_price=batt_sell_curve,
            grid_buy_price=grid_buy_curve,
            solar_profile=pv_hourly_mwh,
            nightly_bess_revenue_eur=nightly_bess_revenue,
            soc_steps=soc_steps,
            initial_soc_mwh=initial_soc,
            final_soc_mwh=final_soc,
            grid_export_limit_mw=grid_export_limit_mw,
            cycle_cost_eur_per_mwh=cycle_cost,
            charge_quantile=charge_quantile,
            discharge_quantile=discharge_quantile,
            max_cycles_per_year=max_cycles,
        )

        with st.spinner("Optimisation économique annuelle en cours..."):
            result = optimize_dispatch_dp(sim_inputs)

        summary_df = build_summary_table(result, pv_stats)
        monthly_df = monthly_dataframe(result)

        idx = pd.date_range(f"{DEFAULT_YEAR}-01-01 00:00:00", periods=HOURS_PER_YEAR, freq="h")
        hourly_df = pd.DataFrame({
            "datetime": idx,
            "pv_generation_mwh": pv_hourly_mwh,
            "pv_price_eur_per_mwh": pv_price_curve,
            "battery_sell_price_eur_per_mwh": batt_sell_curve,
            "grid_buy_price_eur_per_mwh": grid_buy_curve,
            "pv_direct_mwh": result["pv_direct"],
            "pv_to_battery_mwh": result["pv_to_batt"],
            "grid_charge_mwh": result["grid_charge"],
            "battery_discharge_mwh": result["discharge"],
            "battery_soc_mwh_end": result["soc"][1:],
            "pv_direct_revenue_eur": result["pv_direct_revenue"],
            "battery_sale_revenue_eur": result["batt_sale_revenue"],
            "grid_charge_cost_eur": result["grid_charge_cost"],
        })

        excel_bytes = to_excel_bytes(summary_df, monthly_df, hourly_df)

        st.success("Simulation terminée.")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Revenu total", f"{result['total_revenue'][0]:,.0f} EUR")
        k2.metric("Énergie totale vendue", f"{result['energy_sold_total_mwh'][0]:,.0f} MWh")
        k3.metric("Énergie shiftée", f"{result['energy_shifted_mwh'][0]:,.0f} MWh")
        k4.metric("Cycles équivalents", f"{result['equivalent_cycles'][0]:,.1f}")

        st.subheader("Synthèse")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)

        with c1:
            fig1, ax1 = plt.subplots(figsize=(8, 4.5))
            bars = [
                float(result["total_direct_pv_revenue"][0]),
                float(result["total_batt_sale_revenue"][0]),
                -float(result["total_grid_charge_cost"][0]),
                float(result["nightly_revenue_total"][0]),
            ]
            labels = ["PV direct", "Vente batterie", "Coût charge réseau", "SS nuit"]
            ax1.bar(labels, bars)
            ax1.set_title("Décomposition des revenus")
            ax1.set_ylabel("EUR")
            ax1.tick_params(axis="x", rotation=20)
            st.pyplot(fig1)
            plt.close(fig1)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(8, 4.5))
            ax2.plot(monthly_df["month"], monthly_df["net_revenue"])
            ax2.set_title("Revenu net mensuel")
            ax2.set_ylabel("EUR")
            ax2.set_xlabel("Mois")
            ax2.tick_params(axis="x", rotation=45)
            st.pyplot(fig2)
            plt.close(fig2)

        c3, c4 = st.columns(2)

        with c3:
            fig3, ax3 = plt.subplots(figsize=(8, 4.5))
            ax3.plot(monthly_df["month"], monthly_df["pv_direct_mwh"], label="PV direct")
            ax3.plot(monthly_df["month"], monthly_df["shifted_mwh"], label="Énergie shiftée")
            ax3.set_title("Énergies valorisées par mois")
            ax3.set_ylabel("MWh")
            ax3.set_xlabel("Mois")
            ax3.legend()
            ax3.tick_params(axis="x", rotation=45)
            st.pyplot(fig3)
            plt.close(fig3)

        with c4:
            # --- Sélection 3 premiers jours de juin ---
            start_date = pd.Timestamp(f"{DEFAULT_YEAR}-06-01 00:00:00")
            end_date = start_date + pd.Timedelta(hours=72)

            df = hourly_df[
                (hourly_df["datetime"] >= start_date) &
                (hourly_df["datetime"] < end_date)
            ].copy()

            fig, ax1 = plt.subplots(figsize=(12, 5))

            # --- Aires empilées (flux sortants vers réseau) ---
            ax1.stackplot(
                df["datetime"],
                df["pv_direct_mwh"],
                df["battery_discharge_mwh"],
                labels=["PV → Réseau", "Batterie → Réseau"],
                alpha=0.8
            )

            # --- Aires négatives (charges batterie) ---
            ax1.stackplot(
                df["datetime"],
                -df["pv_to_battery_mwh"],
                -df["grid_charge_mwh"],
                labels=["PV → Batterie", "Réseau → Batterie"],
                alpha=0.5
            )

            ax1.axhline(0, linewidth=1)  # ligne zéro
            ax1.set_ylabel("Flux énergie (MWh)")
            ax1.set_xlabel("Date")

            # --- Prix (axe secondaire) ---
            ax2 = ax1.twinx()
            ax2.plot(
                df["datetime"],
                df["pv_price_eur_per_mwh"],  # spot price
                linestyle="--",
                alpha=0.7,
                label="Prix spot"
            )
            ax2.set_ylabel("Prix (EUR/MWh)")

            # --- Légende combinée ---
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

            ax1.set_title("Dispatch énergétique - 3 premiers jours de juin")

            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Table mensuelle")
        st.dataframe(monthly_df, use_container_width=True, hide_index=True)

        st.subheader("Exports")
        st.download_button(
            "Télécharger les résultats en Excel",
            data=excel_bytes,
            file_name="revenus_hybride_pv_batterie.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "Télécharger l'horaire en CSV",
            data=hourly_df.to_csv(index=False).encode("utf-8"),
            file_name="dispatch_horaire_hybride.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Erreur: {e}")

if __name__ == "__main__":
    app()
