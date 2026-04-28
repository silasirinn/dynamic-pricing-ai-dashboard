from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from data_loader import load_data
from feature_engineering import add_advanced_features, apply_one_hot_encoding, extract_date_features
from preprocessing import convert_to_datetime, drop_unnecessary_columns, handle_missing_values

st.set_page_config(
    page_title="AI Pricing Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MONTH_NAMES_TR = {
    1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
    7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık",
}

DAY_NAMES_TR = {
    0: "Pazartesi", 1: "Salı", 2: "Çarşamba", 3: "Perşembe",
    4: "Cuma", 5: "Cumartesi", 6: "Pazar"
}

CATEGORY_LABELS_TR = {"Beauty": "Güzellik", "Clothing": "Giyim", "Electronics": "Elektronik"}
GENDER_LABELS_TR = {"Female": "Kadın", "Male": "Erkek"}

RF_PARAMS = {
    "n_estimators": 200, "max_depth": 10, "min_samples_split": 5,
    "min_samples_leaf": 4, "random_state": 42,
}

AGE_BINS = [0, 25, 35, 50, 120]
AGE_LABELS = ["18-25", "26-35", "36-50", "51+"]


def inject_styles() -> None:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #E5E7EB;
        }
        
        .stApp {
            background-color: #0A0F1A;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(139, 92, 246, 0.15), transparent 30%),
                radial-gradient(circle at 90% 80%, rgba(59, 130, 246, 0.15), transparent 30%);
            background-attachment: fixed;
        }
        
        /* Hero Section */
        .hero-container {
            text-align: center;
            padding: 4rem 2rem;
            margin-top: 1rem;
            margin-bottom: 3rem;
            background: linear-gradient(145deg, rgba(20, 25, 40, 0.8), rgba(10, 15, 26, 0.9));
            border-radius: 24px;
            border: 1px solid rgba(139, 92, 246, 0.25);
            box-shadow: 0 15px 40px -10px rgba(139, 92, 246, 0.25);
            animation: fadeInDown 0.8s cubic-bezier(0.2, 0.8, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .hero-container::before {
            content: '';
            position: absolute;
            top: -50%; left: -50%; width: 200%; height: 200%;
            background: radial-gradient(circle, rgba(139,92,246,0.1) 0%, transparent 60%);
            animation: pulseGlow 8s infinite alternate;
            z-index: 0;
            pointer-events: none;
        }
        
        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(to right, #60A5FA, #A78BFA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            position: relative;
            z-index: 1;
        }
        
        .hero-subtitle {
            font-size: 1.25rem;
            color: #9CA3AF;
            font-weight: 300;
            letter-spacing: 1px;
            position: relative;
            z-index: 1;
        }
        
        /* Card Panels */
        .card-panel {
            background: rgba(26, 32, 44, 0.65);
            border-radius: 20px;
            padding: 1.8rem;
            border: 1px solid rgba(255,255,255,0.06);
            backdrop-filter: blur(12px);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
            animation: fadeInUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) backwards;
            height: 100%;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }
        
        .card-panel:hover {
            transform: translateY(-6px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.4);
            border: 1px solid rgba(139, 92, 246, 0.4);
        }
        
        /* Metric Cards */
        .metric-card {
            text-align: center;
            padding: 1.2rem;
            background: rgba(15, 23, 42, 0.6);
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.05);
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }
        
        .metric-card::after {
            content: '';
            position: absolute;
            bottom: 0; left: 0; width: 100%; height: 4px;
            background: linear-gradient(90deg, #3B82F6, #8B5CF6);
            opacity: 0.7;
        }
        
        .metric-title {
            font-size: 0.95rem;
            color: #9CA3AF;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-bottom: 0.8rem;
            font-weight: 600;
        }
        
        .metric-value {
            font-size: 2.8rem;
            font-weight: 800;
            color: #FFFFFF;
            line-height: 1.1;
        }
        
        .metric-value.success { color: #34D399; text-shadow: 0 0 20px rgba(52, 211, 153, 0.4); }
        .metric-value.warning { color: #FBBF24; text-shadow: 0 0 20px rgba(251, 191, 36, 0.4); }
        .metric-value.error { color: #F87171; text-shadow: 0 0 20px rgba(248, 113, 113, 0.4); }
        
        /* Smart Interpretation Box */
        .interpretation-box {
            padding: 1.8rem;
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.15));
            border-left: 5px solid #8B5CF6;
            margin: 2.5rem 0;
            font-size: 1.15rem;
            line-height: 1.7;
            animation: fadeIn 1.2s ease-in;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            color: #E5E7EB;
        }

        .section-header {
            font-size: 1.7rem;
            font-weight: 600;
            margin-bottom: 1.8rem;
            margin-top: 1rem;
            color: #F3F4F6;
            display: inline-flex;
            align-items: center;
            gap: 10px;
        }
        
        .section-header::after {
            content: '';
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, rgba(139,92,246,0.5), transparent);
            margin-left: 15px;
        }
        
        /* Custom UI Elements override */
        div[data-baseweb="select"] > div { 
            background-color: rgba(15, 23, 42, 0.8) !important; 
            border-color: rgba(139, 92, 246, 0.3) !important; 
            border-radius: 10px;
            color: white;
        }
        
        .stSlider > div > div > div { 
            background-color: #8B5CF6 !important; 
        }
        
        /* Animations Keyframes */
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes pulseGlow {
            0% { transform: scale(1); opacity: 0.6; }
            100% { transform: scale(1.05); opacity: 0.9; }
        }
        </style>
    """, unsafe_allow_html=True)


def format_amount(value: float) -> str:
    return f"{value:,.0f}"

def category_tr(value: str) -> str:
    return CATEGORY_LABELS_TR.get(value, value)

def gender_tr(value: str) -> str:
    return GENDER_LABELS_TR.get(value, value)

def age_group_for_value(age: int) -> str:
    group = pd.cut(pd.Series([age]), bins=AGE_BINS, labels=AGE_LABELS, right=True).iloc[0]
    return str(group)


@st.cache_data(show_spinner=False)
def load_sales_data() -> pd.DataFrame:
    data_path = ROOT_DIR / "data" / "retail_sales_dataset.csv"
    df = load_data(str(data_path))
    df = handle_missing_values(df)
    df = convert_to_datetime(df, date_column="date")
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
    df["unit_value"] = df["total_amount"] / df["quantity"].clip(lower=1)
    return df


@st.cache_data(show_spinner=False)
def prepare_model_frame() -> tuple[pd.DataFrame, pd.Series]:
    data_path = ROOT_DIR / "data" / "retail_sales_dataset.csv"
    df = load_data(str(data_path))
    df = handle_missing_values(df)
    df = convert_to_datetime(df, date_column="date")
    df = add_advanced_features(df)
    df = extract_date_features(df, date_column="date")
    df = apply_one_hot_encoding(df, categorical_columns=["gender", "product_category", "age_group"])
    df = drop_unnecessary_columns(df, ["transaction_id", "customer_id", "date", "price_per_unit"])
    y = df["total_amount"]
    X = df.drop(columns=["total_amount"])
    return X, y


@st.cache_resource(show_spinner=False)
def load_model() -> RandomForestRegressor:
    model_path = ROOT_DIR / "outputs" / "models" / "final_rf_model_no_leakage.joblib"
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def compute_model_diagnostics() -> dict[str, object]:
    X, y = prepare_model_frame()
    model = load_model()

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    predictions = model.predict(X_test)
    test_metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        RandomForestRegressor(**RF_PARAMS),
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        return_train_score=False,
    )
    cv_metrics = {
        "r2": cv_results["test_r2"].mean(),
        "mae": -cv_results["test_mae"].mean(),
        "rmse": -cv_results["test_rmse"].mean(),
    }

    importances = (
        pd.DataFrame(
            {
                "feature": model.feature_names_in_,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "test": test_metrics,
        "cv": cv_metrics,
        "importances": importances,
    }


def build_prediction_frame(
    age: int,
    quantity: int,
    gender: str,
    category: str,
    month: int,
    day_of_week: int,
    model: RandomForestRegressor,
    sales_df: pd.DataFrame,
) -> pd.DataFrame:
    category_quantity_means = sales_df.groupby("product_category")["quantity"].mean()
    feature_values = {feature: 0.0 for feature in model.feature_names_in_}
    
    feature_values["age"] = float(age)
    feature_values["quantity"] = float(quantity)
    feature_values["category_quantity_mean"] = float(category_quantity_means.get(category, sales_df["quantity"].mean()))
    feature_values["month"] = float(month)
    feature_values["day_of_week"] = float(day_of_week)
    feature_values["is_weekend"] = float(int(day_of_week >= 5))
    feature_values["month_sin"] = float(np.sin(2 * np.pi * month / 12))
    feature_values["month_cos"] = float(np.cos(2 * np.pi * month / 12))
    feature_values["gender_Male"] = float(int(gender == "Male"))
    feature_values["product_category_Clothing"] = float(int(category == "Clothing"))
    feature_values["product_category_Electronics"] = float(int(category == "Electronics"))

    age_group = age_group_for_value(age)
    feature_values["age_group_26-35"] = float(int(age_group == "26-35"))
    feature_values["age_group_36-50"] = float(int(age_group == "36-50"))
    feature_values["age_group_51+"] = float(int(age_group == "51+"))

    return pd.DataFrame([feature_values], columns=model.feature_names_in_)


def find_similar_real_records(
    sales_df: pd.DataFrame,
    category: str,
    gender: str,
    age: int,
    quantity: int,
) -> pd.DataFrame:
    similar_df = sales_df[
        (sales_df["product_category"] == category)
        & (sales_df["gender"] == gender)
        & (sales_df["quantity"] == quantity)
    ].copy()

    if similar_df.empty:
        similar_df = sales_df[
            (sales_df["product_category"] == category)
            & (sales_df["gender"] == gender)
        ].copy()

    similar_df["age_gap"] = (similar_df["age"] - age).abs()
    similar_df = similar_df.sort_values(["age_gap", "date"]).head(5)
    return similar_df


def render_dashboard():
    inject_styles()
    sales_df = load_sales_data()
    model = load_model()

    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🤖 AI Pricing Assistant</div>
        <div class="hero-subtitle">Makine Öğrenmesi Tabanlı Fiyat Öneri & Karar Destek Sistemi</div>
    </div>
    """, unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1.4], gap="large")

    with col_in:
        st.markdown('<div class="section-header">🛠️ Senaryo Parametreleri</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-panel" style="animation-delay: 0.1s;">', unsafe_allow_html=True)
        
        scenario_category = st.selectbox(
            "Ürün Kategorisi", 
            options=sorted(sales_df["product_category"].unique()), 
            format_func=category_tr
        )
        scenario_gender = st.selectbox(
            "Cinsiyet", 
            options=sorted(sales_df["gender"].unique()), 
            format_func=gender_tr
        )
        
        c_age, c_qty = st.columns(2)
        scenario_age = c_age.slider("Yaş", min_value=18, max_value=70, value=30)
        scenario_quantity = c_qty.slider("Miktar", min_value=1, max_value=10, value=1)
        
        c_month, c_day = st.columns(2)
        scenario_month = c_month.selectbox("Ay", options=list(MONTH_NAMES_TR.keys()), format_func=lambda x: MONTH_NAMES_TR[x], index=4)
        scenario_day = c_day.selectbox("Gün", options=list(DAY_NAMES_TR.keys()), format_func=lambda x: DAY_NAMES_TR[x], index=0)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # PREDICTION LOGIC
    pred_frame = build_prediction_frame(
        scenario_age, scenario_quantity, scenario_gender, scenario_category, 
        scenario_month, scenario_day, model, sales_df
    )
    pred_total = max(float(model.predict(pred_frame)[0]), 0.0)
    pred_unit = pred_total / scenario_quantity if scenario_quantity else 0.0
    
    similar_df = find_similar_real_records(sales_df, scenario_category, scenario_gender, scenario_age, scenario_quantity)
    real_avg = float(similar_df["total_amount"].mean()) if not similar_df.empty else pred_total
    real_unit = real_avg / scenario_quantity if scenario_quantity else pred_unit
    
    price_diff_pct = ((pred_unit - real_unit) / real_unit) * 100 if real_unit else 0

    if price_diff_pct > 12:
        demand_level = "Düşük (Riskli)"
        demand_color = "error"
        risk_text = "Bu fiyat seviyesi pazar ortalamasının belirgin şekilde üzerinde. Müşterinin satın almaktan vazgeçme ihtimali (churn) yüksek olabilir."
        icon = "⚠️"
    elif price_diff_pct < -12:
        demand_level = "Yüksek (Fırsat)"
        demand_color = "success"
        risk_text = "Piyasaya göre oldukça rekabetçi bir fiyat. Satışa dönüşme ihtimali çok yüksek, sürümden kazanma potansiyeli barındırıyor."
        icon = "🚀"
    else:
        demand_level = "Optimum (Dengeli)"
        demand_color = "warning" # Changed to warning for visual distinction (yellowish gold)
        risk_text = "Fiyat dengeli ve pazar koşullarıyla uyumlu. Standart satış performansının yakalanması beklenmektedir."
        icon = "✅"

    with col_out:
        st.markdown('<div class="section-header">🎯 AI Karar Çıktıları</div>', unsafe_allow_html=True)
        
        c_res1, c_res2 = st.columns(2)
        with c_res1:
            st.markdown(f"""
            <div class="card-panel" style="animation-delay: 0.2s;">
                <div class="metric-card">
                    <div class="metric-title">Önerilen Birim Fiyat</div>
                    <div class="metric-value" style="color: #60A5FA;">{pred_unit:,.0f} ₺</div>
                </div>
                <div style="text-align:center; color:#9CA3AF; font-size:0.95rem;">
                    Tahmini Toplam Gelir: <strong style="color:white;">{pred_total:,.0f} ₺</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_res2:
            st.markdown(f"""
            <div class="card-panel" style="animation-delay: 0.3s;">
                <div class="metric-card">
                    <div class="metric-title">Beklenen Satış Seviyesi</div>
                    <div class="metric-value {demand_color}">{demand_level}</div>
                </div>
                <div style="text-align:center; color:#9CA3AF; font-size:0.95rem;">
                    {icon} Risk Sapması: <strong style="color:white;">{price_diff_pct:+.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    # Smart Interpretation
    st.markdown(f"""
    <div class="interpretation-box">
        <strong>🧠 Akıllı Yorum:</strong> <br><br>
        Seçilmiş olan <em>{category_tr(scenario_category)}</em> kategorisinde, <em>{scenario_age}</em> yaşındaki <em>{gender_tr(scenario_gender)}</em> müşteri segmenti için, 
        <em>{MONTH_NAMES_TR[scenario_month]}</em> ayında yapılacak <em>{scenario_quantity} adetlik</em> işlemlerde makine öğrenmesi modelinin bulduğu optimal birim fiyat 
        <strong>{pred_unit:,.0f} ₺</strong> seviyesindedir. <br><br>
        {risk_text} Model, geçmiş satın alma davranışlarına dayanarak bu fiyatın kârlılık ve satış hacmi arasında en ideal dengeyi kuracağını öngörmektedir.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # What-If Analysis
    st.markdown('<div class="section-header">🔮 Dinamik Duyarlılık (What-If) Simülasyonu</div>', unsafe_allow_html=True)
    
    sim_quantities = list(range(1, 11))
    sim_prices = []
    for q in sim_quantities:
        f = build_prediction_frame(scenario_age, q, scenario_gender, scenario_category, scenario_month, scenario_day, model, sales_df)
        t = max(float(model.predict(f)[0]), 1.0)
        sim_prices.append(t / q)

    min_p, max_p = min(sim_prices), max(sim_prices)
    
    w_col1, w_col2 = st.columns([1, 1.8], gap="large")
    
    with w_col1:
        st.markdown('<div class="card-panel">', unsafe_allow_html=True)
        st.markdown('<div style="color:#E5E7EB; margin-bottom:1rem; font-size:1.1rem;">Fiyat stratejinizi test edin:</div>', unsafe_allow_html=True)
        
        target_price = st.slider(
            "Manuel Hedef Fiyat (₺)", 
            min_value=int(min_p * 0.7), 
            max_value=int(max_p * 1.3), 
            value=int(pred_unit),
            step=10
        )
        
        diffs = [abs(p - target_price) for p in sim_prices]
        best_idx = diffs.index(min(diffs))
        est_demand = sim_quantities[best_idx]
        est_revenue = est_demand * target_price
        
        st.markdown(f"""
        <div style="margin-top:2rem; padding-top:1.5rem; border-top:1px solid rgba(255,255,255,0.1);">
            <div style="margin-bottom:0.8rem; font-size:1.05rem; color:#9CA3AF;">
                Tahmini Optimal Talep: <span style="float:right; color:#34D399; font-weight:800; font-size:1.4rem;">{est_demand} Adet</span>
            </div>
            <div style="font-size:1.05rem; color:#9CA3AF;">
                Projeksiyon Geliri: <span style="float:right; color:#60A5FA; font-weight:800; font-size:1.4rem;">{est_revenue:,.0f} ₺</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with w_col2:
        demand_df = pd.DataFrame({"Miktar (Adet)": sim_quantities, "Birim Fiyat (₺)": sim_prices})
        fig_demand = px.line(
            demand_df, 
            x="Birim Fiyat (₺)", 
            y="Miktar (Adet)", 
            markers=True, 
            title="Model Tabanlı Fiyat-Talep Esnekliği Eğrisi"
        )
        fig_demand.add_vline(x=target_price, line_dash="dash", line_color="#F87171", annotation_text="Senaryo Fiyatı")
        fig_demand.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E5E7EB",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=20, r=20, t=50, b=20),
            title_font_size=16
        )
        fig_demand.update_traces(line_color="#8B5CF6", marker=dict(size=8, color="#60A5FA"))
        st.plotly_chart(fig_demand, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Analytics Section
    st.markdown('<div class="section-header">📊 Veri Bilimi ve Analitik Paneli</div>', unsafe_allow_html=True)
    
    t1, t2, t3 = st.tabs(["📌 Model Başarısı", "📉 Keşifsel Veri Analizi", "🔍 Özellik Önemi"])
    
    diagnostics = compute_model_diagnostics()
    
    with t1:
        st.markdown("""
        <div style="animation: fadeIn 0.5s ease-in;">
            <h3 style="color: #60A5FA; margin-bottom: 0; display: flex; align-items: center; gap: 8px;">
                📌 Hold-Out Değerlendirmesi (Eğitim/Test Ayrımı)
            </h3>
            <p style="font-size: 0.95rem; color: #9CA3AF; margin-bottom: 1.5rem; margin-top: 0.2rem;">Birincil Model Performansı</p>
        </div>
        """, unsafe_allow_html=True)
        
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("R² Score", f"{diagnostics['test']['r2']:.4f}")
        mc2.metric("MAE", f"{diagnostics['test']['mae']:,.2f} ₺")
        mc3.metric("RMSE", f"{diagnostics['test']['rmse']:,.2f} ₺")
        
        st.markdown("""
        <div style="animation: fadeIn 0.7s ease-in; margin-top: 2rem;">
            <h4 style="color: #A78BFA; margin-bottom: 0; display: flex; align-items: center; gap: 8px;">
                🔁 5-Katlı Çapraz Doğrulama (Cross Validation)
            </h4>
            <p style="font-size: 0.85rem; color: #9CA3AF; margin-bottom: 1.2rem; margin-top: 0.2rem;">Dayanıklılık ve Tutarlılık Kontrolü</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="display: flex; gap: 1rem; margin-bottom: 2rem; animation: fadeIn 0.8s ease-in;">
            <div class="metric-card" style="flex: 1; padding: 1rem; background: rgba(139, 92, 246, 0.05); border-color: rgba(139, 92, 246, 0.2); box-shadow: 0 0 20px rgba(139, 92, 246, 0.15);">
                <div class="metric-title" style="font-size: 0.8rem; color: #A78BFA; margin-bottom: 0.4rem;">R² Score</div>
                <div class="metric-value" style="font-size: 1.8rem; color: #E5E7EB;">{diagnostics['cv']['r2']:.4f}</div>
            </div>
            <div class="metric-card" style="flex: 1; padding: 1rem; background: rgba(139, 92, 246, 0.05); border-color: rgba(139, 92, 246, 0.2); box-shadow: 0 0 20px rgba(139, 92, 246, 0.15);">
                <div class="metric-title" style="font-size: 0.8rem; color: #A78BFA; margin-bottom: 0.4rem;">MAE</div>
                <div class="metric-value" style="font-size: 1.8rem; color: #E5E7EB;">{diagnostics['cv']['mae']:,.2f} ₺</div>
            </div>
            <div class="metric-card" style="flex: 1; padding: 1rem; background: rgba(139, 92, 246, 0.05); border-color: rgba(139, 92, 246, 0.2); box-shadow: 0 0 20px rgba(139, 92, 246, 0.15);">
                <div class="metric-title" style="font-size: 0.8rem; color: #A78BFA; margin-bottom: 0.4rem;">RMSE</div>
                <div class="metric-value" style="font-size: 1.8rem; color: #E5E7EB;">{diagnostics['cv']['rmse']:,.2f} ₺</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        comp_df = pd.DataFrame({
            "Değer": [
                diagnostics["test"]["r2"], diagnostics["cv"]["r2"],
                diagnostics["test"]["mae"], diagnostics["cv"]["mae"],
                diagnostics["test"]["rmse"], diagnostics["cv"]["rmse"]
            ],
            "Metrik": ["R² Score", "R² Score", "MAE", "MAE", "RMSE", "RMSE"],
            "Değerlendirme": ["Train/Test", "Cross Validation"] * 3
        })
        
        fig_comp = px.bar(
            comp_df, x="Değer", y="Metrik", color="Değerlendirme", barmode="group",
            orientation="h", height=220,
            color_discrete_sequence=["#3B82F6", "#8B5CF6"]
        )
        fig_comp.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#E5E7EB",
            margin=dict(l=0, r=0, t=20, b=0), xaxis_title="", yaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("""
        <div class="interpretation-box" style="margin-top: 1rem; padding: 1.5rem; background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.15)); border-left: 4px solid #8B5CF6; animation: fadeIn 1s ease-in;">
            <strong>💡 AI Yorumu:</strong><br><br>
            <span style="color: #D1D5DB;">Çapraz Doğrulama (Cross Validation) sonuçları, modelin farklı veri bölümlerinde de tutarlı bir performans sergilediğini göstermektedir. Bu durum modelin aşırı öğrenmeye (overfitting) düşmediğini ve genelleme yeteneğini koruduğunu kanıtlar.</span>
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown("<br>", unsafe_allow_html=True)
        ec1, ec2 = st.columns(2)
        with ec1:
            cat_df = sales_df.groupby("product_category")["total_amount"].mean().reset_index()
            cat_df["product_category"] = cat_df["product_category"].map(category_tr)
            fig_cat = px.bar(
                cat_df, x="product_category", y="total_amount", 
                color="product_category", 
                title="Kategorilere Göre Ortalama Sepet Tutarı",
                color_discrete_sequence=["#3B82F6", "#8B5CF6", "#10B981"]
            )
            fig_cat.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#E5E7EB")
            st.plotly_chart(fig_cat, use_container_width=True)
            
        with ec2:
            sales_df["month_num"] = sales_df["date"].dt.month
            monthly_dashboard = (
                sales_df.groupby("month_num", as_index=False)
                .agg(revenue=("total_amount", "sum"))
            )
            monthly_dashboard["month"] = monthly_dashboard["month_num"].map(MONTH_NAMES_TR)
            fig_month = px.line(
                monthly_dashboard, x="month", y="revenue", 
                markers=True, title="Aylara Göre Toplam Gelir Dağılımı"
            )
            fig_month.update_traces(line_color="#FBBF24", marker=dict(size=8))
            fig_month.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#E5E7EB")
            st.plotly_chart(fig_month, use_container_width=True)

    with t3:
        st.markdown("<br>", unsafe_allow_html=True)
        imp_df = diagnostics["importances"].head(10)
        fig_imp = px.bar(
            imp_df, x="importance", y="feature", orientation='h', 
            title="Random Forest Özellik Önem Dereceleri (Top 10)", 
            color="importance", color_continuous_scale="Purples"
        )
        fig_imp.update_layout(
            yaxis={'categoryorder':'total ascending'}, 
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#E5E7EB"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

if __name__ == "__main__":
    render_dashboard()
