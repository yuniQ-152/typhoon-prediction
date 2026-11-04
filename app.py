from pathlib import Path
import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "typhoon_data.csv"
MODEL_PATH = BASE_DIR / "typhoon_rf_model_small.pkl"

FEATURES = ["PREV_LAT", "PREV_LON", "LAT", "LON", "TIME_GAP", "NEXT_GAP"]

st.set_page_config(page_title="태풍 이동 경로 분석 및 예측", page_icon="🌀", layout="wide")
st.title("🌀 태풍 이동 경로 분석 및 예측")
st.caption("과거 태풍 관측 자료를 분석하고 머신러닝 모델의 예측 결과를 확인하는 교육용 웹앱입니다.")
st.warning("교육·탐구 목적의 결과입니다. 공식 예보가 아니며 실제 안전 판단이나 대피 결정에 사용하지 마세요.")

@st.cache_data
def load_data(path, mtime):
    data = pd.read_csv(path)
    required = {"SID", "NAME", "SEASON", "ISO_TIME", "LAT", "LON"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError("CSV에 필요한 열이 없습니다: " + ", ".join(sorted(missing)))
    data["ISO_TIME"] = pd.to_datetime(data["ISO_TIME"], errors="coerce", utc=True)
    for col in ["LAT", "LON", "SEASON", "WMO_WIND", "WMO_PRES"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["SID", "ISO_TIME", "LAT", "LON"]).copy()
    # IBTrACS 경도가 0~360으로 저장된 경우 -180~180으로 변환
    data["LON"] = ((data["LON"] + 180) % 360) - 180
    data = data[(data["LAT"].between(-90, 90)) & (data["LON"].between(-180, 180))]
    data = data.sort_values(["SID", "ISO_TIME"])
    if "NAME" not in data:
        data["NAME"] = "Unknown"
    if "SEASON" not in data:
        data["SEASON"] = data["ISO_TIME"].dt.year
    return data

@st.cache_resource
def load_model(path, mtime):
    return joblib.load(path)

def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error("웹앱 실행에 필요한 파일이 저장소에 없습니다.")
    st.markdown("GitHub 저장소의 `app.py`와 같은 폴더(저장소 최상위)에 다음 두 파일을 업로드해 주세요:")
    st.code("typhoon_data.csv\ntyphoon_rf_model_small.pkl")
    st.markdown("이 파일들은 Google Colab에서 전처리된 `typhoon_df`와 학습된 `model`을 저장해 만들어야 합니다.")
    st.stop()

try:
    data = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error("데이터 또는 모델 파일을 읽는 중 문제가 발생했습니다.")
    st.exception(exc)
    st.stop()

st.sidebar.header("분석 설정")
storms = data[["SID", "NAME", "SEASON"]].drop_duplicates("SID").copy()
storms["SEASON"] = pd.to_numeric(storms["SEASON"], errors="coerce")
storms["display"] = storms["SEASON"].fillna(0).astype(int).astype(str) + " · " + storms["NAME"].fillna("Unknown").astype(str) + " · " + storms["SID"].astype(str)
storms = storms.sort_values(["SEASON", "NAME"], ascending=[False, True])
selected_display = st.sidebar.selectbox("태풍 선택", storms["display"].tolist())
selected_sid = storms.loc[storms["display"] == selected_display, "SID"].iloc[0]
storm = data[data["SID"] == selected_sid].sort_values("ISO_TIME").copy()

if len(storm) < 3:
    st.warning("선택한 태풍의 관측 자료가 충분하지 않습니다. 다른 태풍을 선택해 주세요.")
    st.stop()

latest = storm.iloc[-1]
horizon = st.sidebar.selectbox("예측 시간 간격", [6, 12, 18, 24], index=1)
st.caption(f"선택한 태풍: {selected_display} · 관측 지점 {len(storm)}개")

left, right = st.columns(2)
with left:
    st.subheader("태풍 이동 경로")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(storm["LON"], storm["LAT"], marker="o", markersize=2, linewidth=1.5, label="Observed track")
    ax.scatter([latest["LON"]], [latest["LAT"]], marker="*", s=130, label="Latest position")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Historical track")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("최근 관측 정보")
    st.metric("최근 위도", f"{latest['LAT']:.2f}°")
    st.metric("최근 경도", f"{latest['LON']:.2f}°")
    st.write("최근 관측 시각:", latest["ISO_TIME"].strftime("%Y-%m-%d %H:%M UTC"))
    if "WMO_WIND" in storm.columns and storm["WMO_WIND"].notna().any():
        st.line_chart(storm.set_index("ISO_TIME")["WMO_WIND"].dropna(), y_label="Wind (knots)")
    if "WMO_PRES" in storm.columns and storm["WMO_PRES"].notna().any():
        st.line_chart(storm.set_index("ISO_TIME")["WMO_PRES"].dropna(), y_label="Pressure (hPa)")

st.subheader("AI 이동 위치 예측")
# Use the most recent pair of observations to estimate recent movement speed.
prev = storm.iloc[-2]
gap = (latest["ISO_TIME"] - prev["ISO_TIME"]).total_seconds() / 3600
if not np.isfinite(gap) or gap <= 0:
    st.error("최근 관측 시간 간격이 올바르지 않아 예측할 수 없습니다.")
    st.stop()

input_df = pd.DataFrame([{
    "PREV_LAT": float(prev["LAT"]),
    "PREV_LON": float(prev["LON"]),
    "LAT": float(latest["LAT"]),
    "LON": float(latest["LON"]),
    "TIME_GAP": float(gap),
    "NEXT_GAP": float(horizon),
}])[FEATURES]

if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)
    if expected != FEATURES:
        st.error("저장된 모델의 입력 열 순서가 웹앱과 다릅니다.")
        st.write("모델 입력 열:", expected)
        st.write("웹앱 입력 열:", FEATURES)
        st.stop()

if st.button("AI 예측 실행", type="primary"):
    try:
        pred = np.asarray(model.predict(input_df)).reshape(-1)
        if len(pred) < 2:
            raise ValueError("모델이 위도와 경도 두 값을 반환하지 않았습니다.")
        pred_lat = float(pred[0])
        pred_lon = float(((pred[1] + 180) % 360) - 180)
        if not (-90 <= pred_lat <= 90):
            raise ValueError("예측 위도가 유효한 범위를 벗어났습니다.")
        distance = float(haversine_km(float(latest["LAT"]), float(latest["LON"]), pred_lat, pred_lon))
        c1, c2, c3 = st.columns(3)
        c1.metric("예측 위도", f"{pred_lat:.2f}°")
        c2.metric("예측 경도", f"{pred_lon:.2f}°")
        c3.metric("최근 위치에서의 거리", f"{distance:.1f} km")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(storm["LON"], storm["LAT"], marker="o", markersize=2, label="Observed track")
        ax.scatter([latest["LON"]], [latest["LAT"]], s=80, label="Latest position")
        ax.scatter([pred_lon], [pred_lat], marker="X", s=100, label=f"Prediction (+{horizon}h)")
        ax.plot([latest["LON"], pred_lon], [latest["LAT"], pred_lat], linestyle="--", label="Predicted segment")
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_title("Observed track and model prediction")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        st.info("표시한 거리는 최근 관측 위치와 예측 위치 사이의 직선거리이며, 예측 정확도 자체를 뜻하지는 않습니다.")
    except Exception as exc:
        st.error("예측 중 오류가 발생했습니다.")
        st.exception(exc)

with st.expander("데이터 일부 보기"):
    st.dataframe(storm[["ISO_TIME", "NAME", "SEASON", "LAT", "LON"] + [c for c in ["WMO_WIND", "WMO_PRES"] if c in storm.columns]].tail(20), use_container_width=True)

st.caption("데이터 출처: NOAA IBTrACS. 이 앱은 학교 탐구 및 교육용이며 공식 태풍 예보 시스템이 아닙니다.")
