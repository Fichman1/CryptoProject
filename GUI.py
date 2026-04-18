import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# --- הגדרות דף ---
st.set_page_config(page_title="AI Model Comparison", layout="wide")

# מכיוון שאנחנו רצים בקולאב, הנתיב תמיד קבוע וידוע!
BASE_DIR = '/content/drive/MyDrive/CryptoProject'

st.sidebar.title("הגדרות (Settings)")
model_choice = st.sidebar.selectbox("בחר מודל להצגה:", ("Transformer (TFT)", "LSTM"))

# קביעת שם הקובץ לפי הבחירה
file_name = 'transformer_dashboard_data.csv' if model_choice == "Transformer (TFT)" else 'lstm_dashboard_data.csv'
DATA_PATH = os.path.join(BASE_DIR, file_name)

st.title(f"📊 Crypto AI Dashboard - {model_choice}")
st.markdown("---")

@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'open_time'}, inplace=True)
    elif 'open_time' not in df.columns:
        df.rename(columns={df.columns[0]: 'open_time'}, inplace=True)
        
    df['open_time'] = pd.to_datetime(df['open_time'])
    df.set_index('open_time', inplace=True)
    return df

df = load_data(DATA_PATH)

if df is None:
    st.warning(f"⚠️ לא נמצא קובץ נתונים עבור {model_choice} בנתיב {DATA_PATH}. וודא שהרצת את האימון בקולאב.")
else:
    # מדדים (Metrics)
    latest_price = df['close'].iloc[-1]
    predicted_price = df['Predicted_Close'].iloc[-1]
    diff = predicted_price - latest_price
    pct = (diff / latest_price) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("מחיר נוכחי", f"${latest_price:,.2f}")
    col2.metric("חיזוי AI (נר הבא)", f"${predicted_price:,.2f}", f"{pct:+.2f}%")
    
    signal = "🟢 LONG" if pct > 0.05 else ("🔴 SHORT" if pct < -0.05 else "⚪ NEUTRAL")
    col3.subheader(f"Signal: {signal}")

    st.markdown("---")
    num_candles = st.sidebar.slider("נרות להצגה:", 50, len(df), 150)
    df_view = df.tail(num_candles)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_view.index, open=df_view['open'], high=df_view['high'], low=df_view['low'], close=df_view['close'], name="Market"))
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view['Predicted_Close'], line=dict(color='cyan', width=2, dash='dot'), name=f"AI {model_choice}"))

    fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, title=f"Actual vs {model_choice} Prediction")
    st.plotly_chart(fig, use_container_width=True)
