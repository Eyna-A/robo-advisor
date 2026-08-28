"""
app.py
======
بک‌اند سبک FastAPI که خروجی واقعی پایپ‌لاین (lightgbm + portfolio_optimizer) را
از فایل‌های اکسل/دیتابیس می‌خواند و در قالب JSON به داشبورد سرو می‌کند.

🆕 این نسخه سه endpoint جدید دارد:
  - GET  /api/backtest-metrics  → backtest_metrics.json خام (DSR/Sharpe/...)
  - GET  /api/model-health      → fold metrics + اهمیت فیچرها از مدل آموزش‌دیده
  - POST /api/pipeline/run      → اجرای main.py به‌صورت subprocess (فقط محلی!)

و در انتها، پوشه‌ی frontend/ را با StaticFiles سرو می‌کند تا دشبورد از همین
سرور (همان origin) لود شود -- بدون نیاز به آدرس API جداگانه در فرانت.

اجرا:
    pip install fastapi uvicorn pandas openpyxl lightgbm
    uvicorn app:app --reload --port 8000
"""

import os
import sys
import json
import logging
import asyncio
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dashboard_api")

# مسیرها نسبت به ریشه‌ی پروژه‌ی پایتون (همان جایی که main.py اجرا می‌شود)
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(__file__), ".."))
LIVE_PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "excel_outputs", "live_market_predictions.xlsx")
EQUITY_CURVE_PATH = os.path.join(PROJECT_ROOT, "excel_outputs", "backtest_equity_curve.xlsx")
BACKTEST_METRICS_PATH = os.path.join(PROJECT_ROOT, "excel_outputs", "backtest_metrics.json")
FIXED_INCOME_KEY = "صندوق درآمد ثابت"

# 🆕 مسیرهای مدل -- برای /api/model-health
MODEL_DIR = os.path.join(PROJECT_ROOT, "ai_models")
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
MODEL_BOOSTER_PATH = os.path.join(MODEL_DIR, "lgb_robo_advisor.txt")

# 🆕 برای /api/pipeline/run -- python.exe که main.py را اجرا می‌کند.
# پیش‌فرض: همان interpreter که خودِ uvicorn با آن اجرا شده. اگر backend و
# پایپ‌لاین دو venv جدا دارند (یکی برای fastapi، یکی برای pandas/lightgbm)،
# این متغیر محیطی را صریح ست کنید:
#   $env:PIPELINE_PYTHON = "E:\StockPrj\pipeline Iran\.venv\Scripts\python.exe"
MAIN_PY_PATH = os.path.join(PROJECT_ROOT, "main.py")
PIPELINE_PYTHON = os.environ.get("PIPELINE_PYTHON", sys.executable)

# 🆕 پوشه‌ی فرانت -- خواهر backend/ است، صرف‌نظر از PROJECT_ROOT (که فقط
# برای مسیر دیتای پایپ‌لاین قابل‌تنظیم است، نه ساختار خودِ ریپو).
FRONTEND_DIR = str(Path(__file__).resolve().parent.parent / "frontend")


def load_backtest_metrics() -> Optional[dict]:
    """بک‌تست فقط یک‌بار (برای یک سناریو) اجرا می‌شود، نه به ازای هر
    risk/horizon؛ پس همین چند عدد واقعی را برای هر درخواستی برمی‌گردانیم —
    به‌جای None که در فرانت به اشتباه به‌صورت +۰٫۰٪ نمایش داده می‌شود."""
    if not os.path.exists(BACKTEST_METRICS_PATH):
        return None
    with open(BACKTEST_METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def jalali_int_to_label(jalali_date) -> str:
    s = str(int(jalali_date))
    return f"{s[:4]}/{s[4:6]}/{s[6:8]}"


def _ensure_project_root_on_path():
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


app = FastAPI(title="Robo Advisor Dashboard API")

# در dev معمولاً فرانت روی پورت دیگری اجرا می‌شود؛ CORS باز است. اگر دشبورد
# را از همین سرور سرو می‌کنید (پایین، StaticFiles)، اصلاً به CORS نیازی
# نیست چون همه‌چیز روی یک origin است -- این middleware فقط برای dev جداگانه
# (مثلاً Live Server) نگه داشته شده.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# /api/rankings
# ---------------------------------------------------------------------------
@app.get("/api/rankings")
def get_rankings():
    """
    تبدیل خروجی live_predictor.py (live_market_predictions.xlsx) به همان شکلی
    که دشبورد انتظار دارد:
        { symbol, alpha_score, predicted_class, price, change_percent,
          drop_prob, neutral_prob, growth_prob, is_stale }
    """
    if not os.path.exists(LIVE_PREDICTIONS_PATH):
        raise HTTPException(
            status_code=503,
            detail="هنوز پیش‌بینی لایو تولید نشده. ابتدا پایپ‌لاین را اجرا کنید.",
        )

    df = pd.read_excel(LIVE_PREDICTIONS_PATH)

    required_cols = {
        'نماد', 'قیمت پایانی', 'امتیاز خرید (Alpha Score)', 'درصد تغییر',
        'احتمال ریزش/عقب‌ماندگی (کلاس ۰)', 'احتمال خنثی/همگام بازار (کلاس ۱)',
        'احتمال رشد شارپ > ۵٪ (کلاس ۲)',
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(status_code=500, detail=f"ستون‌های مورد انتظار یافت نشد: {missing}")

    probs = df[[
        'احتمال ریزش/عقب‌ماندگی (کلاس ۰)',
        'احتمال خنثی/همگام بازار (کلاس ۱)',
        'احتمال رشد شارپ > ۵٪ (کلاس ۲)',
    ]].values
    predicted_class = np.argmax(probs, axis=1)

    stale_col = 'داده قدیمی / مشکوک به توقف نماد'
    has_stale_col = stale_col in df.columns

    out = []
    for i, row in df.iterrows():
        out.append({
            "symbol": row['نماد'],
            "alpha_score": round(float(row['امتیاز خرید (Alpha Score)']), 2),
            "predicted_class": int(predicted_class[i]),
            "price": int(row['قیمت پایانی']),
            "change_percent": float(row['درصد تغییر']),
            "drop_prob": float(row['احتمال ریزش/عقب‌ماندگی (کلاس ۰)']),
            "neutral_prob": float(row['احتمال خنثی/همگام بازار (کلاس ۱)']),
            "growth_prob": float(row['احتمال رشد شارپ > ۵٪ (کلاس ۲)']),
            "is_stale": bool(row[stale_col]) if has_stale_col else False,
        })

    return out


# ---------------------------------------------------------------------------
# /api/equity-curve
# ---------------------------------------------------------------------------
@app.get("/api/equity-curve")
def get_equity_curve():
    """
    خروجی backtester.py (backtest_equity_curve.xlsx) را به فرمت مورد انتظار
    داشبورد ([{date, portfolio_value, market_value}, ...]) تبدیل می‌کند.
    """
    if not os.path.exists(EQUITY_CURVE_PATH):
        raise HTTPException(
            status_code=503,
            detail="هنوز بک‌تست اجرا نشده. backtester.py را با یک start_date آگاهانه اجرا کنید.",
        )

    df = pd.read_excel(EQUITY_CURVE_PATH)
    required_cols = {'date', 'total_value', 'market_value'}
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"ستون‌های {missing} در backtest_equity_curve.xlsx یافت نشد.",
        )

    out = [
        {
            "date": jalali_int_to_label(row['date']),
            "portfolio_value": float(row['total_value']),
            "market_value": float(row['market_value']),
        }
        for _, row in df.iterrows()
    ]
    return out


# ---------------------------------------------------------------------------
# 🆕 /api/backtest-metrics
# ---------------------------------------------------------------------------
@app.get("/api/backtest-metrics")
def get_backtest_metrics():
    """
    خروجی خام backtest_metrics.json (شامل DSR اگر backtester.py را با نسخه‌ی
    به‌روزشده اجرا کرده باشید -- به README/چت مراجعه کنید).
    """
    metrics = load_backtest_metrics()
    if metrics is None:
        raise HTTPException(
            status_code=503,
            detail="هنوز backtest_metrics.json وجود ندارد. backtester.py را اجرا کنید.",
        )
    return metrics


# ---------------------------------------------------------------------------
# 🆕 /api/model-health
# ---------------------------------------------------------------------------
@app.get("/api/model-health")
def get_model_health():
    """
    fold_metrics از model_metadata.json (نوشته‌ی train_model.py) به‌علاوه‌ی
    اهمیت فیچرها -- که train_model.py آن را فقط لاگ می‌کند و ذخیره نمی‌کند،
    پس اینجا مستقیماً از خودِ فایل مدل (.txt) با LightGBM دوباره محاسبه
    می‌شود، دقیقاً با همان FEATURE_COLS که train_model.py استفاده کرده.
    """
    if not os.path.exists(MODEL_METADATA_PATH):
        raise HTTPException(
            status_code=503,
            detail="هنوز مدلی آموزش ندیده. train_model.py را اجرا کنید.",
        )

    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_importance = []
    if os.path.exists(MODEL_BOOSTER_PATH):
        _ensure_project_root_on_path()
        try:
            import lightgbm as lgb
            from train_model import FEATURE_COLS

            booster = lgb.Booster(model_file=MODEL_BOOSTER_PATH)
            gains = booster.feature_importance(importance_type="gain")
            pairs = sorted(zip(FEATURE_COLS, gains), key=lambda p: p[1], reverse=True)
            feature_importance = [[name, round(float(val), 1)] for name, val in pairs[:10]]
        except Exception as e:
            logger.warning(f"⚠️ محاسبه‌ی اهمیت فیچرها ناموفق بود: {e}")

    return {
        "trained_at": metadata.get("trained_at"),
        "folds": metadata.get("fold_metrics", []),
        "feature_importance": feature_importance,
    }


# ---------------------------------------------------------------------------
# /api/portfolio/optimize
# ---------------------------------------------------------------------------
class OptimizeRequest(BaseModel):
    capital: float
    risk_appetite: str  # 'low' | 'medium' | 'high'
    time_horizon: str   # 'short' | 'mid' | 'long'


@app.post("/api/portfolio/optimize")
def post_portfolio_optimize(req: OptimizeRequest):
    _ensure_project_root_on_path()

    try:
        from portfolio_optimizer import optimize_portfolio
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"portfolio_optimizer.py پیدا نشد: {e}")

    if req.risk_appetite not in ('low', 'medium', 'high'):
        raise HTTPException(status_code=400, detail="risk_appetite باید یکی از low/medium/high باشد.")
    if req.time_horizon not in ('short', 'mid', 'long'):
        raise HTTPException(status_code=400, detail="time_horizon باید یکی از short/mid/long باشد.")

    try:
        allocation_df = optimize_portfolio(
            capital=req.capital,
            risk_appetite=req.risk_appetite,
            time_horizon=req.time_horizon,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"{e} — ابتدا live_predictor.py را اجرا کنید.",
        )
    except Exception as e:
        logger.exception("optimize_portfolio failed")
        raise HTTPException(status_code=500, detail=str(e))

    if isinstance(allocation_df, dict):
        weights = {k: v / req.capital for k, v in allocation_df.items()}
        bt = load_backtest_metrics() or {}
        metrics = {
            "total_return": bt.get("total_return"),
            "sharpe_ratio": bt.get("sharpe_ratio"),
            "max_drawdown": bt.get("max_drawdown"),
            "risk_exposure": 0.0,
        }
        return {"portfolio_weights": weights, "metrics": metrics}

    weights = {}
    for _, row in allocation_df.iterrows():
        pct_str = row['وزن از کل سبد'].replace('%', '')
        weights[row['نماد']] = round(float(pct_str) / 100.0, 4)

    bt = load_backtest_metrics() or {}
    metrics = {
        "total_return": bt.get("total_return"),
        "sharpe_ratio": bt.get("sharpe_ratio"),
        "max_drawdown": bt.get("max_drawdown"),
        "risk_exposure": round(1 - weights.get(FIXED_INCOME_KEY, 0.0), 4),
    }

    return {"portfolio_weights": weights, "metrics": metrics}


# ---------------------------------------------------------------------------
# 🆕 /api/pipeline/run
# ---------------------------------------------------------------------------
class PipelineRunResponse(BaseModel):
    status: str
    returncode: int
    stdout_tail: str
    stderr_tail: str


@app.post("/api/pipeline/run", response_model=PipelineRunResponse)
async def run_pipeline():
    """
    main.py را به‌صورت subprocess اجرا می‌کند و تا پایان اجرا صبر می‌کند
    (main.py حدود ۳۰-۹۰ ثانیه طول می‌کشد، طبق لاگ‌های واقعی شما).

    subprocess (نه import مستقیم run_full_pipeline) عمداً انتخاب شده: خودِ
    main.py در صورت خطا sys.exit(1) صدا می‌زند -- اگر مستقیم import و
    فراخوانی می‌شد، همین sys.exit کل پردازش FastAPI را هم می‌کشت.
    asyncio.to_thread هم استفاده شده تا این ۳۰-۹۰ ثانیه، event loop سرور را
    بلاک نکند.

    ⚠️ امنیت: این endpoint هیچ احراز هویتی ندارد -- برای استفاده‌ی محلی
    (127.0.0.1) طراحی شده. هرگز با --host 0.0.0.0 یا پشت یک تونل عمومی این
    سرور را اجرا نکنید، چون یعنی هرکسی که به این پورت برسد می‌تواند
    پایپ‌لاین شما را اجرا کند.
    """
    if not os.path.exists(MAIN_PY_PATH):
        raise HTTPException(
            status_code=404,
            detail=f"main.py یافت نشد در '{MAIN_PY_PATH}' — متغیر محیطی PROJECT_ROOT را بررسی کنید.",
        )

    def _run() -> subprocess.CompletedProcess:
        return subprocess.run(
            [PIPELINE_PYTHON, "main.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )

    logger.info(f"🚀 اجرای پایپ‌لاین: {PIPELINE_PYTHON} main.py  (cwd={PROJECT_ROOT})")

    try:
        result = await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="اجرای پایپ‌لاین بیش از ۱۵ دقیقه طول کشید (timeout).")
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"اجرای python ممکن نشد ({PIPELINE_PYTHON}): {e}. "
                   "اگر backend و پایپ‌لاین venv جدا دارند، PIPELINE_PYTHON را صریح ست کنید.",
        )

    ok = result.returncode == 0
    if ok:
        logger.info("✅ main.py با موفقیت اجرا شد.")
    else:
        logger.error(f"❌ main.py با کد {result.returncode} خارج شد:\n{result.stderr[-4000:]}")

    return {
        "status": "ok" if ok else "error",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "live_predictions_exist": os.path.exists(LIVE_PREDICTIONS_PATH),
        "equity_curve_exists": os.path.exists(EQUITY_CURVE_PATH),
    }


# ---------------------------------------------------------------------------
# 🆕 سرو کردن دشبورد از همین سرور (باید بعد از همه‌ی route های /api باشد،
# وگرنه StaticFiles مسیر "/" کل بقیه را می‌بلعد)
# ---------------------------------------------------------------------------
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    logger.info(f"🖥️  دشبورد از '{FRONTEND_DIR}' روی '/' سرو می‌شود.")
else:
    logger.warning(f"⚠️ پوشه‌ی فرانت پیدا نشد: '{FRONTEND_DIR}' — دشبورد از این سرور سرو نمی‌شود.")
