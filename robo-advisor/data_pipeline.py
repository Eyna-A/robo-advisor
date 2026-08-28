

import os
import sqlite3
import logging
import time
from contextlib import contextmanager
from typing import Optional

import pandas as pd
import numpy as np
import requests

try:
    import pytse_client as tse
except ImportError: 
    tse = None

try:
    import jdatetime
except ImportError:
    jdatetime = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("data_pipeline")

DB_NAME = "tsetmc_market_data.db"
OUTPUT_DIR = "excel_outputs"


DOLLAR_API_URL = "https://raw.githubusercontent.com/pesiran/irandata/master/fx/usd_irr_daily.json"


DOLLAR_CSV_PATH = "dollar_rates.csv"
REQUEST_TIMEOUT_SECONDS = 15


MIN_VALID_JALALI_YEAR = 1380

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  

os.makedirs(OUTPUT_DIR, exist_ok=True)


@contextmanager
def get_connection():
    """
    مدیریت امن اتصال SQLite با تنظیمات بهینه برای bulk operations.

    WAL mode اجازه می‌دهد خواندن (مثلاً live_predictor) و نوشتن هم‌زمان
    انجام شود. synchronous=NORMAL سرعت insert را ۳-۵x افزایش می‌دهد با
    ریسک بسیار کم (فقط در صورت قطع برق ممکن است آخرین تراکنش از دست برود).
    """
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA cache_size = -64000")  
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    """ایجاد جداول دیتابیس و INDEXهای حیاتی برای کارایی JOIN."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS instruments(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL UNIQUE,
            name TEXT
        );""")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_prices(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instrument_id INTEGER NOT NULL,
            jalali_date INTEGER NOT NULL,
            open_price REAL,
            high_price REAL,
            low_price REAL,
            close_price REAL,
            volume INTEGER,
            FOREIGN KEY (instrument_id) REFERENCES instruments(id),
            UNIQUE (instrument_id, jalali_date)
        );""")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS macro_data(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            jalali_date INTEGER NOT NULL UNIQUE,
            dollar_rate REAL NOT NULL
        );""")

  
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_daily_prices_jalali_date
        ON daily_prices(jalali_date);
        """)
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_daily_prices_instrument_date
        ON daily_prices(instrument_id, jalali_date);
        """)
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_macro_data_jalali_date
        ON macro_data(jalali_date);
        """)
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_instruments_ticker
        ON instruments(ticker);
        """)

        cursor.execute("ANALYZE;")

    logger.info("✓ دیتابیس آماده است (با INDEXهای بهینه‌شده).")


def purge_invalid_dates():
    """
    پاک‌سازی ردیف‌های تاریخ‌نامعتبر از هر سه جدول (daily_prices، macro_data،
    و instruments یتیم). این تابع idempotent است: اگر چیزی برای پاک‌سازی
    نباشد، فقط لاگ می‌کند و کاری نمی‌کند — پس اجرای خودکارش در هر بار
    شروع پایپ‌لاین بی‌خطر و ارزان است.
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM daily_prices WHERE jalali_date < ?",
            (MIN_VALID_JALALI_YEAR * 10000,)
        )
        bad_count = cursor.fetchone()[0]
        if bad_count > 0:
            cursor.execute(
                "SELECT instrument_id, jalali_date FROM daily_prices WHERE jalali_date < ? LIMIT 20",
                (MIN_VALID_JALALI_YEAR * 10000,)
            )
            sample = cursor.fetchall()
            logger.warning(f"🧹 حذف {bad_count} ردیف نامعتبر از daily_prices "
                           f"(نمونه: {sample[:5]}{'...' if bad_count > 5 else ''})")
            cursor.execute(
                "DELETE FROM daily_prices WHERE jalali_date < ?",
                (MIN_VALID_JALALI_YEAR * 10000,)
            )
        else:
            logger.info("✓ هیچ ردیف تاریخ نامعتبری در daily_prices یافت نشد.")

        cursor.execute(
            "SELECT COUNT(*) FROM macro_data WHERE jalali_date < ?",
            (MIN_VALID_JALALI_YEAR * 10000,)
        )
        bad_macro = cursor.fetchone()[0]
        if bad_macro > 0:
            logger.warning(f"🧹 حذف {bad_macro} ردیف نامعتبر از macro_data...")
            cursor.execute(
                "DELETE FROM macro_data WHERE jalali_date < ?",
                (MIN_VALID_JALALI_YEAR * 10000,)
            )

        cursor.execute("""
            DELETE FROM instruments
            WHERE id NOT IN (SELECT DISTINCT instrument_id FROM daily_prices)
        """)

        logger.info("✅ پاک‌سازی تاریخ‌های نامعتبر تکمیل شد.")


def _is_valid_jalali_date(jalali_int) -> bool:
    """اعتبارسنجی تاریخ شمسی با jdatetime (در صورت نصب)."""
    s = str(jalali_int)
    if len(s) != 8:
        return False
    year, month, day = int(s[:4]), int(s[4:6]), int(s[6:8])
    if not (MIN_VALID_JALALI_YEAR <= year <= 1500):
        return False
    if jdatetime is not None:
        try:
            jdatetime.date(year, month, day)
            return True
        except ValueError:
            return False
    return 1 <= month <= 12 and 1 <= day <= 31


def _extract_date_source(history: pd.DataFrame) -> pd.Series:
    """
    🔴 رفع باگ واقعی: منبع درست تاریخ را در DataFrame خروجی pytse_client پیدا می‌کند.

    لاگ اجرای واقعی‌ات نشان داد history.index گاهی یک عدد صحیح موقعیتی ساده
    (0, 1, 2, ..., 284, 285, ...) است، نه Timestamp — یعنی فرض قبلی کد
    (که تاریخ در index است) همیشه درست نیست. این تابع ابتدا دنبال یک ستون
    صریح تاریخ می‌گردد (رایج‌ترین نام‌ها در pytse_client)، و فقط اگر پیدا
    نشد به index برمی‌گردد؛ در آن حالت هم یک هشدار تشخیصی می‌زند تا اگر
    باز هم مشکل بود، بتوانی دقیقاً ستون‌های واقعی history را ببینی.
    """
    candidate_cols = ["date", "Date", "jdate", "j_date", "jalali_date"]
    for col in candidate_cols:
        if col in history.columns:
            logger.info(f"ℹ️ منبع تاریخ: ستون '{col}' (نه index) استفاده می‌شود.")
            return history[col]

    logger.warning(
        "⚠️ هیچ ستون تاریخ صریحی در history پیدا نشد "
        f"(ستون‌های موجود: {list(history.columns)})؛ به history.index بازمی‌گردیم. "
        "اگر بعد از این هم همه‌ی ردیف‌ها رد شدند، به‌احتمال زیاد تاریخ واقعی در "
        "ستون دیگری با نام متفاوت است — لیست ستون‌های بالا را بررسی کن."
    )
    date_series = history.index.to_series().reset_index(drop=True)
    date_series.index = history.index
    return date_series


def _gregorian_to_jalali_series(dates: pd.Series) -> pd.Series:
    """
    تبدیل وکتوریزه‌ی سری تاریخ‌های میلادی به اعداد شمسی ۸ رقمی.

    محافظت حیاتی (از نسخه‌ی «واقعی» گرفته شد): قبل از تبدیل، نوع هر مقدار
    صراحتاً بررسی می‌شود. اگر ورودی Timestamp/str/datetime64 نباشد (مثلاً
    یک عدد صحیح موقعیتی خام از یک index غیرقابل‌اعتماد)، رد می‌شود تا
    هرگز بی‌سروصدا به تاریخ epoch (1348/10/11) تبدیل نشود.
    """
    if jdatetime is None:
        logger.error("❌ jdatetime نصب نیست. pip install jdatetime")
        return pd.Series([None] * len(dates))

    def _convert_single(val):
        if pd.isna(val):
            return None
        if not isinstance(val, (pd.Timestamp, str, np.datetime64)):
            return None
        try:
            ts = pd.Timestamp(val)
            if pd.isna(ts):
                return None
            j = jdatetime.date.fromgregorian(date=ts.date())
            result = int(f"{j.year:04d}{j.month:02d}{j.day:02d}")
            return result if _is_valid_jalali_date(result) else None
        except Exception:
            return None

    return dates.apply(_convert_single)


def clean_and_validate_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    اعتبارسنجی وکتوریزه‌ی داده‌های خام — جایگزین حلقه‌ی row-by-row.
    سرعت برای ۲۵۰۰ ردیف: ~50x سریع‌تر از نسخه‌ی row-by-row.
    """
    if df.empty:
        return df

    required = ["open", "high", "low", "close", "volume", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"ستون‌های گمشده: {missing}")
        return df.iloc[0:0]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = (
        df["open"].notna() &
        df["high"].notna() &
        df["low"].notna() &
        df["close"].notna() &
        df["volume"].notna() &
        (df["open"] > 0) &
        (df["close"] > 0) &
        (df["high"] > 0) &
        (df["low"] > 0) &
        (df["high"] >= df["low"]) &
        (df["close"] >= df["low"] - 1e-6) &
        (df["close"] <= df["high"] + 1e-6) &
        (df["volume"] >= 0) &
        df["date"].apply(_is_valid_jalali_date)
    )

    rejected = (~valid).sum()
    if rejected > 0:
        logger.info(f"   {rejected} ردیف نامعتبر فیلتر شد.")

    return df[valid].reset_index(drop=True)


def fetch_and_save_real_data(target_ticker: str, retry_count: int = 0) -> bool:
    """دانلود تاریخچه با Retry و Exponential Backoff، با منبع تاریخ اصلاح‌شده."""
    if tse is None:
        logger.error("❌ pytse_client نصب نیست. `pip install pytse-client` را اجرا کنید.")
        return False

    logger.info(f"🔄 دانلود '{target_ticker}' (تلاش {retry_count + 1}/{MAX_RETRIES})...")

    try:
        ticker_info = tse.Ticker(target_ticker)
        history = ticker_info.history

        if history is None or history.empty:
            logger.warning(f"⚠️ دیتایی برای {target_ticker} پیدا نشد.")
            return False

        history = history.copy()

        
        date_source = _extract_date_source(history)
        history["jalali_date"] = _gregorian_to_jalali_series(date_source)
        history = history.dropna(subset=["jalali_date"])

        if history.empty:
            logger.warning(
                f"⚠️ هیچ تاریخ معتبری برای {target_ticker} تولید نشد. "
                f"ستون‌های history: {list(ticker_info.history.columns)} — "
                "این را بررسی کن تا منبع واقعی تاریخ را پیدا کنیم."
            )
            return False

        close_col = "adjClose" if "adjClose" in history.columns else "close"

        formatted = pd.DataFrame({
            "date": history["jalali_date"].astype(int),
            "open": history["open"],
            "high": history["high"],
            "low": history["low"],
            "close": history[close_col],
            "volume": history["volume"],
        })

        formatted = clean_and_validate_vectorized(formatted)

        if formatted.empty:
            logger.warning(f"⚠️ پس از اعتبارسنجی، هیچ ردیفی برای {target_ticker} باقی نماند.")
            return False

        valid_rows = formatted.to_numpy().tolist()
        save_to_pipeline(target_ticker, valid_rows)
        return True

    except (requests.RequestException, ConnectionError, TimeoutError) as e:
        if retry_count < MAX_RETRIES - 1:
            wait = RETRY_BACKOFF_BASE * (2 ** retry_count)
            logger.warning(f"⚠️ خطای شبکه برای {target_ticker}: {e}. تلاش مجدد در {wait}s...")
            time.sleep(wait)
            return fetch_and_save_real_data(target_ticker, retry_count + 1)
        logger.error(f"❌ خطای دریافت داده برای {target_ticker} پس از {MAX_RETRIES} تلاش: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ خطای غیرمنتظره برای {target_ticker}: {e}", exc_info=True)
        return False


def save_to_pipeline(ticker: str, valid_rows: list):
    """
    ذخیره‌ی دسته‌ای با ON CONFLICT DO UPDATE.

    نکته: cursor.rowcount بعد از executemany در sqlite3 پایتون غیرقابل‌اعتماد
    است، به‌خصوص با تداخل UNIQUE. به‌جایش با conn.total_changes قبل/بعد،
    تعداد واقعی ردیف‌های تازه درج/به‌روزرسانی‌شده اندازه‌گیری می‌شود.
    """
    if not valid_rows:
        logger.warning(f"⚠️ {ticker}: هیچ ردیفی برای ذخیره نیست.")
        return

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO instruments(ticker, name) VALUES (?, ?)",
            (ticker, ticker)
        )
        cursor.execute("SELECT id FROM instruments WHERE ticker = ?", (ticker,))
        row = cursor.fetchone()
        if row is None:
            logger.error(f"❌ درج نماد {ticker} ناموفق بود.")
            return
        instrument_id = row[0]

        full_rows = [(instrument_id, *r) for r in valid_rows]

        changes_before = conn.total_changes
        cursor.executemany("""
            INSERT INTO daily_prices
            (instrument_id, jalali_date, open_price, high_price, low_price, close_price, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(instrument_id, jalali_date) DO UPDATE SET
                open_price = excluded.open_price,
                high_price = excluded.high_price,
                low_price = excluded.low_price,
                volume = excluded.volume,
                close_price = excluded.close_price
        """, full_rows)
        actually_inserted = conn.total_changes - changes_before

        already_present = len(full_rows) - actually_inserted
        logger.info(f"📊 {ticker}: {actually_inserted} روز جدید/به‌روزرسانی‌شده، "
                    f"{already_present} روز بدون تغییر.")


def _load_dollar_from_csv(csv_path: str) -> dict:
    """
    خواندن نرخ دلار از فایل CSV محلی (منبع اصلی و توصیه‌شده).
    اگر فایل وجود نداشت، دیکشنری خالی برمی‌گرداند (نه خطا) تا fallback به API
    امکان‌پذیر باشد.
    """
    if not os.path.exists(csv_path):
        return {}

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception as e:
        logger.warning(f"⚠️ خواندن فایل CSV دلار ناموفق بود: {e}")
        return {}

    required_cols = {"jalali_date", "dollar_rate"}
    if not required_cols.issubset(df.columns):
        logger.warning(f"⚠️ فایل CSV دلار باید ستون‌های {required_cols} را داشته باشد؛ "
                        f"ستون‌های موجود: {list(df.columns)}")
        return {}

    df = df.dropna(subset=["jalali_date", "dollar_rate"])
    return dict(zip(df["jalali_date"].astype(str), df["dollar_rate"]))


def _fetch_dollar_from_api(api_url: str) -> dict:
    """
    تلاش برای دریافت نرخ دلار از API آنلاین.
    در صورت هر نوع خطا (شبکه، 404، JSON نامعتبر)، دیکشنری خالی برمی‌گرداند
    (نه exception) تا فراخوان بتواند تصمیم بگیرد چه کند.
    """
    try:
        response = requests.get(api_url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        dollar_json = response.json()
        if not isinstance(dollar_json, dict) or not dollar_json:
            logger.warning("⚠️ پاسخ API دلار ساختار نامعتبر داشت.")
            return {}
        return dollar_json
    except requests.RequestException as e:
        logger.warning(f"⚠️ دریافت از API دلار ناموفق بود: {e}")
        return {}
    except ValueError as e:
        logger.warning(f"⚠️ پاسخ API دلار JSON معتبر نبود: {e}")
        return {}


def fetch_and_inject_real_dollar_data(csv_path: str = DOLLAR_CSV_PATH, api_url: str = DOLLAR_API_URL):
    """
    تزریق نرخ دلار آزاد به دیتابیس — با دو منبع به ترتیب اولویت:

      ۱) فایل CSV محلی (DOLLAR_CSV_PATH) — منبع اصلی و توصیه‌شده، چون
         APIهای رایگان دلار ایران ناپایدارند.
      ۲) اگر CSV موجود نبود، تلاش برای API آنلاین (ممکن است در دسترس نباشد).

    اگر هیچ‌کدام داده‌ای ندادند، Pipeline متوقف می‌شود (fail-fast) — چون
    داده‌ی مصنوعی دلار مستقیم وارد فیچرهای مدل می‌شود و خرابی آن بی‌سروصدا
    کیفیت مدل را خراب می‌کند.
    """
    logger.info("💵 دریافت نرخ دلار آزاد...")

    dollar_data = _load_dollar_from_csv(csv_path)
    source_used = None

    if dollar_data:
        source_used = f"CSV محلی ({csv_path})"
        logger.info(f"✅ {len(dollar_data)} رکورد از فایل CSV محلی خوانده شد.")
    else:
        logger.info(f"ℹ️ فایل CSV دلار پیدا نشد یا خالی بود ({csv_path})؛ تلاش برای API آنلاین...")
        dollar_data = _fetch_dollar_from_api(api_url)
        if dollar_data:
            source_used = f"API آنلاین ({api_url})"
            logger.info(f"✅ {len(dollar_data)} رکورد از API دریافت شد.")

    if not dollar_data:
        raise RuntimeError(
            "دریافت داده‌ی دلار از هیچ منبعی ممکن نشد (نه CSV محلی، نه API آنلاین).\n"
            f"  راه‌حل: یک فایل CSV با ستون‌های jalali_date,dollar_rate در مسیر "
            f"'{csv_path}' بسازید.\n"
            "  منبع پیشنهادی: آرشیو تاریخی tgju.org یا bonbast.com (دستی export کنید).\n"
            "Pipeline متوقف شد تا داده‌ی مصنوعی وارد مدل نشود."
        )

    with get_connection() as conn:
        cursor = conn.cursor()
        records = []
        skipped = 0
        for date_str, rate in dollar_data.items():
            try:
                jalali_date = int(str(date_str).replace("-", "")[:8])
                rate_val = float(rate)
                if rate_val <= 0 or not _is_valid_jalali_date(jalali_date):
                    skipped += 1
                    continue
                
                records.append((jalali_date, rate_val))
            except (TypeError, ValueError):
                skipped += 1
                continue

        if not records:
            raise RuntimeError("هیچ رکورد معتبری در داده‌ی دلار دریافتی یافت نشد.")

        cursor.executemany(
            "INSERT OR REPLACE INTO macro_data (jalali_date, dollar_rate) VALUES (?, ?)",
            records
        )
        logger.info(f"✅ {len(records)} روز نرخ دلار از «{source_used}» ذخیره شد "
                    f"({skipped} رکورد نامعتبر رد شد).")


def export_all_to_excel():
    """استخراج داده‌ی هر نماد (هم‌تراز با دلار) به فایل اکسل مجزا."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT ticker FROM instruments")
        tickers = [row[0] for row in cursor.fetchall()]

        logger.info(f"💾 ساخت فایل‌های اکسل در پوشه '{OUTPUT_DIR}'...")

        for ticker in tickers:
           
            query = """
            SELECT
                CAST(dp.jalali_date AS INTEGER) as [تاریخ شمسی],
                dp.open_price as [قیمت باز شدن],
                dp.high_price as [بالاترین قیمت],
                dp.low_price as [پایین‌ترین قیمت],
                dp.close_price as [قیمت پایانی],
                dp.volume as [حجم معاملات],
                md.dollar_rate as [نرخ دلار واقعی]
            FROM daily_prices dp
            JOIN instruments i ON dp.instrument_id = i.id
            JOIN macro_data md ON CAST(dp.jalali_date AS INTEGER) = CAST(md.jalali_date AS INTEGER)
            WHERE i.ticker = ?
            ORDER BY dp.jalali_date ASC
            """
            df = pd.read_sql_query(query, conn, params=(ticker,))

            if not df.empty:
                file_path = os.path.join(OUTPUT_DIR, f"{ticker}_Clean_Data.xlsx")
                df.to_excel(file_path, index=False)
                logger.info(f"✅ {ticker}: {df.shape[0]} ردیف ذخیره شد.")
            else:
                logger.warning(f"⚠️ نماد {ticker} به دلیل عدم تلاقی تاریخ با دیتای دلار صادر نشد.")

    logger.info("🎉 عملیات با موفقیت پایان یافت.")


def run_pipeline(target_tickers, skip_dollar: bool = False):
    """نقطه‌ی ورود قابل‌فراخوانی پایپ‌لاین (برای استفاده در main.py یا تست‌ها)."""
    init_database()
    purge_invalid_dates()

    results = {}
    for ticker in target_tickers:
        results[ticker] = fetch_and_save_real_data(ticker)

    failed = [t for t, ok in results.items() if not ok]
    if failed:
        logger.warning(f"⚠️ نمادهای ناموفق در دانلود: {failed}")

    if not any(results.values()):
        raise RuntimeError("هیچ نمادی با موفقیت دانلود نشد؛ پایپ‌لاین متوقف می‌شود.")

    if not skip_dollar:
        fetch_and_inject_real_dollar_data()

    export_all_to_excel()


if __name__ == "__main__":
    TARGET_TICKERS = ["فولاد", "خودرو", "فملی", "شستا"]
    run_pipeline(TARGET_TICKERS)
