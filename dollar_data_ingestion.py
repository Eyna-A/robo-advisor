
import csv
import sqlite3
import sys
import pandas as pd

OUTPUT_CSV = "dollar_rates.csv"
OUTPUT_DB = "macro_data.db"

GITHUB_RAW_URL = "https://raw.githubusercontent.com/kooroshkz/Dollar-Rial-Toman-Live-Price-Dataset/main/data/Dollar_Rial_Price_Dataset.csv"


def sync_from_verified_dataset():
    print("🔄 Connecting to Verified USD/IRR Historical Dataset Ingestion Pipeline...")

    try:
        df = pd.read_csv(GITHUB_RAW_URL)
        print(f"✅ Successfully downloaded {len(df)} historical records.")
    except Exception as e:
        print(f"❌ Failed to fetch dataset from GitHub: {e}")
        sys.exit(1)

    required_cols = ["Persian Date", "Close Price"]
    if not all(col in df.columns for col in required_cols):
        print("❌ Dataset structure has changed. Checking fallback columns...")
        return False

    processed_records = {}

    print("📊 Formatting and filtering data for 1395-1405 trading regimes...")

    for _, row in df.iterrows():
        try:
            p_date_str = str(row["Persian Date"]).strip()
            if "/" in p_date_str:
                jalali_int = int(p_date_str.replace("/", ""))
            else:
                continue

            year = jalali_int // 10000
            if not (1395 <= year <= 1405):
                continue

            price_rial = float(row["Close Price"])
            rate_toman = int(price_rial / 10)

            processed_records[jalali_int] = rate_toman

        except (ValueError, TypeError):
            continue

    if not processed_records:
        print("❌ No records matched the 1395-1405 timeframe.")
        return False

    sorted_data = sorted(processed_records.items())


    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["jalali_date", "dollar_rate"])
        writer.writerows(sorted_data)
    print(
        f"💾 CSV pipeline updated: {len(sorted_data)} rows saved to '{OUTPUT_CSV}'"
    )

    try:
        conn = sqlite3.connect(OUTPUT_DB)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS dollar_history (
                jalali_date INTEGER PRIMARY KEY,
                dollar_rate INTEGER
            )
        """
        )
        c.executemany(
            "INSERT OR REPLACE INTO dollar_history VALUES (?, ?)", sorted_data
        )
        conn.commit()
        conn.close()
        print(f"🗄️ Local Data Warehouse synced successfully! ('{OUTPUT_DB}')")
    except Exception as db_err:
        print(f"⚠️ Database storage failed: {db_err}")

    print(
        f"\n Done! Data range: {sorted_data[0][0]} -> {sorted_data[-1][0]} | Latest Close: {sorted_data[-1][1]:,} Toman"
    )
    return True


if __name__ == "__main__":
    sync_from_verified_dataset()