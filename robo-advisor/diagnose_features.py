import sqlite3
import pandas as pd

DB_NAME = "tsetmc_market_data.db"

def ultimate_debug():
    """
    Inspects date format compatibility and column types between 
    daily_prices and macro_data tables, testing a basic JOIN query.
    """
    conn = sqlite3.connect(DB_NAME)
    print("🔬 --- Starting Date Format Diagnostics --- 🔬\n")
    
    print("📅 [Table daily_prices - Stock Prices]")
    try:
        df_prices = pd.read_sql_query("SELECT jalali_date FROM daily_prices WHERE jalali_date IS NOT NULL LIMIT 5", conn)
        print("Sample Data:")
        print(df_prices)
        print("Data Type (Dtype) in Python:")
        print(df_prices.dtypes)
        if not df_prices.empty:
            val = df_prices['jalali_date'].iloc[0]
            print(f"Exact type of first value: {type(val)} | Length of string/value: {len(str(val))}")
    except Exception as e:
        print(f"❌ Error reading daily_prices: {e}")
        
    print("\n" + "="*50 + "\n")
    
    print("💵 [Table macro_data - Dollar Data]")
    try:
        df_macro = pd.read_sql_query("SELECT jalali_date FROM macro_data WHERE jalali_date IS NOT NULL LIMIT 5", conn)
        print("Sample Data:")
        print(df_macro)
        print("Data Type (Dtype) in Python:")
        print(df_macro.dtypes)
        if not df_macro.empty:
            val = df_macro['jalali_date'].iloc[0]
            print(f"Exact type of first value: {type(val)} | Length of string/value: {len(str(val))}")
    except Exception as e:
        print(f"❌ Error reading macro_data: {e}")

    print("\n" + "="*50 + "\n")

    print("🔗 [Experimental Simple JOIN Test]")
    try:
        test_query = """
        SELECT dp.jalali_date as dp_date, md.jalali_date as md_date, md.dollar_rate
        FROM daily_prices dp
        LEFT JOIN macro_data md ON CAST(dp.jalali_date AS TEXT) = CAST(md.jalali_date AS TEXT)
        LIMIT 5
        """
        df_test = pd.read_sql_query(test_query, conn)
        print(df_test)
    except Exception as e:
        print(f"❌ Error executing JOIN test: {e}")

    conn.close()

if __name__ == "__main__":
    ultimate_debug()