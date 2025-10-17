# Dutchie Case Study Solution

This repository contains a minimal end‑to‑end solution for the Dutchie AI/Automation Engineer case study.  
It demonstrates how to ingest point‑of‑sale (POS) exports, clean and standardize the data into a simple star schema, and expose high‑level KPIs through a one‑screen dashboard that a general manager can use to run the day.

## Getting Started

The solution uses Python, DuckDB and Streamlit.  It is designed to run locally on a developer’s machine and does **not** require any external services.  If you prefer to work in a virtual environment, create and activate one before installing dependencies.

1. **Clone or download** this repository and navigate into the directory:

   ```bash
   git clone <your‑fork‑or‑zip> && cd dutchie_case_study_solution
   ```

2. **Install dependencies**.  We recommend upgrading pip/setuptools first.  On Windows replace `python` with `py` if needed.

   ```bash
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -r requirements.txt
   ```

3. **Run the ETL** (optional).  You can upsert data into the local DuckDB warehouse from the command line if you prefer.  You must supply at least one sales file and one line_items file.  These may be CSV or JSON:

   ```bash
   python etl.py sales=./sample_data/sales.csv items=./sample_data/line_items.csv
   ```

4. **Launch the dashboard**.  Start Streamlit and open the provided local URL in your browser.  The first render will prompt you to upload your POS exports.  All processing happens client‑side and writes to `./warehouse/dutchie.duckdb`.

   ```bash
   streamlit run app.py
   ```

## Project Structure

```
dutchie_case_study_solution/
├─ app.py                # Streamlit interface (Manager View)
├─ etl.py                # Data ingest, cleaning, star schema logic
├─ requirements.txt      # Libraries required to run the project
├─ README.md             # This file
├─ warehouse/            # DuckDB database (created at runtime)
└─ sample_data/          # Tiny example exports (Sales, Line Items)
```

### `etl.py`

`etl.py` contains functions to load raw exports, standardize fields (names, categories, tenders, order types), convert timestamps to store‑local time, derive useful columns (date, hour, daypart), calculate discount rates, and write the data into a DuckDB file.  A minimal star schema is created with fact tables for sales and line items and dimension tables for products, staff, locations and the calendar.  Views `v_sales_enriched` and `v_items_enriched` join these dimensions for easier analysis.

To upsert new files from the command line run:

```bash
python etl.py sales=path/to/sales.csv items=path/to/line_items.csv
```

If the DuckDB file already exists, data will be appended and de‑duplicated on primary keys.

### `app.py`

`app.py` exposes a one‑screen dashboard built with Streamlit.  When you first load the page, it allows you to upload new exports and calls into `etl.py` to persist them.  Once data is available, you can filter by date range, store location, order type, daypart, category and cashier.  The app calculates the required KPIs:

* **Net Sales & Same‑Store comparison** – total net sales for the current filter window and the same period last week with percentage change.
* **Basket Economics** – average order value (AOV), items per ticket and tender mix chart.
* **Discount / Promo Impact** – discount rate (% of subtotal discounted), AOV for orders with vs. without a promo, and top promotions by total discount.
* **Voids & Refunds Exceptions** – counts and rates of voided/refunded transactions and spike detection (flag if >2× median for a given day).
* **Top/Bottom Movers** – top 10 SKUs by net sales and, when available, margin dollars; category contribution 80/20 chart.
* **Compliance Panel** – quick breakdown of adult‑use vs medical tickets, tax totals (excise/local/state), and a simple issues list for anomalies (negative totals, mismatched tax math, orphan refunds).
* **Heatmap by Hour** – throughput over hours of the day, overlaying counts of voids and discounts to highlight training windows.

There is also a “Notes for Today” text area at the bottom that saves your notes to a local file (`notes.txt`) so that they persist between sessions.

## Assumptions

* The POS exports follow the Dutchie schema used in the sample data.  If your field names differ, update `standardize_columns()` in `etl.py`.
* Columbus and Cincinnati stores reside in the `America/New_York` timezone.  Modify the `tz_lookup` dictionary in `etl.py` if you operate in other regions.
* A transaction with `voided=true` is excluded from net sales but still counted in exception rates.  Refunds flip the sign of the sale to reduce net sales.  Discount rate is computed as `discount_amount / subtotal` and clipped between 0 and 1.
* Staff identifiers are hashed to preserve anonymity.  No customer PII is loaded.

## Next Steps

Given more time, we would add automated tests around the ETL logic, integrate with the real Dutchie POS API instead of file uploads, and deploy the dashboard behind authentication.  We would also refine the exception logic to learn from historical spikes and tune thresholds.