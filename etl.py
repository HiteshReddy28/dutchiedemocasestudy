"""
ETL script for the Dutchie case study.

This module provides functions to ingest raw Dutchie POS exports (sales and line
items), clean and normalize the data, derive useful temporal fields and
calculations, and persist the result to a local DuckDB database using a
minimal star schema. Views are created for convenience to join fact and
dimension tables on the fly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional, List, Mapping

import pandas as pd
import numpy as np
import duckdb
import pytz
from dateutil import parser

DB_PATH = Path(__file__).resolve().parent / "warehouse" / "dutchie.duckdb"
DB_PATH.parent.mkdir(exist_ok=True, parents=True)


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection to our warehouse."""
    return duckdb.connect(str(DB_PATH))


# -------------------------------
# Helpers / normalization
# -------------------------------

def _hash_staff_id(staff_id: str) -> str:
    """Stable hash for staff identifiers to pseudonymize them."""
    if staff_id is None or (isinstance(staff_id, float) and np.isnan(staff_id)):
        return "unknown"
    return hashlib.sha1(str(staff_id).encode("utf-8")).hexdigest()[:10]


def _stable_location_id(name: str) -> int:
    """Deterministic positive int for a location name (stable across runs)."""
    h = hashlib.sha1((name or "").strip().lower().encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 2_147_483_647   # fits 32-bit int


def standardize_columns(df: pd.DataFrame, type_: str) -> pd.DataFrame:
    """Normalize column names to lower case and rename known fields."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {
        "id": "sale_id" if type_ == "sales" else "line_item_id",
        "saleid": "sale_id",
        "staff": "staff_id",
        "employee_id": "staff_id",
        "employeeid": "staff_id",
        "created_at": "ts",
        "timestamp": "ts",
        "location": "location_name",
        "store": "location_name",
        "order type": "order_type",
        "order_type": "order_type",
        "tax_total": "tax_amount",  # tolerate both spellings
    }
    for src, tgt in rename_map.items():
        if src in df.columns and tgt not in df.columns:
            df = df.rename(columns={src: tgt})
    return df


def normalize_strings(s: Optional[str]) -> Optional[str]:
    """Strip whitespace and title-case a string; return None for empty."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip()
    return s.title() if s else None


def to_store_time(ts: str | pd.Timestamp, location_name: str) -> pd.Timestamp:
    """Convert an ISO timestamp to store-local timezone."""
    tz_lookup = {
        "columbus": "America/New_York",
        "cincinnati": "America/New_York",
    }
    dt = parser.parse(ts) if not isinstance(ts, pd.Timestamp) else ts
    loc_key = (location_name or "").strip().lower()
    tz_name = tz_lookup.get(loc_key, "America/New_York")
    local_tz = pytz.timezone(tz_name)
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(local_tz)


def derive_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar columns (local time, date, hour, day_of_week, daypart)."""
    df = df.copy()
    df["ts_local"] = [to_store_time(t, loc) for t, loc in zip(df.get("ts"), df.get("location_name"))]
    df["date"] = df["ts_local"].dt.date
    df["hour"] = df["ts_local"].dt.hour
    df["day_of_week"] = df["ts_local"].dt.day_name()

    def to_daypart(h: int) -> str:
        h = int(h)
        if 0 <= h <= 11:
            return "open-noon"
        if 12 <= h <= 16:
            return "noon-5"
        return "5-close"

    df["daypart"] = df["hour"].apply(to_daypart)
    return df


def _calendar_key_from_ts_local(ts_local: pd.Series) -> pd.Series:
    """Return integer UTC-minute epoch key (stable, tz-agnostic for joins)."""
    # Ensure tz-aware then convert to UTC; use int64 ns -> minutes
    return (ts_local.dt.tz_convert("UTC").view("int64") // 60_000_000_000)


def compute_discount_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute discount rate as discount_amount / subtotal, clipped to [0, 1]."""
    df = df.copy()
    if "subtotal" not in df.columns or "discount_amount" not in df.columns:
        df["discount_rate"] = 0.0
        return df
    denom = df["subtotal"].replace(0, np.nan)
    df["discount_rate"] = (df["discount_amount"] / denom).clip(lower=0, upper=1.0).fillna(0.0)
    return df


def load_and_transform(file_path: str, type_: str) -> pd.DataFrame:
    """Load a raw export file (CSV or JSON) and apply cleaning and transforms."""
    file_path = str(file_path)
    if file_path.lower().endswith(".json"):
        df = pd.read_json(file_path, convert_dates=False)
    else:
        df = pd.read_csv(file_path)

    # Standardize columns
    df = standardize_columns(df, type_)

    # Normalize string/object columns
    for c in [c for c in df.columns if df[c].dtype == object]:
        df[c] = df[c].apply(normalize_strings)

    if type_ == "sales":
        df = derive_calendar(df)

        for b in ["voided", "refunded"]:
            if b not in df.columns:
                df[b] = False

        if "discount_amount" in df.columns and "subtotal" in df.columns:
            df = compute_discount_rate(df)

        # Refunds reduce net sales
        if "total" in df.columns:
            refunded_mask = df["refunded"].fillna(False)
            df.loc[refunded_mask, "total"] = -df.loc[refunded_mask, "total"].abs()

    # Items need no special transforms here
    return df


# -------------------------------
# Persistence (DuckDB)
# -------------------------------

def persist_tables(conn: duckdb.DuckDBPyConnection, sales_df: Optional[pd.DataFrame], items_df: Optional[pd.DataFrame]) -> None:
    """Create tables if missing and insert data into fact and dimension tables."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS DimLocation (
            location_id INT PRIMARY KEY,
            location_name TEXT,
            tz TEXT
        );
        CREATE TABLE IF NOT EXISTS DimStaff (
            staff_key TEXT PRIMARY KEY,
            staff_id TEXT
        );
        CREATE TABLE IF NOT EXISTS DimProduct (
            product_key TEXT PRIMARY KEY,
            product_id TEXT,
            product_name TEXT,
            category TEXT
        );
        CREATE TABLE IF NOT EXISTS DimCalendar (
            calendar_key BIGINT PRIMARY KEY,
            ts_local TIMESTAMP,
            date DATE,
            hour INT,
            day_of_week TEXT,
            daypart TEXT
        );
        CREATE TABLE IF NOT EXISTS FactSales (
            sale_id TEXT PRIMARY KEY,
            calendar_key BIGINT,
            location_id INT,
            staff_key TEXT,
            order_type TEXT,
            subtotal DOUBLE,
            discount_amount DOUBLE,
            tax_amount DOUBLE,
            total DOUBLE,
            number_of_items INT,
            voided BOOLEAN,
            refunded BOOLEAN,
            discount_rate DOUBLE
        );
        CREATE TABLE IF NOT EXISTS FactLineItems (
            line_item_id TEXT PRIMARY KEY,
            sale_id TEXT,
            product_key TEXT,
            quantity INT,
            unit_price DOUBLE,
            total DOUBLE,
            discount DOUBLE
        );
        """
    )

    def upsert(table: str, df: pd.DataFrame, key_cols: List[str]):
        if df is None or df.empty:
            return
        temp = f"{table}_staging"
        conn.register(temp, df)
        where = " AND ".join([f"t.{c} = s.{c}" for c in key_cols])
        conn.execute(f"DELETE FROM {table} t USING {temp} s WHERE {where};")
        cols = ", ".join(df.columns)
        conn.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM {temp};")
        conn.unregister(temp)

    # --------- Dimensions ----------
    if sales_df is not None and not sales_df.empty:
        # DimLocation (deterministic IDs)
        locs = (
            sales_df[["location_name"]]
            .dropna()
            .drop_duplicates()
            .assign(
                location_id=lambda d: d["location_name"].apply(_stable_location_id),
                tz=lambda d: d["location_name"].str.lower().map({
                    "columbus": "America/New_York",
                    "cincinnati": "America/New_York",
                }).fillna("America/New_York"),
            )
        )
        upsert("DimLocation", locs[["location_id", "location_name", "tz"]], ["location_id"])

        # DimCalendar from sales ts_local; calendar_key is UTC-minute epoch
        cal = sales_df[["ts_local"]].dropna().drop_duplicates().copy()
        cal["calendar_key"] = _calendar_key_from_ts_local(cal["ts_local"])
        cal["date"] = cal["ts_local"].dt.date
        cal["hour"] = cal["ts_local"].dt.hour
        cal["day_of_week"] = cal["ts_local"].dt.day_of_week.replace(
            dict(zip(range(7), ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]))
        )
        # safer: just recompute names
        cal["day_of_week"] = cal["ts_local"].dt.day_name()

        def _to_daypart(h: int) -> str:
            h = int(h)
            if 0 <= h <= 11:
                return "open-noon"
            if 12 <= h <= 16:
                return "noon-5"
            return "5-close"

        cal["daypart"] = cal["hour"].apply(_to_daypart)

        # Store ts_local as naive timestamp (string cast) to avoid tz type issues in DuckDB
        cal_out = cal.copy()
        cal_out["ts_local"] = pd.to_datetime(cal_out["ts_local"].dt.tz_localize(None))
        upsert("DimCalendar", cal_out[["calendar_key", "ts_local", "date", "hour", "day_of_week", "daypart"]], ["calendar_key"])

        # DimStaff
        if "staff_id" in sales_df.columns:
            staff = sales_df[["staff_id"]].dropna().drop_duplicates()
            staff = staff.assign(staff_key=staff["staff_id"].apply(_hash_staff_id))
            upsert("DimStaff", staff[["staff_key", "staff_id"]], ["staff_key"])

    if items_df is not None and not items_df.empty:
        # DimProduct – if product_id exists; otherwise hash (sku|name)
        prods = items_df.copy()
        if "product_id" not in prods.columns:
            prods["product_id"] = (
                prods[["product_name"]]
                .fillna("")
                .astype(str)
                .apply(lambda r: r["product_name"], axis=1)
            )
        prods = prods[["product_id", "product_name", "category"]].drop_duplicates()
        prods["product_key"] = prods.apply(
            lambda r: hashlib.sha1(str(r["product_id"]).encode("utf-8")).hexdigest()[:10],
            axis=1,
        )
        upsert("DimProduct", prods[["product_key", "product_id", "product_name", "category"]], ["product_key"])

    # --------- Facts ----------
    if sales_df is not None and not sales_df.empty:
        # Map location_id & calendar_key on the fly without timestamp merges
        sales = sales_df.copy()
        sales["location_id"] = sales["location_name"].apply(_stable_location_id)
        sales["calendar_key"] = _calendar_key_from_ts_local(sales["ts_local"])
        sales["staff_key"] = sales.get("staff_id", pd.Series(index=sales.index)).apply(_hash_staff_id)

        # Coerce numerics/booleans
        for c in ["subtotal", "discount_amount", "tax_amount", "total", "number_of_items"]:
            if c in sales.columns:
                sales[c] = pd.to_numeric(sales[c], errors="coerce").fillna(0)
        for c in ["voided", "refunded"]:
            if c in sales.columns:
                sales[c] = sales[c].fillna(False).astype(bool)
        if "discount_rate" not in sales.columns:
            # fallback compute if not present
            if "subtotal" in sales.columns and "discount_amount" in sales.columns:
                denom = sales["subtotal"].replace(0, np.nan)
                sales["discount_rate"] = (sales["discount_amount"] / denom).clip(0, 1).fillna(0.0)
            else:
                sales["discount_rate"] = 0.0

        fact_cols = [
            "sale_id", "calendar_key", "location_id", "staff_key", "order_type",
            "subtotal", "discount_amount", "tax_amount", "total",
            "number_of_items", "voided", "refunded", "discount_rate",
        ]
        upsert("FactSales", sales[[c for c in fact_cols if c in sales.columns]], ["sale_id"])

    if items_df is not None and not items_df.empty:
        li = items_df.copy()
        # Attach product_key same way as in DimProduct
        if "product_id" not in li.columns:
            li["product_id"] = (
                li[["product_name"]].fillna("").astype(str).apply(lambda r: r["product_name"], axis=1)
            )
        li["product_key"] = li["product_id"].apply(lambda x: hashlib.sha1(str(x).encode("utf-8")).hexdigest()[:10])

        # Normalize numerics
        if "quantity" in li.columns:
            li["quantity"] = pd.to_numeric(li["quantity"], errors="coerce").fillna(0).astype(int)
        for c in ["unit_price", "total", "discount"]:
            if c in li.columns:
                li[c] = pd.to_numeric(li[c], errors="coerce").fillna(0.0)

        fact_li_cols = ["line_item_id", "sale_id", "product_key", "quantity", "unit_price", "total", "discount"]
        upsert("FactLineItems", li[[c for c in fact_li_cols if c in li.columns]], ["line_item_id"])

    # --------- Views (no joins on timestamps) ----------
    ensure_views(conn)


def ensure_views(conn: duckdb.DuckDBPyConnection) -> None:
    """Create convenience views joining dimensions into sales and item facts."""
    conn.execute(
        """
        CREATE OR REPLACE VIEW v_sales_enriched AS
        SELECT fs.sale_id, fs.calendar_key, fs.location_id, fs.staff_key,
               fs.order_type, fs.subtotal, fs.discount_amount, fs.tax_amount,
               fs.total, fs.number_of_items, fs.voided, fs.refunded,
               fs.discount_rate,
               dl.location_name, dl.tz,
               dc.ts_local, dc.date, dc.hour, dc.day_of_week, dc.daypart,
               ds.staff_id
        FROM FactSales fs
        LEFT JOIN DimLocation dl ON fs.location_id = dl.location_id
        LEFT JOIN DimCalendar dc ON fs.calendar_key = dc.calendar_key
        LEFT JOIN DimStaff ds ON fs.staff_key = ds.staff_key;

        CREATE OR REPLACE VIEW v_items_enriched AS
        SELECT li.line_item_id, li.sale_id, li.product_key, li.quantity,
               li.unit_price, li.total, li.discount,
               fs.location_id, fs.order_type, fs.voided, fs.refunded,
               dl.location_name, dl.tz,
               dc.ts_local, dc.date, dc.hour, dc.daypart,
               dp.product_id, dp.product_name, dp.category
        FROM FactLineItems li
        LEFT JOIN FactSales fs ON li.sale_id = fs.sale_id
        LEFT JOIN DimLocation dl ON fs.location_id = dl.location_id
        LEFT JOIN DimCalendar dc ON fs.calendar_key = dc.calendar_key
        LEFT JOIN DimProduct dp ON li.product_key = dp.product_key;
        """
    )


def upsert_from_files(sales_files: Iterable[str] | None = None, item_files: Iterable[str] | None = None) -> None:
    """Ingest one or more files of sales and line items and persist them."""
    sales_df: Optional[pd.DataFrame] = None
    if sales_files:
        for f in sales_files:
            cur = load_and_transform(f, "sales")
            sales_df = pd.concat([sales_df, cur], ignore_index=True) if sales_df is not None else cur

    items_df: Optional[pd.DataFrame] = None
    if item_files:
        for f in item_files:
            cur = load_and_transform(f, "line_items")
            items_df = pd.concat([items_df, cur], ignore_index=True) if items_df is not None else cur

    with get_connection() as conn:
        persist_tables(conn, sales_df, items_df)


if __name__ == "__main__":
    import sys
    sales_files: List[str] = []
    item_files: List[str] = []
    for arg in sys.argv[1:]:
        if arg.startswith("sales="):
            sales_files = arg.split("=", 1)[1].split(",")
        elif arg.startswith("items=") or arg.startswith("item="):
            item_files = arg.split("=", 1)[1].split(",")
    upsert_from_files(sales_files or None, item_files or None)
