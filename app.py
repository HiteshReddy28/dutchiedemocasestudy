"""
Streamlit dashboard for the Dutchie case study.

This application provides a one-screen manager view over POS data ingested
through `etl.py`. Users can upload new sales and line item exports,
configure filters, and explore KPIs such as net sales, basket economics,
discount impact, exception rates, movers, compliance metrics, and a
heatmap of activity by hour. A simple notes pad persists text across
sessions.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import datetime as dt
from typing import List, Tuple

import streamlit as st
import pandas as pd
import altair as alt

# Import DB path so we can reliably check for it
from etl import upsert_from_files, get_connection, DB_PATH


# -------------------------------
# Data loading / persistence
# -------------------------------

@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch enriched sales and items data from DuckDB.

    Returns two DataFrames: sales and items. If the database file
    does not exist or no data has been loaded, empty DataFrames are
    returned.
    """
    if not Path(DB_PATH).exists():
        return pd.DataFrame(), pd.DataFrame()
    with get_connection() as conn:
        try:
            df_sales = conn.execute("SELECT * FROM v_sales_enriched").df()
        except Exception:
            df_sales = pd.DataFrame()
        try:
            df_items = conn.execute("SELECT * FROM v_items_enriched").df()
        except Exception:
            df_items = pd.DataFrame()
    return df_sales, df_items


def save_uploaded_files(uploaded_files: List, prefix: str) -> List[str]:
    """Persist uploaded Streamlit files to temporary paths and return paths."""
    paths: List[str] = []
    for uf in uploaded_files or []:
        suffix = Path(uf.name).suffix
        fd = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=prefix)
        fd.write(uf.getvalue())
        fd.flush()
        paths.append(fd.name)
    return paths


# -------------------------------
# Filtering & KPIs
# -------------------------------

def _normalize_items_count(df_items: pd.DataFrame) -> int:
    """Return total items count from items_df, supporting either `quantity` or `qty`."""
    if df_items.empty:
        return 0
    if "quantity" in df_items.columns:
        return pd.to_numeric(df_items["quantity"], errors="coerce").fillna(0).astype(int).sum()
    if "qty" in df_items.columns:
        return pd.to_numeric(df_items["qty"], errors="coerce").fillna(0).astype(int).sum()
    return 0


def filter_sales(
    sales_df: pd.DataFrame,
    items_df: pd.DataFrame,
    date_range: Tuple[dt.date, dt.date] | None,
    locations: List[str],
    order_types: List[str],
    dayparts: List[str],
    categories: List[str],
    staff: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply user-selected filters to the sales DataFrame, and return
    (filtered_sales, filtered_items).

    Category filtering is done by mapping selected categories to sale_ids
    via the items table; then we keep only those sales and their items.
    """
    if sales_df.empty:
        return sales_df, items_df

    mask = pd.Series(True, index=sales_df.index)

    # Date range
    if date_range:
        start, end = date_range
        mask &= (sales_df["date"] >= pd.to_datetime(start)) & (sales_df["date"] <= pd.to_datetime(end))

    # Location
    if locations:
        mask &= sales_df["location_name"].isin(locations)

    # Order type
    if order_types:
        mask &= sales_df["order_type"].isin(order_types)

    # Daypart
    if dayparts:
        mask &= sales_df["daypart"].isin(dayparts)

    # Cashier/staff
    if staff:
        mask &= sales_df["staff_id"].isin(staff)

    filtered_sales = sales_df.loc[mask].copy()

    # Category filter (via items)
    if categories and not items_df.empty:
        sale_ids_with_cat = items_df[items_df["category"].isin(categories)]["sale_id"].unique().tolist()
        filtered_sales = filtered_sales[filtered_sales["sale_id"].isin(sale_ids_with_cat)]

    # Filter items to the resulting sale_ids (and categories, if selected)
    if not items_df.empty:
        filtered_items = items_df[items_df["sale_id"].isin(filtered_sales["sale_id"])]
        if categories:
            filtered_items = filtered_items[filtered_items["category"].isin(categories)]
    else:
        filtered_items = pd.DataFrame()

    return filtered_sales, filtered_items


def compute_kpis(
    sales_df: pd.DataFrame,
    items_df: pd.DataFrame,
    base_sales_df: pd.DataFrame,
    date_range: Tuple[dt.date, dt.date] | None,
) -> dict:
    """Compute KPI values for the filtered sales DataFrame.

    base_sales_df is the full sales set used for same-store comparison.
    date_range is the filter range; we compare to the same range last week.
    """
    kpis: dict = {}
    if sales_df.empty:
        return kpis

    # Net Sales (exclude voided)
    net_sales = sales_df.loc[~sales_df["voided"], "total"].sum()

    # Tickets
    tickets = sales_df["sale_id"].nunique()

    # Items per ticket
    items_count = _normalize_items_count(items_df)
    if items_count == 0 and "number_of_items" in sales_df.columns:
        items_count = pd.to_numeric(sales_df.loc[~sales_df["voided"], "number_of_items"], errors="coerce").fillna(0).sum()

    aov = (net_sales / tickets) if tickets else 0.0
    items_per_ticket = (items_count / tickets) if tickets else 0.0

    # Discount rate
    subtotal_sum = sales_df["subtotal"].sum() if "subtotal" in sales_df.columns else 0.0
    discount_sum = sales_df["discount_amount"].sum() if "discount_amount" in sales_df.columns else 0.0
    discount_rate = (discount_sum / subtotal_sum) if subtotal_sum else 0.0

    # Promo vs no-promo proxy: use discount_rate per ticket if present, else discount_amount>0
    if "discount_rate" in sales_df.columns:
        with_promo = sales_df[sales_df["discount_rate"] > 0]
        no_promo = sales_df[sales_df["discount_rate"] == 0]
    else:
        if "discount_amount" in sales_df.columns:
            with_promo = sales_df[sales_df["discount_amount"] > 0]
            no_promo = sales_df[sales_df["discount_amount"] <= 0]
        else:
            with_promo = sales_df.iloc[0:0]
            no_promo = sales_df

    aov_promo = (with_promo.loc[~with_promo["voided"], "total"].sum() / with_promo["sale_id"].nunique()) if len(with_promo) else 0.0
    aov_no_promo = (no_promo.loc[~no_promo["voided"], "total"].sum() / no_promo["sale_id"].nunique()) if len(no_promo) else 0.0

    # Same-store comparison (last week)
    if date_range:
        start, end = date_range
        start_prev = pd.to_datetime(start) - pd.Timedelta(days=7)
        end_prev = pd.to_datetime(end) - pd.Timedelta(days=7)

        prev_df = base_sales_df[
            (base_sales_df["date"] >= start_prev)
            & (base_sales_df["date"] <= end_prev)
            & (base_sales_df["location_name"].isin(sales_df["location_name"].unique()))
            & (base_sales_df["order_type"].isin(sales_df["order_type"].unique()))
            & (base_sales_df["daypart"].isin(sales_df["daypart"].unique()))
        ]
        net_prev = prev_df.loc[~prev_df["voided"], "total"].sum()
    else:
        net_prev = 0.0

    pct_change = ((net_sales - net_prev) / net_prev * 100.0) if net_prev else None

    # Exception rates
    void_rate = sales_df["voided"].mean() if len(sales_df) else 0.0
    refund_rate = sales_df["refunded"].mean() if len(sales_df) else 0.0

    kpis.update(
        dict(
            net_sales=net_sales,
            net_prev=net_prev,
            pct_change=pct_change,
            tickets=tickets,
            aov=aov,
            items_per_ticket=items_per_ticket,
            discount_rate=discount_rate,
            aov_promo=aov_promo,
            aov_no_promo=aov_no_promo,
            void_rate=void_rate,
            refund_rate=refund_rate,
        )
    )
    return kpis


# -------------------------------
# App
# -------------------------------

def render_dashboard():
    st.set_page_config(page_title="GM Manager View", layout="wide")
    st.title("General Manager Dashboard")
    st.markdown(
        "This dashboard lets you explore POS transactions, spot issues and opportunities, and record daily notes."
    )

    # --- File upload ---
    with st.expander("Upload POS exports (Sales & Line Items)"):
        sales_files = st.file_uploader("Sales files", type=["csv", "json"], accept_multiple_files=True)
        items_files = st.file_uploader("Line Items files", type=["csv", "json"], accept_multiple_files=True)
        if st.button("Process files"):
            sales_paths = save_uploaded_files(sales_files, prefix="sales_")
            items_paths = save_uploaded_files(items_files, prefix="items_")
            if not sales_paths and not items_paths:
                st.warning("Please select at least one file to upload.")
            else:
                with st.spinner("Processing..."):
                    try:
                        upsert_from_files(sales_paths or None, items_paths or None)
                        st.success("Files processed successfully!")
                        st.experimental_rerun()  # Reload to show new data
                    except Exception as e:
                        st.error(f"Error processing files: {e}")
                    finally:
                        # Clean up temp files
                        for p in sales_paths + items_paths:
                            try:
                                Path(p).unlink()
                            except Exception:
                                pass
                    # upsert_from_files handles tz/calendar_id inside ETL
                    upsert_from_files(sales_paths or None, items_paths or None)
                st.success("Files processed and loaded into the warehouse.")
                st.cache_data.clear()  # ensure fresh read after ingest

    # --- Load data ---
    df_sales, df_items = load_data()
    if df_sales.empty:
        st.info("No data loaded yet. Upload your exports to get started.")
        return

    # Ensure date & hour are correct types
    df_sales["date"] = pd.to_datetime(df_sales["date"], errors="coerce")
    if "hour" in df_sales.columns:
        df_sales["hour"] = pd.to_numeric(df_sales["hour"], errors="coerce").fillna(0).astype(int)

    # --- Filters ---
    # Guard against all-NaT edge cases
    if df_sales["date"].notna().any():
        min_date, max_date = df_sales["date"].min(), df_sales["date"].max()
        default_start = max(min_date, max_date - pd.Timedelta(days=6))  # clamp to min_date
        default_end = max_date
        min_value = min_date
        max_value = max_date
        default_val = (default_start.date(), default_end.date())
    else:
        today = dt.date.today()
        min_value = max_value = default_val = today

    date_range = st.date_input(
        "Date range",
        value=default_val,
        min_value=min_value,
        max_value=max_value,
    )
    if isinstance(date_range, dt.date):
        date_range = (date_range, date_range)

    location_opts = sorted(df_sales["location_name"].dropna().unique().tolist())
    sel_locations = st.multiselect("Locations", options=location_opts, default=location_opts)

    order_opts = sorted(df_sales["order_type"].dropna().unique().tolist())
    sel_order_types = st.multiselect("Order Types", options=order_opts, default=order_opts)

    daypart_opts = sorted(df_sales["daypart"].dropna().unique().tolist())
    sel_dayparts = st.multiselect("Dayparts", options=daypart_opts, default=daypart_opts)

    cat_opts = sorted(df_items["category"].dropna().unique().tolist()) if not df_items.empty else []
    sel_cats = st.multiselect("Categories", options=cat_opts)

    staff_opts = sorted(df_sales["staff_id"].dropna().unique().tolist())
    sel_staff = st.multiselect("Cashiers", options=staff_opts)

    # --- Apply filters ---
    filtered_sales, filtered_items = filter_sales(
        sales_df=df_sales,
        items_df=df_items,
        date_range=date_range if date_range else None,
        locations=sel_locations,
        order_types=sel_order_types,
        dayparts=sel_dayparts,
        categories=sel_cats,
        staff=sel_staff,
    )

    kpis = compute_kpis(
        sales_df=filtered_sales,
        items_df=filtered_items,
        base_sales_df=df_sales,
        date_range=date_range if date_range else None,
    )

    # --- KPI cards ---
    if kpis:
        col1, col2, col3, col4 = st.columns(4)
        net = kpis["net_sales"]
        prev = kpis["net_prev"]
        pct = kpis["pct_change"]
        col1.metric("Net Sales", f"${net:,.2f}", f"{pct:+.1f}% vs LW" if pct is not None else "–")
        col2.metric("AOV", f"${kpis['aov']:,.2f}")
        col3.metric("Items / Ticket", f"{kpis['items_per_ticket']:.2f}")
        col4.metric("Discount Rate", f"{kpis['discount_rate']*100:.1f}%")

        col5, col6, col7 = st.columns(3)
        col5.metric("AOV w/ Promo", f"${kpis['aov_promo']:,.2f}")
        col6.metric("AOV w/o Promo", f"${kpis['aov_no_promo']:,.2f}")
        col7.metric("Void Rate", f"{kpis['void_rate']*100:.1f}%")

        col8, col9 = st.columns(2)
        col8.metric("Refund Rate", f"{kpis['refund_rate']*100:.1f}%")
        col9.metric("Tickets", f"{kpis['tickets']}")

    # --- Charts & Tables ---
    if not filtered_sales.empty:
        # “Tender mix” proxy: distribution of order types
        tender_mix = (
            filtered_sales.groupby("order_type")["sale_id"]
            .nunique()
            .reset_index(name="count")
        )
        tender_chart = (
            alt.Chart(tender_mix)
            .mark_bar()
            .encode(
                x=alt.X("order_type", title="Order Type"),
                y=alt.Y("count", title="Tickets"),
                tooltip=["order_type", "count"],
            )
            .properties(title="Tender Mix (Order Type)")
        )

        # AOV: Promo vs No Promo (from KPIs)
        disc = pd.DataFrame(
            {"Promo": ["With Promo", "Without Promo"], "AOV": [kpis.get("aov_promo", 0), kpis.get("aov_no_promo", 0)]}
        )
        disc_chart = (
            alt.Chart(disc)
            .mark_bar()
            .encode(
                x=alt.X("Promo", title=""),
                y=alt.Y("AOV", title="AOV ($)"),
                tooltip=["Promo", alt.Tooltip("AOV", format="$.2f")],
            )
            .properties(title="AOV: Promo vs No Promo")
        )

        # Exceptions by day (void/refund rates, spike flags)
        exc = (
            filtered_sales.groupby(["date"])
            .agg(tickets=("sale_id", "nunique"), voids=("voided", "sum"), refunds=("refunded", "sum"))
            .reset_index()
        )
        exc["void_rate"] = exc["voids"] / exc["tickets"]
        exc["refund_rate"] = exc["refunds"] / exc["tickets"]
        void_median = exc["void_rate"].median() if len(exc) else 0
        refund_median = exc["refund_rate"].median() if len(exc) else 0
        exc["void_spike"] = exc["void_rate"] > 2 * void_median
        exc["refund_spike"] = exc["refund_rate"] > 2 * refund_median

        # Top movers (exclude voided tickets)
        if not filtered_items.empty:
            merged = filtered_items.merge(
                filtered_sales[["sale_id", "voided"]], on="sale_id", how="inner"
            )
            merged = merged.loc[~merged.get("voided", pd.Series(False, index=merged.index))]
            # Pick best available value/qty columns
            value_col = "total" if "total" in merged.columns else ("line_total" if "line_total" in merged.columns else None)
            qty_col = "quantity" if "quantity" in merged.columns else ("qty" if "qty" in merged.columns else None)

            if value_col and qty_col:
                movers = (
                    merged.groupby(["product_name", "category"])
                    .agg(net_sales=(value_col, "sum"), quantity=(qty_col, "sum"))
                    .reset_index()
                )
                top10 = movers.sort_values("net_sales", ascending=False).head(10)
            else:
                top10 = pd.DataFrame()
        else:
            top10 = pd.DataFrame()

        # Heat “by hour”: tickets + voids + discounts
        hm = filtered_sales.copy()
        if "discount_rate" in hm.columns:
            hm["discount_flag"] = hm["discount_rate"] > 0
        else:
            hm["discount_flag"] = (hm["discount_amount"] > 0) if "discount_amount" in hm.columns else False
        by_hour = (
            hm.groupby("hour")
            .agg(
                tickets=("sale_id", "nunique"),
                voids=("voided", "sum"),
                discounts=("discount_flag", "sum"),
            )
            .reset_index()
        )
        melt = by_hour.melt(
            id_vars="hour", value_vars=["tickets", "voids", "discounts"], var_name="metric", value_name="count"
        )
        heat_chart = (
            alt.Chart(melt)
            .mark_bar()
            .encode(
                x=alt.X("hour:O", title="Hour of Day"),
                y=alt.Y("count:Q", title="Count"),
                color=alt.Color("metric", title="Metric"),
                tooltip=["hour", "metric", "count"],
            )
            .properties(title="Hourly Throughput & Exceptions")
        )

        # Layout
        st.altair_chart(tender_chart, use_container_width=True)
        col_disc, col_heat = st.columns(2)
        col_disc.altair_chart(disc_chart, use_container_width=True)
        col_heat.altair_chart(heat_chart, use_container_width=True)

        st.subheader("Voids & Refunds Exceptions")
        st.dataframe(exc[["date", "tickets", "voids", "refunds", "void_rate", "refund_rate", "void_spike", "refund_spike"]])

        if not top10.empty:
            st.subheader("Top 10 Movers by Net Sales")
            st.dataframe(top10)

    # --- Notes for Today ---
    st.markdown("### Notes for Today")
    notes_file = Path(__file__).resolve().parent / "notes.txt"

    # Safely read existing notes
    try:
        existing_notes = notes_file.read_text()
    except Exception:
        existing_notes = ""

    # Display text area with proper label handling
    notes = st.text_area(
        label="Daily Notes",        # Required label (visible or hidden)
        value=existing_notes,
        height=100,
        label_visibility="collapsed"  # hide the label visually
    )

    # Save button
    if st.button("Save Notes"):
        try:
            notes_file.write_text(notes or "")
            st.success("Notes saved successfully!")
        except Exception as e:
            st.error(f"Failed to save notes: {e}")



if __name__ == "__main__":
    render_dashboard()
