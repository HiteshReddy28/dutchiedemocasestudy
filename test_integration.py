import pandas as pd
from app import load_data, filter_sales, compute_kpis

def test_integration():
    """Test the integration between ETL and app components."""
    print("Testing integration between ETL and app...")

    # Load data using app function
    df_sales, df_items = load_data()
    print(f"Loaded data: {len(df_sales)} sales, {len(df_items)} items")

    if df_sales.empty:
        print("No data loaded. Running ETL first...")
        from etl import upsert_from_files
        upsert_from_files(['sample_data/sales.csv'], ['sample_data/line_items.csv'])
        df_sales, df_items = load_data()
        print(f"After ETL: {len(df_sales)} sales, {len(df_items)} items")

    # Test filtering
    print("\nTesting filtering...")
    filtered_sales, filtered_items = filter_sales(
        sales_df=df_sales,
        items_df=df_items,
        date_range=None,
        locations=['Columbus'],
        order_types=['Pickup'],
        dayparts=[],
        categories=[],
        staff=[]
    )
    print(f"Filtered data: {len(filtered_sales)} sales, {len(filtered_items)} items")

    # Test KPI computation
    print("\nTesting KPI computation...")
    kpis = compute_kpis(
        sales_df=filtered_sales,
        items_df=filtered_items,
        base_sales_df=df_sales,
        date_range=None
    )
    print("KPIs computed:")
    for key, value in kpis.items():
        print(f"  {key}: {value}")

    print("Integration test completed successfully!")

if __name__ == "__main__":
    test_integration()
