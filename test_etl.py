import pandas as pd
from etl import load_and_transform, upsert_from_files, get_connection

def test_etl():
    """Test the ETL process with sample data."""
    print("Testing ETL with sample data...")

    # Test loading sales data
    sales_df = load_and_transform('sample_data/sales.csv', 'sales')
    print(f"Sales data loaded: {len(sales_df)} rows")
    print("Sales columns:", list(sales_df.columns))
    print("Sample sales data:")
    print(sales_df.head())

    # Test loading line items data
    items_df = load_and_transform('sample_data/line_items.csv', 'line_items')
    print(f"\nLine items data loaded: {len(items_df)} rows")
    print("Line items columns:", list(items_df.columns))
    print("Sample line items data:")
    print(items_df.head())

    # Test upserting data
    print("\nUpserting data into database...")
    upsert_from_files(['sample_data/sales.csv'], ['sample_data/line_items.csv'])

    # Verify data in database
    with get_connection() as conn:
        sales_count = conn.execute('SELECT COUNT(*) FROM FactSales').fetchone()[0]
        items_count = conn.execute('SELECT COUNT(*) FROM FactLineItems').fetchone()[0]
        print(f"Data upserted: {sales_count} sales, {items_count} line items")

        # Test enriched views
        enriched_sales = conn.execute('SELECT COUNT(*) FROM v_sales_enriched').fetchone()[0]
        enriched_items = conn.execute('SELECT COUNT(*) FROM v_items_enriched').fetchone()[0]
        print(f"Enriched views: {enriched_sales} sales, {enriched_items} items")

    print("ETL test completed successfully!")

if __name__ == "__main__":
    test_etl()
