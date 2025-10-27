import duckdb

def test_database():
    con = duckdb.connect('warehouse/dutchie.duckdb')
    tables = con.execute('SHOW TABLES').fetchall()
    print("Tables in database:", tables)

    # Test if views exist
    views = con.execute("SELECT table_name FROM information_schema.tables WHERE table_type = 'VIEW'").fetchall()
    print("Views in database:", views)

    # Test sample query on FactSales
    try:
        sales_count = con.execute('SELECT COUNT(*) FROM FactSales').fetchone()[0]
        print(f"Number of sales records: {sales_count}")
    except Exception as e:
        print(f"Error querying FactSales: {e}")

    # Test sample query on FactLineItems
    try:
        items_count = con.execute('SELECT COUNT(*) FROM FactLineItems').fetchone()[0]
        print(f"Number of line_items records: {items_count}")
    except Exception as e:
        print(f"Error querying FactLineItems: {items_count}")

    # Test enriched views
    try:
        enriched_sales = con.execute('SELECT COUNT(*) FROM v_sales_enriched').fetchone()[0]
        print(f"Number of enriched sales records: {enriched_sales}")
    except Exception as e:
        print(f"Error querying v_sales_enriched: {e}")

    try:
        enriched_items = con.execute('SELECT COUNT(*) FROM v_items_enriched').fetchone()[0]
        print(f"Number of enriched items records: {enriched_items}")
    except Exception as e:
        print(f"Error querying v_items_enriched: {e}")

    con.close()

if __name__ == "__main__":
    test_database()
