import os

from scrape_pipeline import (
    scrape_books,
    clean_data
)

from database import (
    load_data
)

from queries import (
    execute_queries
)


def main():

   
    os.makedirs(
        "data",
        exist_ok=True
    )

    os.makedirs(
        "output",
        exist_ok=True
    )

   

    print("=" * 70)
    print("STEP 1: SCRAPING BOOKS")
    print("=" * 70)

    raw_df = scrape_books()

    print(
        f"Raw books scraped: {len(raw_df)}"
    )

   

    print("\n" + "=" * 70)
    print("STEP 2: CLEANING DATA")
    print("=" * 70)

    cleaned_df = clean_data(
        raw_df
    )

    print(
        f"Cleaned books: {len(cleaned_df)}"
    )

    print(
        f"Categories: "
        f"{cleaned_df['category'].nunique()}"
    )

    print("\nSample:")
    print(
        cleaned_df.head().to_string(
            index=False
        )
    )

   
    cleaned_df.to_csv(
        "data/books_cleaned.csv",
        index=False
    )

   

    print("\n" + "=" * 70)
    print("STEP 3: LOADING SQLITE DATABASE")
    print("=" * 70)

    load_data(
        cleaned_df
    )

   

    print("\n" + "=" * 70)
    print("STEP 4: EXECUTING SQL QUERIES")
    print("=" * 70)

    results = execute_queries()

  

    print("\n" + "=" * 70)
    print("STEP 5: PANDAS VALIDATION")
    print("=" * 70)

    demonstrate_pandas_validation(
        cleaned_df
    )

    print("\nPipeline completed successfully.")


def demonstrate_pandas_validation(
    cleaned_df
):

    import sqlite3
    import pandas as pd

    conn = sqlite3.connect(
        "zepto_books.db"
    )

  

    join_query = """
        SELECT
            c.category_id,
            c.category_name,
            b.book_id,
            b.title,
            b.rating,
            b.price_inr,
            b.in_stock
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY
            c.category_name,
            b.rating DESC
    """

    sql_result = pd.read_sql(
        join_query,
        conn
    )

    print(
        "\nJOIN result using pd.read_sql():"
    )

    print(
        sql_result.head(10).to_string(
            index=False
        )
    )

   

    categories_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """,
        conn
    )

    books_df = pd.read_sql(
        """
        SELECT
            book_id,
            title,
            rating,
            price_inr,
            in_stock,
            category_id
        FROM books
        """,
        conn
    )

    merge_result = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    merge_result = merge_result[
        [
            "category_id",
            "category_name",
            "book_id",
            "title",
            "rating",
            "price_inr",
            "in_stock"
        ]
    ]

    merge_result = merge_result.sort_values(
        [
            "category_name",
            "rating"
        ],
        ascending=[
            True,
            False
        ]
    ).reset_index(drop=True)

    sql_result = sql_result.reset_index(
        drop=True
    )

    print(
        "\nJOIN result using pd.merge():"
    )

    print(
        merge_result.head(10).to_string(
            index=False
        )
    )

    # ----------------------------------
    # Compare
    # ----------------------------------

    equivalent = sql_result.equals(
        merge_result
    )

    print(
        "\nDo SQL JOIN and pandas merge "
        f"produce equivalent output? {equivalent}"
    )

  
    comparison = pd.DataFrame({
        "SQL_JOIN": [
            sql_result.head(10).to_dict(
                orient="records"
            )
        ],
        "PANDAS_MERGE": [
            merge_result.head(10).to_dict(
                orient="records"
            )
        ]
    })

    comparison.to_json(
        "output/join_comparison.json",
        orient="records",
        indent=4
    )

    conn.close()


if __name__ == "__main__":
    main()