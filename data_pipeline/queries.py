import sqlite3
import pandas as pd


DATABASE_NAME = "zepto_books.db"


queries = {

    "Q1_SELECT_WHERE": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating >= 4
    """,

    "Q2_ORDER_BY_LIMIT": """
        SELECT title, price_inr, rating
        FROM books
        ORDER BY price_inr DESC
        LIMIT 10
    """,

    "Q3_DISTINCT": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name
    """,

    "Q4_BETWEEN": """
        SELECT title, price_gbp, price_inr
        FROM books
        WHERE price_gbp BETWEEN 10 AND 30
        ORDER BY price_gbp
    """,

    "Q5_IN": """
        SELECT title, rating, in_stock
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC
    """,

    "Q6_JOIN": """
        SELECT
            c.category_name,
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
}


def execute_queries():

    conn = sqlite3.connect(
        DATABASE_NAME
    )

    all_results = {}

    for name, query in queries.items():

        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

        print(query.strip())

        df = pd.read_sql(
            query,
            conn
        )

        all_results[name] = df

        print("\nOutput:")
        print(df.to_string(index=False))

    conn.close()

    return all_results


if __name__ == "__main__":

    results = execute_queries()

    with open(
        "output/query_outputs.txt",
        "w",
        encoding="utf-8"
    ) as file:

        for name, df in results.items():

            file.write(
                "\n" + "=" * 70 + "\n"
            )

            file.write(
                name + "\n"
            )

            file.write(
                "=" * 70 + "\n"
            )

            file.write(
                df.to_string(index=False)
            )

            file.write("\n")

    print(
        "\nSaved query outputs to "
        "output/query_outputs.txt"
    )