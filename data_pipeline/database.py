import sqlite3
import pandas as pd


DATABASE_NAME = "zepto_books.db"


def create_database():

    conn = sqlite3.connect(DATABASE_NAME)

    cursor = conn.cursor()

   
    cursor.execute(
        "PRAGMA foreign_keys = ON"
    )

  

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
    """)

 

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    conn.commit()

    return conn


def load_data(cleaned_df):

    conn = create_database()

    cursor = conn.cursor()

  
    cursor.execute(
        "DELETE FROM books"
    )

    cursor.execute(
        "DELETE FROM categories"
    )

    conn.commit()

  

    categories = (
        cleaned_df["category"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    for category in categories:

        cursor.execute(
            """
            INSERT OR IGNORE INTO categories
            (category_name)
            VALUES (?)
            """,
            (category,)
        )

    conn.commit()

  

    for _, row in cleaned_df.iterrows():

        cursor.execute(
            """
            SELECT category_id
            FROM categories
            WHERE category_name = ?
            """,
            (row["category"],)
        )

        result = cursor.fetchone()

        category_id = result[0]

        cursor.execute(
            """
            INSERT INTO books
            (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(row["in_stock"]),
                category_id
            )
        )

    conn.commit()

    print(
        "Database successfully populated."
    )

    return conn


if __name__ == "__main__":

    df = pd.read_csv(
        "data/books_cleaned.csv"
    )

    load_data(df)