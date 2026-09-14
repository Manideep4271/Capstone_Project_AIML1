import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin


BASE_URL = "https://books.toscrape.com/"
CATALOGUE_URL = "https://books.toscrape.com/catalogue/page-{}.html"

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

GBP_TO_INR = 105.50


def scrape_books():
    books = []

    for page_number in range(1, 6):

        url = CATALOGUE_URL.format(page_number)

        response = requests.get(url, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        products = soup.select("article.product_pod")

        for product in products:

            title = product.h3.a.get("title", "").strip()

            price_text = product.select_one(".price_color").get_text(strip=True)

            rating_element = product.select_one("p.star-rating")
            rating_text = ""

            if rating_element:
                classes = rating_element.get("class", [])
                rating_text = next(
                    (value for value in classes if value in RATING_MAP),
                    ""
                )

            availability_element = product.select_one(
                ".availability"
            )

            availability_text = (
                availability_element.get_text(" ", strip=True)
                if availability_element
                else ""
            )

           
            book_url = urljoin(
                BASE_URL,
                product.h3.a.get("href")
            )

            detail_response = requests.get(
                book_url,
                timeout=15
            )
            detail_response.raise_for_status()

            detail_soup = BeautifulSoup(
                detail_response.text,
                "html.parser"
            )

            breadcrumb = detail_soup.select(
                ".breadcrumb li"
            )

            category = ""

            if len(breadcrumb) >= 3:
                category = breadcrumb[-2].get_text(
                    strip=True
                )

            books.append({
                "title": title,
                "price": price_text,
                "star_rating": rating_text,
                "availability": availability_text,
                "category": category
            })

    return pd.DataFrame(books)


def clean_data(df):

    df = df.copy()

   

    df["price_gbp"] = (
        df["price"]
        .astype(str)
        .str.replace("£", "", regex=False)
        .str.strip()
    )

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

   
    if df["price_gbp"].isna().any():
        median_price = df["price_gbp"].median()
        df["price_gbp"] = df["price_gbp"].fillna(
            median_price
        )

  

    df["rating"] = df["star_rating"].map(
        RATING_MAP
    )

  
    if df["rating"].isna().any():
        median_rating = int(
            round(df["rating"].median())
        )

        df["rating"] = df["rating"].fillna(
            median_rating
        )

    df["rating"] = df["rating"].astype(int)

  

    df["in_stock"] = (
        df["availability"]
        .astype(str)
        .str.contains(
            "In stock",
            case=False,
            na=False
        )
    )

  

    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

   
    cleaned_df = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category"
        ]
    ].copy()

    return cleaned_df


if __name__ == "__main__":

    raw_df = scrape_books()

    print(
        f"Scraped rows: {len(raw_df)}"
    )

    cleaned_df = clean_data(raw_df)

    print("\nCleaned Data:")
    print(cleaned_df.head())

    print("\nData types:")
    print(cleaned_df.dtypes)

    print("\nCategories:")
    print(
        cleaned_df["category"]
        .nunique()
    )

    cleaned_df.to_csv(
        "data/books_cleaned.csv",
        index=False
    )

    print(
        "\nSaved: data/books_cleaned.csv"
    )