# Module 1 — Data Pipeline

## Objective

This module implements an end-to-end data pipeline using data from `books.toscrape.com`.

The pipeline performs:

1. Web scraping
2. Data cleaning
3. Currency conversion
4. SQLite database creation
5. SQL analysis
6. Pandas validation

## Data Source

The project uses `books.toscrape.com`, a public website designed for scraping practice.

The pipeline uses Python `requests` and `BeautifulSoup` to collect book information.

The first five catalogue pages are scraped, providing at least 60 book records.

## Fields Collected

The raw fields are:

* `title`
* `price`
* `star_rating`
* `availability`
* `category`

## Data Cleaning

### Price

The `£` currency symbol is removed and the value is converted to a floating-point number.

The cleaned column is:

```text
price_gbp
```

### Rating

The textual ratings are converted using:

```text
One   → 1
Two   → 2
Three → 3
Four  → 4
Five  → 5
```

The resulting column is:

```text
rating
```

### Availability

The availability text is converted into a Boolean:

```text
In stock → True
Otherwise → False
```

The resulting column is:

```text
in_stock
```

### Missing values

Numeric parsing failures are converted to missing values using `errors="coerce"`.

For numeric fields, missing values are replaced using the median of the available values, as required by the assignment.

The pipeline therefore does not crash when an individual numeric value cannot be parsed.

## Currency Conversion

The assignment specifies the fixed baseline:

```text
1 GBP = 105.50 INR
```

Therefore:

```text
price_inr = price_gbp × 105.50
```

This is a fixed project-defined conversion rate and does not use an external currency API.

## Database Design

A normalized SQLite database named `zepto_books.db` is created.

### categories

```text
category_id     INTEGER PRIMARY KEY
category_name   TEXT UNIQUE
```

### books

```text
book_id         INTEGER PRIMARY KEY
title           TEXT
price_gbp       REAL
price_inr       REAL
rating          INTEGER
in_stock        INTEGER
category_id     INTEGER FOREIGN KEY
```

The relationship is:

```text
categories
     |
     | category_id
     |
     ↓
books
```

This avoids storing the category name repeatedly for every book.

## SQL Queries

The pipeline executes six SQL queries.

### Query 1 — SELECT and WHERE

Returns books with a rating of at least 4.

### Query 2 — ORDER BY and LIMIT

Returns the ten most expensive books by INR price.

### Query 3 — DISTINCT

Returns the unique categories.

### Query 4 — BETWEEN

Returns books whose GBP price is between £10 and £30.

### Query 5 — IN

Returns books having a rating of 4 or 5.

### Query 6 — JOIN

Joins the `books` and `categories` tables using `category_id`.

## Pandas Validation

The JOIN result is first generated using:

```python
pd.read_sql()
```

The same relationship is then reproduced in memory using:

```python
pd.merge()
```

The two results are sorted into the same order and compared using:

```python
sql_result.equals(merge_result)
```

The expected result is:

```text
Do SQL JOIN and pandas merge produce equivalent output? True
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python run_pipeline.py
```

The pipeline will:

```text
Scrape
  ↓
Clean
  ↓
Convert GBP → INR
  ↓
Save CSV
  ↓
Create SQLite database
  ↓
Execute SQL queries
  ↓
Validate JOIN with pandas
```

## Output Files

The pipeline generates:

```text
data/books_cleaned.csv
zepto_books.db
output/query_outputs.txt
output/join_comparison.json
```

All generated data can be recreated by running `run_pipeline.py`.
