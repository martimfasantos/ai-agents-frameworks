import sqlite3

from smolagents import CodeAgent, OpenAIModel, tool

from settings import settings

"""
-------------------------------------------------------
In this example, we explore smolagents for text-to-SQL:

- Creating an in-memory SQLite database with sample data
- Defining a tool that executes SQL queries
- Agent translating natural language to SQL
- CodeAgent advantage: writing and executing SQL in Python

Text-to-SQL is a natural fit for CodeAgent since it can
write Python code that constructs and runs SQL queries,
handling complex joins and aggregations natively.

For more details, visit:
https://huggingface.co/docs/smolagents/examples/text_to_sql
-------------------------------------------------------
"""

# --- 1. Create in-memory SQLite database with sample data ---
db = sqlite3.connect(":memory:", check_same_thread=False)
cursor = db.cursor()

cursor.executescript("""
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        salary REAL NOT NULL,
        hire_date TEXT NOT NULL
    );

    INSERT INTO employees VALUES (1, 'Alice Johnson', 'Engineering', 120000, '2020-03-15');
    INSERT INTO employees VALUES (2, 'Bob Smith', 'Marketing', 85000, '2019-07-22');
    INSERT INTO employees VALUES (3, 'Carol White', 'Engineering', 135000, '2018-01-10');
    INSERT INTO employees VALUES (4, 'David Brown', 'Sales', 92000, '2021-06-01');
    INSERT INTO employees VALUES (5, 'Eve Davis', 'Engineering', 128000, '2022-09-12');
    INSERT INTO employees VALUES (6, 'Frank Wilson', 'Marketing', 78000, '2020-11-30');
    INSERT INTO employees VALUES (7, 'Grace Lee', 'Sales', 98000, '2019-04-18');
    INSERT INTO employees VALUES (8, 'Henry Taylor', 'Engineering', 142000, '2017-08-25');
""")
db.commit()


# --- 2. Define the SQL execution tool ---
@tool
def run_sql_query(query: str) -> str:
    """Execute a SQL query against the employees database and return results.

    The database has an 'employees' table with columns:
    id (INTEGER), name (TEXT), department (TEXT), salary (REAL), hire_date (TEXT).

    Args:
        query: A valid SQL query string.

    Returns:
        The query results as a formatted string.
    """
    try:
        cursor = db.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        if not rows:
            return "Query returned no results."
        columns = [desc[0] for desc in cursor.description]
        result_lines = [" | ".join(columns)]
        result_lines.append("-" * len(result_lines[0]))
        for row in rows:
            result_lines.append(" | ".join(str(val) for val in row))
        return "\n".join(result_lines)
    except Exception as e:
        return f"SQL Error: {e}"


@tool
def get_table_schema(table_name: str) -> str:
    """Get the schema of a table in the database.

    Args:
        table_name: The name of the table to inspect.

    Returns:
        The CREATE TABLE statement for the table.
    """
    cursor = db.cursor()
    cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    result = cursor.fetchone()
    if result:
        return result[0]
    return f"Table '{table_name}' not found."


# --- 3. Create the model and agent ---
model = OpenAIModel(
    model_id=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)

agent = CodeAgent(
    tools=[run_sql_query, get_table_schema],
    model=model,
    max_steps=5,
)

# --- 4. Run natural language queries ---
print("=== Text-to-SQL Demo ===\n")

queries = [
    "What is the average salary by department? Show all departments.",
    "Who are the top 3 highest-paid employees? Show their names and salaries.",
]

for i, query in enumerate(queries, 1):
    print(f"--- Query {i}: {query} ---")
    result = agent.run(f"{query} Reply concisely with the data.")
    print(f"Result: {result}\n")

# --- 5. Clean up ---
db.close()
