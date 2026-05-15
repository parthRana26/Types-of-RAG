from sqlalchemy import create_engine

engine = create_engine("sqlite:///company.db")

with engine.begin() as conn:
    conn.exec_driver_sql("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        department TEXT,
        salary INTEGER
    )
    """)

    conn.exec_driver_sql("""
    DELETE FROM employees
    """)

    conn.exec_driver_sql("""
    INSERT INTO employees (name, department, salary) VALUES
    ('Alice', 'AI', 100000),
    ('Bob', 'Backend', 80000),
    ('Charlie', 'AI', 120000)
    """)