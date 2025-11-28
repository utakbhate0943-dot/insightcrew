from dotenv import load_dotenv
import os, urllib, sqlalchemy
from sqlalchemy import text

def main():
    load_dotenv()
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_DATABASE")
    username = os.getenv("DB_USERNAME")
    password = os.getenv("DB_PASSWORD")
    driver = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")

    if not (server and database and username and password):
        print("Missing one of DB_SERVER/DB_DATABASE/DB_USERNAME/DB_PASSWORD")
        return

    conn_str = (
        f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};"
        f"UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=10;"
    )
    print("Trying connection string (masked):", conn_str.replace(password, '***'))
    quoted = urllib.parse.quote_plus(conn_str)
    engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={quoted}")
    try:
        with engine.connect() as conn:
            r = conn.execute(text("SELECT 1 as ok")).fetchone()
            print('Connection OK:', r)
    except Exception as e:
        print("Connection failed:", repr(e))

if __name__ == '__main__':
    main()