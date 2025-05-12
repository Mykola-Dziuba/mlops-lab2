# connect.py

from sqlalchemy.engine import URL
from sqlalchemy import create_engine

# Define database connection URL
db_url = URL.create(
    drivername="postgresql+psycopg2",
    username="postgres",
    password="password",
    host="localhost",
    port=5555,
    database="similarity_search_service_db"
)

# Create SQLAlchemy engine
engine = create_engine(db_url)

print("Connection successful")
