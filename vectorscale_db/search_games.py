# search_games.py
"""
Lightweight similarity search for Steam-games stored in Postgres/TimescaleDB
with vectorscale.  The model and DB engine are initialised once at import time.
"""

# --- imports --------------------------------------------------------------
from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer

from models import Games, engine  # local project modules

# --- global objects -------------------------------------------------------
# model is loaded once and reused (≈1 GB of RAM, but huge speed-up)
MODEL = SentenceTransformer("distiluse-base-multilingual-cased-v2")

# reuse a single prepared statement template (SQLAlchemy will cache the plan)
BASE_STMT = select(Games)  # filters and ordering will be chained later


# --- search function ------------------------------------------------------
def find_games(
    description: str,
    k: int = 5,
    *,
    windows: Optional[bool] = None,
    linux: Optional[bool] = None,
    mac: Optional[bool] = None,
    price: Optional[float] = None,
) -> List[Games]:
    """
    Return up to *k* games most similar to *description*,
    optionally filtered by platform flags and max price.
    """
    # 1) Encode query once – 512-dimensional sentence embedding
    query_emb = MODEL.encode(description).tolist()

    # 2) Build SQL statement dynamically
    stmt = (
        BASE_STMT.order_by(
            Games.game_description_embedding.cosine_distance(query_emb)
        )
        .limit(k)
        # request rows to be prefetched into memory for speed
        .execution_options(prebuffer_rows=True)
    )

    # optional attribute filters
    if price is not None:
        stmt = stmt.filter(Games.price <= price)
    if windows:
        stmt = stmt.filter(Games.windows.is_(True))
    if linux:
        stmt = stmt.filter(Games.linux.is_(True))
    if mac:
        stmt = stmt.filter(Games.mac.is_(True))

    # 3) Execute and return objects
    with Session(engine) as session:
        return session.scalars(stmt).all()


# --- simple CLI test ------------------------------------------------------
if __name__ == "__main__":
    results = find_games(
        "Home decorating",
        k=3,
        mac=True,
        price=5,
    )
    for g in results:
        print(f"{g.name} — ${g.price}")
