# models.py

# --- imports ---------------------------------------------------------------
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Integer,
    String,
    Float,
    Boolean,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from typing import List, Optional
import numpy as np
from connect import db_url  # external connection settings

# --- base class ------------------------------------------------------------
class Base(DeclarativeBase):
    """SQLAlchemy declarative base (abstract)."""
    __abstract__ = True


# --- Images table ----------------------------------------------------------
class Images(Base):
    """
    Stores image path and 512-dimensional embedding.
    """
    __tablename__ = "images"
    VECTOR_LENGTH = 512

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_path: Mapped[str] = mapped_column(String(256))
    image_embedding: Mapped[List[float]] = mapped_column(Vector(VECTOR_LENGTH))


# --- Games table -----------------------------------------------------------
class Games(Base):
    """
    Stores Steam-game metadata and 512-dimensional text embedding.
    """
    __tablename__ = "games"
    __table_args__ = {"extend_existing": True}
    VECTOR_LENGTH = 512  # distiluse-base-multilingual-cased-v2 output size

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    description: Mapped[str] = mapped_column(String(4096))
    windows: Mapped[bool] = mapped_column(Boolean)
    linux: Mapped[bool] = mapped_column(Boolean)
    mac: Mapped[bool] = mapped_column(Boolean)
    price: Mapped[float] = mapped_column(Float)
    game_description_embedding: Mapped[List[float]] = mapped_column(
        Vector(VECTOR_LENGTH)
    )


# --- engine & metadata -----------------------------------------------------
engine = create_engine(db_url)
Base.metadata.create_all(engine)
print("Tables created successfully")


# --- utility: insert image -------------------------------------------------
def insert_image(engine, image_path: str, image_embedding: List[float]) -> None:
    """Insert single image row."""
    with Session(engine) as session:
        img = Images(image_path=image_path, image_embedding=image_embedding)
        session.add(img)
        session.commit()


# --- seed a few demo images ------------------------------------------------
if __name__ == "__main__":
    N = 10  # demo rows
    for i in range(N):
        path = f"image_{i}.jpg"
        embed = np.random.rand(512).tolist()
        insert_image(engine, path, embed)
    print("Demo data inserted")

    # example similarity query
    with Session(engine) as s:
        original = s.execute(select(Images).limit(1)).scalar_one()
    with Session(engine) as s:
        similar = (
            s.execute(
                select(Images)
                .filter(Images.id != original.id)
                .order_by(
                    Images.image_embedding.cosine_distance(original.image_embedding)
                )
                .limit(5)
            )
            .scalars()
            .all()
        )
    print("Similar images:")
    for img in similar:
        print(img.id, img.image_path)
