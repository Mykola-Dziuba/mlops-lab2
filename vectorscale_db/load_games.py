# load_games.py
# --- imports ------------------------------------------------------------
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sqlalchemy.orm import Session
from models import Games, engine        # uses classes from models.py

# --- model --------------------------------------------------------------
checkpoint = "distiluse-base-multilingual-cased-v2"
model = SentenceTransformer(checkpoint)

def generate_embeddings(text: str) -> list[float]:
    """Return 512-dimensional embedding for given text."""
    return model.encode(text).tolist()

# --- dataset ------------------------------------------------------------
columns_to_keep = ["Name", "Windows", "Linux", "Mac",
                   "About the game", "Supported languages", "Price"]
N = 40_000
dataset = (
    load_dataset("FronkonGames/steam-games-dataset", split="train")
    .select_columns(columns_to_keep)
    .select(range(N))
)

# --- insert loop --------------------------------------------------------
# --- insert loop (batched) ----------------------------------------------
BATCH = 128                          # batch size
batch_texts = []
batch_rows = []

with tqdm(total=len(dataset)) as pbar:
    for row in dataset:
        name = row["Name"]
        desc = row["About the game"] or ""
        if not name or not desc:              # skip bad rows
            pbar.update(1)
            continue

        batch_texts.append(desc)
        batch_rows.append(row)

        # when batch is full – process
        if len(batch_texts) == BATCH:
            embeddings = model.encode(batch_texts).tolist()  # vectorize once
            objects = [
                Games(
                    name=r["Name"],
                    description=t[:4096],
                    windows=r["Windows"],
                    linux=r["Linux"],
                    mac=r["Mac"],
                    price=r["Price"],
                    game_description_embedding=e,
                )
                for r, t, e in zip(batch_rows, batch_texts, embeddings)
            ]
            with Session(engine) as session:
                session.bulk_save_objects(objects)   # one insert
                session.commit()
            batch_texts.clear()
            batch_rows.clear()
            pbar.update(BATCH)

    # flush remaining <BATCH rows
    if batch_texts:
        embeddings = model.encode(batch_texts).tolist()
        objects = [
            Games(
                name=r["Name"],
                description=t[:4096],
                windows=r["Windows"],
                linux=r["Linux"],
                mac=r["Mac"],
                price=r["Price"],
                game_description_embedding=e,
            )
            for r, t, e in zip(batch_rows, batch_texts, embeddings)
        ]
        with Session(engine) as session:
            session.bulk_save_objects(objects)
            session.commit()
        pbar.update(len(batch_rows))


