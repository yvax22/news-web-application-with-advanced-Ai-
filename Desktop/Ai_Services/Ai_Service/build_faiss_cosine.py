
import os
import hashlib
import pyodbc
import numpy as np
import pandas as pd
import faiss
from datetime import datetime
from embeddings_model import generate_embedding

FAISS_INDEX_PATH = "data/faiss_index_cosine.bin"
EMBEDDINGS_PATH = "data/embeddings_cosine.npy"
NEWS_IDS_PATH = "data/news_ids_cosine.npy"
NEWS_TEXTS_PATH = "data/news_texts_cosine.npy"
LOG_PATH = "data/added_news_log_cosine.csv"

DB_CONN_STR = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=YASEEN;"
    "Database=FND-Yasseen;"
    "Trusted_Connection=yes;"
)

if os.path.exists(LOG_PATH):
    added_df = pd.read_csv(LOG_PATH)
    added_hashes = set(added_df["hash"].tolist())
else:
    added_df = pd.DataFrame(columns=["hash", "news_id", "created_at"])
    added_hashes = set()

if os.path.exists(FAISS_INDEX_PATH):
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    all_embeddings = np.load(EMBEDDINGS_PATH)
    news_ids = np.load(NEWS_IDS_PATH, allow_pickle=True).tolist()
    news_texts = np.load(NEWS_TEXTS_PATH, allow_pickle=True).tolist()
else:
    faiss_index = faiss.IndexFlatIP(768)  # Cosine via Inner Product with normalized vectors
    all_embeddings = np.zeros((0, 768), dtype='float32')
    news_ids = []
    news_texts = []

conn = pyodbc.connect(DB_CONN_STR)
df_news = pd.read_sql("SELECT Id, Title, Content, CreatedAt FROM News WHERE Title IS NOT NULL AND Content IS NOT NULL", conn)

new_entries = []
for _, row in df_news.iterrows():
    news_id = str(row["Id"])
    text = f"{row['Title']} {row['Content']}"
    created_at = row["CreatedAt"]
    unique_hash = hashlib.md5((news_id + text).encode()).hexdigest()

    if unique_hash in added_hashes:
        continue

    try:
        embedding = generate_embedding(text).astype("float32")
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm  # Normalize to use cosine similarity
        faiss_index.add(embedding.reshape(1, -1))
        all_embeddings = np.vstack([all_embeddings, embedding])
        news_ids.append(news_id)
        news_texts.append(text)

        new_entries.append({
            "hash": unique_hash,
            "news_id": news_id,
            "created_at": created_at
        })

        print(f"✅ تمت إضافة الخبر {news_id}")
    except Exception as e:
        print(f"❌ فشل في الخبر {news_id}: {e}")

if new_entries:
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    np.save(EMBEDDINGS_PATH, all_embeddings)
    np.save(NEWS_IDS_PATH, news_ids)
    np.save(NEWS_TEXTS_PATH, news_texts)
    added_df = pd.concat([added_df, pd.DataFrame(new_entries)], ignore_index=True)
    added_df.to_csv(LOG_PATH, index=False)
    print(f"🧠 تم تحديث قاعدة المعرفة (Cosine) بـ {len(new_entries)} خبر جديد.")
else:
    print("🔍 لا توجد أخبار جديدة.")
