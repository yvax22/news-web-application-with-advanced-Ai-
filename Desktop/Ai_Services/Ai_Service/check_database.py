import threading
import time
import hashlib
import pyodbc
import numpy as np
import pandas as pd
import faiss
from embeddings_model import generate_embedding
import os
def update_faiss_loop():
    while True:
        try:
            # إعداد الاتصال وق paths
            DB_CONN_STR = (
                "Driver={ODBC Driver 17 for SQL Server};"
                "Server=YASEEN;"
                "Database=FND-Yasseen;"
                "Trusted_Connection=yes;"
            )

            FAISS_INDEX_PATH = "data/faiss_index.bin"
            EMBEDDINGS_PATH = "data/embeddings.npy"
            NEWS_IDS_PATH = "data/news_ids.npy"
            NEWS_TEXTS_PATH = "data/news_texts.npy"
            LOG_PATH = "data/added_news_log.csv"

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
                faiss_index = faiss.IndexFlatL2(768)
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
                    embedding = generate_embedding(text)
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
                print(f"🧠 تم تحديث قاعدة المعرفة بـ {len(new_entries)} خبر جديد.")
            else:
                print("🔍 لا توجد أخبار جديدة.")

        except Exception as e:
            print(f"❌ خطأ أثناء التحديث المجدول: {e}")

        time.sleep(60)  # تحديث كل 30 ثانية
def monitor_user_reads_and_update_recommendations():
    while True:
        try:
            DB_CONN_STR = (
                "Driver={ODBC Driver 17 for SQL Server};"
                "Server=YASEEN;"  
                "Database=FND-Yasseen;"
                "Trusted_Connection=yes;"
            )

            conn = pyodbc.connect(DB_CONN_STR)

            # تحميل قاعدة FAISS
            faiss_index = faiss.read_index("data/faiss_index.bin")
            df_news = pd.read_sql("SELECT Id, Title, Content, CategoryId FROM News", conn)
            df_users = pd.read_sql("SELECT DISTINCT UserId FROM UserHistories", conn)

            for user_id in df_users["UserId"].tolist():
                df_user_reads = pd.read_sql(
                    '''
                    SELECT TOP 10 N.Id AS NewsId, N.Title, N.CategoryId, N.Content, U.ReadAt
                    FROM UserHistories U
                    JOIN News N ON N.Id = U.NewsId
                    WHERE U.UserId = ?
                    ORDER BY U.ReadAt DESC
                    ''',
                    conn, params=[user_id]
                )

                print(f"📡 توصيات محدثة للمستخدم {user_id}")
                print(df_user_reads[['Title', 'CategoryId']])

                user_embeddings = []
                for _, row in df_user_reads.iterrows():
                    text = f"{row['Title']} {row['Content']}"
                    emb = generate_embedding(text).astype("float32")
                    user_embeddings.append(emb)

                if user_embeddings:
                    user_profile = np.mean(np.vstack(user_embeddings), axis=0).astype("float32")
                    D, I = faiss_index.search(np.expand_dims(user_profile, axis=0), 5)

                    print("📄 أخبار موصى بها:")
                    for idx in I[0]:
                        if 0 <= idx < len(df_news):
                            r = df_news.iloc[idx]
                            print(" -", r["Title"])

        except Exception as e:
            print(f"❌ خطأ أثناء مراقبة سجل القراءة: {e}")

        time.sleep(60)
