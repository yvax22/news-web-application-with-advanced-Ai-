import os
import re  
import numpy as np
import pandas as pd
import faiss
import pyodbc
import uvicorn
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.docstore.document import Document
from embeddings_model import generate_embedding
from transformers import pipeline
import threading
import requests
import time
from pydantic import BaseModel
from fastapi import FastAPI,HTTPException
import openai 
import os 
from dotenv import load_dotenv
from openai import OpenAI
import httpx
from langchain.llms import Ollama
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.schema import Document
from typing import List, Dict, Optional
from datetime import datetime
from summarize import router as summarize_router , summarize_old_news
from check_database import update_faiss_loop, monitor_user_reads_and_update_recommendations
import os

# إعدادات البيئة
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
all_documents = []

# المسارات
NEWS_TEXTS_PATH = "C:/Users/yasee/Desktop/rag+llm/FastAPI/data/news_texts.npy"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

faiss_index = faiss.read_index("data/faiss_index_cosine.bin")
all_embeddings = np.load("data/embeddings_cosine.npy")
news_ids = np.load("data/news_ids_cosine.npy", allow_pickle=True).tolist()
#===========================================================================
# إعداد FastAPI
app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# تحميل FAISS والبيانات
faiss_index = faiss.read_index("C:/Users/yasee/Desktop/recommendatio_RAg/data/faiss_index.bin")
all_embeddings = np.load("C:/Users/yasee/Desktop/recommendatio_RAg/data/embeddings.npy")

conn = pyodbc.connect(
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=YASEEN;"
    "Database=FND-Yasseen;"
    "Trusted_Connection=yes;"
)

df_news = pd.read_sql("SELECT Id, Title, Content, image, CategoryId ,CreatedAt FROM News WHERE Title IS NOT NULL AND Content IS NOT NULL", conn)
print("✅ الأعمدة:", df_news.columns.tolist())
df_economy = df_news[df_news['Content'].str.contains("اقتصاد|تمويل|البنك|السوق", na=False)]
print("🔎 عدد الأخبار الاقتصادية:", len(df_economy))
print(df_economy[['Title', 'Content']].head(3))

# إعداد RAG
all_documents = [
    Document(page_content=f"{row['Title']} {row['Content']}")
    for _, row in df_news.iterrows()
]

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectorstore = LangFAISS.from_documents(all_documents, embedding_model)

llm = ChatOllama(
    model="iKhalid/ALLaM:7b",
    temperature=0.3,
    system_prompt="""أنت مساعد ذكي يساعد في الإجابة على الأسئلة بناءً على المعلومات المتوفرة في السياق.
    يجب أن:
    1. تستخدم المعلومات من السياق المقدم للإجابة على الأسئلة
    2. تجيب بشكل مباشر ومفيد
    3. لا تقل "لا يمكنني الإجابة" إذا كانت المعلومات متوفرة في السياق
    4. تلخص المعلومات المتوفرة بشكل واضح ومنظم
    5. تذكر أن لديك معلومات حقيقية في السياق يمكنك استخدامها
    6. عليك ان تكون قادر على توليد اجابات ذكية بناءا على السياق الذي لديك بحيث تكون الاجابة تتعلق بلسؤال 
    """

    
    )

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(
        search_kwargs={
            "k": 5,
            "score_threshold": 0.5,
            }
    ),
    chain_type="stuff",
    return_source_documents=True
    )


#=================================================================

# نماذج الإدخال
class RecommendRequest(BaseModel):
    user_id: str
    top_k: int = 10

class FullNews(BaseModel):
    news_id: str
    title: str
    content: str
    image: str | None = ""

class RAGRequest(BaseModel):
    question: str

def reload_vectorstore():
    global vectorstore
    vectorstore = LangFAISS.from_documents(all_documents, embedding_model)

class TextInput(BaseModel):
    text: str


#=================================================================
@app.on_event("startup")
def start_background_tasks():
    threading.Thread(target=update_faiss_loop, daemon=True).start()
    threading.Thread(target=monitor_user_reads_and_update_recommendations, daemon=True).start()
# التوصية
# الدالة المساعدة للصورة
def resolve_image_path(image_value: str) -> str:
    if not isinstance(image_value, str) or not image_value.strip():
        return "/images/default-news.png"
    if image_value.strip().lower().startswith(("http", "https")):
        return image_value.strip()
    return f"/images/{image_value.strip()}"
@app.post("/recommend")
def recommend_news(req: RecommendRequest):
    try:
        # 1. قراءة سجل المستخدم
        df_user = pd.read_sql("""
            SELECT TOP 5 N.Id, N.Title, N.Content, N.CategoryId, C.Name AS CategoryName, U.ReadAt
            FROM News N
            JOIN UserHistories U ON N.Id = U.NewsId
            LEFT JOIN Categories C ON N.CategoryId = C.Id
            WHERE U.UserId = ?
            ORDER BY U.ReadAt DESC
        """, conn, params=[req.user_id])

        if df_user.empty:
            return {"recommendations": [], "message": "🛑 لا توجد بيانات قراءة للمستخدم."}

        # 2. تحديد الفئة المسيطرة من سجل القراءة
        topic_weights = df_user['CategoryName'].value_counts(normalize=True).to_dict()
        dominant_topic = max(topic_weights, key=topic_weights.get)
        print(f"📊 الموضوع المسيطر: {dominant_topic} ({int(topic_weights[dominant_topic]*100)}%)")

        # 3. بناء تمثيل المستخدم
        user_embeddings = []
        for _, row in df_user.iterrows():
            text = f"{row['Title']} {row['Content']}"
            emb = generate_embedding(text).astype("float32")
            user_embeddings.append(emb)

        user_profile = np.mean(np.vstack(user_embeddings), axis=0).astype("float32")

        # 4. تحميل الفهرس
        faiss_index = faiss.read_index("data/faiss_index.bin")
        news_ids = np.load("data/news_ids.npy", allow_pickle=True).tolist()
        news_texts = np.load("data/news_texts.npy", allow_pickle=True).tolist()

        # 5. البحث عن الأخبار المشابهة
        D, I = faiss_index.search(np.expand_dims(user_profile, axis=0), 20)

        # 6. تحميل الأخبار الكاملة
        df_news = pd.read_sql("""
            SELECT N.Id, N.Title, N.Content, N.CategoryId, C.Name AS CategoryName, N.CreatedAt, N.image
            FROM News N
            LEFT JOIN Categories C ON N.CategoryId = C.Id
        """, conn)

        hybrid_results = []
        for idx in I[0]:
            if 0 <= idx < len(news_ids):
                nid = news_ids[idx]
                r = df_news[df_news["Id"].astype(str) == str(nid)]
                if not r.empty:
                    r = r.iloc[0]
                    created_at = r["CreatedAt"] if isinstance(r["CreatedAt"], datetime) else datetime.now()
                    final_score = D[0][list(I[0]).index(idx)]
                    hybrid_results.append({
                        "news_id": str(r["Id"]),
                        "title": r["Title"],
                        "abstract": r["Content"][:250],
                        "image": resolve_image_path(r.get("image", "")),
                        "source": "content",
                        "score": float(final_score),
                        "category_id": int(r.get("CategoryId")) if r.get("CategoryId") is not None else None,
                        "category_name": r.get("CategoryName", "عام"),
                        "created_at": created_at.isoformat()
                    })

        # 7. فلترة النتائج بحسب الموضوع المسيطر
        filtered_results = [r for r in hybrid_results if r.get("category_name") == dominant_topic]

        print(f"🧹 تم فلترة النتائج لتطابق الموضوع '{dominant_topic}' ({len(filtered_results)} نتيجة)")

        return {
            "dominant_topic": dominant_topic,
            "dominant_topic_percent": int(topic_weights[dominant_topic]*100),
            "recommendations": filtered_results
        }

    except Exception as e:
        return {"error": f"❌ خطأ في التوصية: {str(e)}"}

    
#=========================================================================

from html import unescape

from numpy.linalg import norm
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

@app.post("/rag_answer")
def generate_rag_answer(req: RAGRequest) :
    try:
        # 1. استرجاع الوثائق الأولية
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        raw_docs = retriever.invoke(req.question)
        print(f"📚 تم استرجاع {len(raw_docs)} وثيقة مبدئية.")

        # 2. حساب تشابه الوثائق مع السؤال
        question_emb = generate_embedding(req.question).astype("float32")
        filtered_docs = []
        for i, doc in enumerate(raw_docs):
            doc_emb = generate_embedding(doc.page_content).astype("float32")
            sim = cosine_similarity(question_emb, doc_emb)
            print(f"🔎 وثيقة {i+1} تشابه مع السؤال: {sim:.4f}")
            if sim >= 0.25:  # عتبة الصلة
                filtered_docs.append(doc)

        if not filtered_docs:
            print("🔁 لم يتم تجاوز العتبة، سيتم استخدام كل الوثائق الأولية.")
            filtered_docs = raw_docs  # fallback للوثائق المبدئية

            

        local_context = "\n\n".join([unescape(doc.page_content.strip()) for doc in filtered_docs])

        # 3. تحديد المجال
        def detect_domain(question):
            keywords = {
                "اقتصادي": ["اقتصاد", "نمو", "أسواق", "عملات", "أسعار"],
                "سياسي": ["رئيس", "انتخابات", "حكومة", "احتجاج", "دبلوماسي"],
                "رياضي": ["مباراة", "فريق", "لاعب", "دوري", "كأس"],
                "تقني": ["آبل", "Apple", "iPhone", "iOS", "ماك", "Mac", "تقنية", "تكنولوجيا"]
            }
            for domain, words in keywords.items():
                if any(word in question for word in words):
                    return domain
            return "عام"

        domain = detect_domain(req.question)

        prompts = {
            "اقتصادي": f"""أنت محلل اقتصادي محترف. لديك المعلومات التالية:\n\n{local_context}\n\nقم بتحليل وتلخيص الأخبار الاقتصادية بشكل منظم:""",
            "سياسي": f"""أنت محلل سياسي محترف. لديك المعلومات التالية:\n\n{local_context}\n\nقم بتحليل وتلخيص الأخبار السياسية بشكل منظم:""",
            "رياضي": f"""أنت محلل رياضي محترف. لديك المعلومات التالية:\n\n{local_context}\n\nقم بتحليل وتلخيص الأخبار الرياضية بشكل منظم:""",
            "تقني": f"""أنت محلل تقني محترف. لديك المعلومات التالية:\n\n{local_context}\n\nقم بتحليل وتلخيص الأخبار التقنية بشكل منظم:""",
            "عام": f"""أنت محلل إخباري محترف. لديك المعلومات التالية:\n\n{local_context}\n\nقم بتحليل وتلخيص الأخبار بشكل منظم:"""
        }

        selected_prompt = prompts.get(domain, prompts["عام"])

        # 4. توليد الإجابة
        answer = llm.invoke(selected_prompt)
        answer = answer.content if hasattr(answer, "content") else str(answer)

        print("\n🔍 السؤال:", req.question)
        print("📝 الإجابة:", answer)
        print("📄 عدد الوثائق المستخدمة:", len(filtered_docs))

        # 5. تقييم
        fallback_docs = [doc.page_content for doc in filtered_docs]

        return {
            "answer": answer,
            "sources": {
                "local_news": len(filtered_docs),
            },
            
            "fallback_documents": []
        }

    except Exception as e:
        print(f"❌ خطأ شامل: {str(e)}")
        return {
            "answer": "❌ حدث خطأ داخلي.",
            "sources": {"local_news": 0},
            "fallback_documents": []
        }

        

#=====================================================================================

# إضافة خبر إلى FAISS
@app.post("/add_news_to_faiss")
def add_news_to_faiss(news_id: str = Form(...)):
    global all_embeddings, vectorstore

    row = pd.read_sql(
        "SELECT Id, Title, Content FROM News WHERE Id = ?",
        conn,
        params=[news_id],
    )
    if row.empty:
        return {"status": "not_found"}

    text = f"{row.iloc[0]['Title']} {row.iloc[0]['Content']}"
    embedding = generate_embedding(text).astype("float32")
    doc = Document(page_content=text)

    # ✅ أضف الوثيقة إلى vectorstore مباشرةً
    vectorstore.add_documents([doc])

    # ✅ تحديث embeddings
    all_embeddings = np.vstack([all_embeddings, embedding])
    np.save("data/embeddings.npy", all_embeddings)
    doc = Document(page_content=text)
    vectorstore.add_documents([doc])
    all_documents.append(doc)
    print(f"✅ Added news_id={news_id} to FAISS and vectorstore")
    return {"status": "added"}

#=========================================================================

# API لإضافة خبر يدويًا
from langchain.docstore.document import Document

@app.post("/api/add_news_full")
async def add_news_full(news: FullNews):
    global all_embeddings, vectorstore

    # نص الخبر الكامل
    text = f"{news.title} {news.content}"
    
    # إنشاء وثيقة LangChain
    doc = Document(page_content=text)

    # توليد embedding وإضافة الوثيقة إلى vectorstore
    embedding = generate_embedding(text).astype("float32")
    vectorstore.add_documents([doc])

    # تحديث الـ FAISS وملف الـ embeddings
    new_embeddings = np.vstack([all_embeddings, embedding])
    np.save("data/embeddings.npy", new_embeddings)
    faiss.write_index(faiss_index, "data/faiss_index.bin")
    all_embeddings = new_embeddings
    doc = Document(page_content=text)
    vectorstore.add_documents([doc])
    all_documents.append(doc)
    print(f"✅ [API] Added news_id={news.news_id} to FAISS and vectorstore")
    return {"status": "added", "news_id": news.news_id}

#==============================================================================

# تصنيف الأخبار العامة عند بدء التشغيل
CATEGORY_TRANSLATIONS = {
    "Politics": "سياسة",
    "Culture": "ثقافة",
    "Sports": "رياضة",
    "Medical": "طب",
    "Tech": "تكنولوجيا",
    "Religion": "دين",
    "Finance": "اقتصاد",
    "Education": "تعليم",
    "StyleAndBeauty": "أسلوب وجمال",
    "Entertainment": "ترفيه",
    "Crime": "جريمة",
    "General": "عام"
}
classifier = pipeline("text-classification", model="Ammar-alhaj-ali/arabic-MARBERT-news-article-classification")

@app.get("/classify_all_general_news")
def classify_all_general_news():
    cursor = conn.cursor()
    cursor.execute("SELECT Id, Title, Content FROM News WHERE CategoryId = 12")
    news_rows = cursor.fetchall()
    updated = 0

    for row in news_rows:
        news_id, title, content = row
        full_text = f"{title} {content}"
        result = classifier(full_text)[0]
        english_label = result["label"]
        predicted_category = CATEGORY_TRANSLATIONS.get(english_label, english_label)

        cursor.execute("SELECT Id FROM Categories WHERE Name = ?", predicted_category)
        existing_cat = cursor.fetchone()

        if existing_cat:
            category_id = existing_cat[0]
        else:
            cursor.execute("INSERT INTO Categories (Name) OUTPUT INSERTED.Id VALUES (?)", predicted_category)
            category_id = cursor.fetchone()[0]
            conn.commit()

        cursor.execute("UPDATE News SET CategoryId = ? WHERE Id = ?", category_id, news_id)
        conn.commit()
        updated += 1

    return {"updated_news": updated}

# تنفيذ التصنيف التلقائي عند الإقلاع
def auto_classify_on_startup():
    time.sleep(2)
    try:
        requests.get("http://127.0.0.1:8500/classify_all_general_news")
        print("✅ تم تصنيف الأخبار العامة تلقائيًا.")
    except Exception as e:
        print(f"❌ فشل الاتصال الذاتي: {e}")

threading.Thread(target=auto_classify_on_startup).start()
#====================================================================================================================

app.include_router(summarize_router)

@app.on_event("startup")
def startup_event():
    threading.Thread(target=summarize_old_news).start()

# تشغيل التطبيق  =================================================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8500)

#===================================================================================================
