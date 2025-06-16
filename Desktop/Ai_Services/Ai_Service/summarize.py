from fastapi import APIRouter, Form, Body
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# تحميل النموذج
model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# إنشاء Router خاص بالتلخيص
router = APIRouter()

# نموذج الإدخال لـ JSON
class NewsRequest(BaseModel):
    news_id: int
    content: str

# دالة التلخيص
def summarize(text: str) -> str:
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        input_ids,
        max_length=100,
        min_length=25,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)



@router.post("/summarize")
def summarize_news(news: NewsRequest):
    summary = summarize(news.content)
    # اختياري: أرسل ملخص الخبر إلى قاعدة البيانات هنا أو أعده للعميل
    return {"news_id": news.news_id, "Abstract": summary}




import pyodbc
import threading
import time
def get_connection():
    return pyodbc.connect(
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=YASEEN;"
        "Database=FND-Yasseen;"
        "Trusted_Connection=yes;"  
    )
def summarize_old_news():
    time.sleep(5)  # انتظر قليلاً لضمان تشغيل السيرفر
    conn = get_connection()
    cursor = conn.cursor()

    print("🔁 البحث عن الأخبار القديمة غير المُلخصة...")

    cursor.execute("SELECT Id, Content FROM News WHERE Abstract IS NULL OR Abstract = ''")
    rows = cursor.fetchall()

    print(f"📋 عدد الأخبار غير الملخصة: {len(rows)}")

    for row in rows:
        news_id = row.Id
        content = row.Content

        try:
            summary = summarize(content)
            update_query = "UPDATE News SET Abstract = ? WHERE Id = ?"
            cursor.execute(update_query, (summary, news_id))
            conn.commit()
            print(f"✅ تم تلخيص الخبر ID={news_id}")
        except Exception as e:
            print(f"❌ خطأ عند تلخيص الخبر ID={news_id}: {e}")

    cursor.close()
    conn.close()

