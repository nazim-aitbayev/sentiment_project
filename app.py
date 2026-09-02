from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import joblib
import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent 

app = FastAPI()

# Загружаем модель и TF-IDF
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Настриаиваем AI-агента
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("Не задан OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4.1-mini",temperature=0)

@tool
def predict_sentiment_tool(text:str) -> str:
    """Определяет тональность отзыва: positive, neutral или negative."""
    text_tfidf = tfidf.transform([text])
    prediction = model.predict(text_tfidf)[0]
    return prediction

agent = create_agent(model=llm, tools=[predict_sentiment_tool],
        system_prompt = ("Ты аналитик отзывов для бизнеса. " 
                         "Для определения тональности отзыва обязательно используй "
                                               "инструмент predict_sentiment_tool. "
                         "После определения тональности кратко объясни результат"))
print("Агент успешно создан")

@app.get("/",
response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Анализ тональности отзывов</title>
        </head>
        <body>
            <h1>Анализ тональности отзывов</h1>
            <p>Введите отзыв, и модель определит его тональность.</p>

            <textarea id="review" rows="5" cols="60" placeholder="Введите отзыв"></textarea>
            <br><br>
            <button onclick="predict()">Определить тональность</button>

            <p id="result"></p>

            <script>
                async function predict() {
                    const text = document.getElementById("review").value;
                    const response = await fetch(
                        "/agent?text=" + encodeURIComponent(text)
                    );
                    const data = await response.json();
                    
                 document.getElementById("result").innerText = "Анализ AI-агента: " + data.analysis;              
                
 }
            </script>
        </body>
    </html>
    """

@app.get("/predict")
def predict(text: str):
    text_tfidf = tfidf.transform([text])
    prediction = model.predict(text_tfidf)[0]

    return {
        "text": text,
        "sentiment": prediction
    }

@app.get("/agent")
def analyze_with_agent(text: str):
    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"Определи тональность отзыва: {text}"
            }
        ]
    })

    return {
        "text": text,
        "analysis": response["messages"][-1].content
    }
