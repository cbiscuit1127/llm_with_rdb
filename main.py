import os
from dotenv import load_dotenv  # 追加
from getpass import getpass

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
# from langchain_openai import OpenAIEmbeddings  # Gemini用の埋め込みモデルを差し替える場合はここを変更

# =========================
# 1. 環境変数の設定
# =========================
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))  # 追加

# .envファイルからAPIキーを取得
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key = getpass("Enter your Gemini API Key:")
os.environ["GOOGLE_API_KEY"] = api_key

# =========================
# 2. データベース接続 (PostgreSQL)
# =========================
# 例: postgres://user:pass@localhost:5432/mydb
# pg_uri = getpass("Enter your PostgreSQL URI (e.g., postgresql+psycopg2://user:pass@localhost:5432/db): ")
pg_uri = 'postgresql+psycopg2://postgres:postgres@localhost:5432/pckeiba'
db = SQLDatabase.from_uri(pg_uri, sample_rows_in_table_info=3)

# =========================
# 3. Gemini モデルの定義
# =========================
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

# =========================
# 4. SQL例を定義
# =========================
examples = [
    {
        "input": "2023年の売上を月別に教えて",
        "query": "SELECT date_trunc('month', order_date) AS month, SUM(total) FROM orders WHERE order_date >= '2023-01-01' AND order_date < '2024-01-01' GROUP BY month ORDER BY month;"
    },
    {
        "input": "在庫が少ない商品を一覧にして",
        "query": "SELECT product_name, stock FROM products WHERE stock < 10;"
    },
    {
        "input": "顧客ごとの総売上を出して",
        "query": "SELECT customer_id, SUM(total) FROM orders GROUP BY customer_id;"
    }
]

# =========================
# 5. ベクトル検索でSQL例を選択
# =========================
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=GoogleGenerativeAIEmbeddings(model='embedding-001'),  # Gemini埋め込みモデルがあれば置換可
    vectorstore_cls=Chroma,
    k=3,
    input_keys=["input"]
)

# =========================
# 6. Few-shot プロンプト構築
# =========================
system_prefix = """
You are an agent designed to interact with a SQL database. Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples, limit your query to at most {top_k} results.
Avoid SELECT * and only query relevant columns.
Double-check SQL before executing. If unrelated to the database, respond with "I don't know".
Here are some examples of user inputs and their corresponding SQL queries:
"""

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template("User input: {input}\nSQL query: {query}"),
    input_variables=["input", "dialect", "top_k"],
    prefix=system_prefix,
    suffix=""
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# =========================
# 7. エージェント構築
# =========================
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    verbose=True
)

# =========================
# 8. 質問を入力して実行
# =========================
while True:
    query = input("\n質問を入力してください（終了するにはexit）: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = agent_executor.invoke({"input": query})
    print("\n回答:", result.get("output", "（出力なし）"))
