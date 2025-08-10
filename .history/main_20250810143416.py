# main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import whisper
from dotenv import load_dotenv

from openai import OpenAI as OpenAIClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

load_dotenv()  # reads .env

AUDIO_FILE = "sample_audio.wav"  # use WAV for Whisper stability

def require_api_key():
    api = os.getenv("OPENAI_API_KEY")
    if not api:
        print("ERROR: OPENAI_API_KEY is not set. Put it in .env or in PyCharm env vars.")
        sys.exit(1)
    return api

def transcribe(path: str) -> str:
    print("🔹 Starting ASR...")
    model = whisper.load_model("base")
    # omit language to let Whisper auto-detect (works for English/Turkish)
    result = model.transcribe(path)
    return (result.get("text") or "").strip()

def oai_complete(client, prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

def main():
    require_api_key()
    if not os.path.exists(AUDIO_FILE):
        print(f"ERROR: {AUDIO_FILE} not found. Create one (see sample_audio*.py).")
        sys.exit(1)

    # 1) ASR
    transcript = transcribe(AUDIO_FILE)
    print("\n--- Transcript ---\n", transcript or "(empty)")

    client = OpenAIClient()

    # 2) Summarize / Sentiment / Action Items
    summary   = oai_complete(client, f"Summarize this conversation in 3 bullet points:\n{transcript}")
    sentiment = oai_complete(client, f"Give the overall sentiment as a single word (positive/negative/neutral):\n{transcript}")
    actions   = oai_complete(client, f"Extract concrete action items as a bullet list:\n{transcript}")

    print("\n--- Summary ---\n", summary)
    print("\n--- Sentiment ---\n", sentiment)
    print("\n--- Action Items ---\n", actions)

    # 3) RAG QA over this transcript
    print("\n🔹 Building Vector DB...")
    docs = [Document(page_content=transcript, metadata={"source": "meeting"})]
    embeddings = OpenAIEmbeddings()
    vstore = FAISS.from_documents(docs, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        chain_type="stuff",
        retriever=vstore.as_retriever()
    )
    query = "What action items were decided during the meeting?"
    answer = qa.invoke(query)

    print("\n--- Query ---\n", query)
    print("\n--- Answer ---\n", answer)

if __name__ == "__main__":
    main()
