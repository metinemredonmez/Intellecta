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

from hallucination_guard import (
    check_claims,
    extract_claims_for_guard,
)

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
    result = model.transcribe(path)
    return (result.get("text") or "").strip()

def oai_complete(client, prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

def print_guard_report(title: str, results):
    print(f"\n=== Hallucination Check: {title} ===")
    if not results:
        print("(no claims)")
        return
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] CLAIM: {r['claim']}")
        print(f"  - similarity_ok: {r['similarity_ok']} (scores={ [round(s,3) for s in r['similarity_scores']] })")
        print(f"  - llm_grounded:  {r['llm_grounded']} (conf={r['llm_conf']:.2f})")
        print(f"  - overall_confidence: {r['overall_confidence']:.2f}  {'⚠️' if r['flagged'] else '✅'}")
        if r["support_passages"]:
            sp = r['support_passages'][0][:200].replace("\n", " ")
            print(f"  - support: {sp}...")
        print(f"  - reason: {r['reason']}")

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
    qa_result = qa.invoke(query)
    # RetrievalQA sometimes returns dict; show safe
    answer_text = qa_result["result"] if isinstance(qa_result, dict) and "result" in qa_result else str(qa_result)

    print("\n--- Query ---\n", query)
    print("\n--- Answer ---\n", answer_text)

    # 4) Hallucination Guard – claims from summary/actions/qa
    claims = extract_claims_for_guard(summary, actions, answer_text)
    sum_res = check_claims(transcript, claims["summary"])
    act_res = check_claims(transcript, claims["actions"])
    qa_res  = check_claims(transcript, claims["qa"])

    print_guard_report("Summary", sum_res)
    print_guard_report("Action Items", act_res)
    print_guard_report("QA Answer", qa_res)

if __name__ == "__main__":
    main()
