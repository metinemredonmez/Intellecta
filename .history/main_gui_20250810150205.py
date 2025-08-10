# main_gui.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # must be set before importing whisper/faiss

import whisper
import tkinter as tk
from tkinter import filedialog, scrolledtext
import threading
from dotenv import load_dotenv

from openai import OpenAI as OpenAIClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

from hallucination_guard import check_claims, extract_claims_for_guard

load_dotenv()
client = OpenAIClient()

def insert_guard_section(title: str, results, box: tk.Text):
    box.insert(tk.END, f"\n=== Hallucination Check: {title} ===\n")
    if not results:
        box.insert(tk.END, "(no claims)\n")
        return
    for r in results:
        tag = "ok"
        if r["overall_confidence"] < 0.5 or (not r["similarity_ok"]) or (not r["llm_grounded"]):
            tag = "warn"
        line = (
            f"- CLAIM: {r['claim']}\n"
            f"  overall_conf={r['overall_confidence']:.2f}  "
            f"similarity_ok={r['similarity_ok']}  llm_grounded={r['llm_grounded']}\n"
        )
        box.insert(tk.END, line, tag)

def analyze_audio(audio_path: str):
    output_box.delete(1.0, tk.END)
    output_box.insert(tk.END, f"🔹 Selected file: {audio_path}\n\n")

    # 1) ASR
    output_box.insert(tk.END, "🔹 Running ASR...\n")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    transcript = result["text"].strip()
    output_box.insert(tk.END, "\n--- Transcript ---\n" + transcript + "\n")

    # 2) Summary
    summary = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Summarize this conversation in 3 bullet points:\n{transcript}"}],
        temperature=0
    ).choices[0].message.content
    output_box.insert(tk.END, "\n--- Summary ---\n" + summary + "\n")

    # 3) Sentiment
    sentiment = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Overall sentiment as one word (positive/negative/neutral):\n{transcript}"}],
        temperature=0
    ).choices[0].message.content
    output_box.insert(tk.END, "\n--- Sentiment ---\n" + sentiment + "\n")

    # 4) Action Items
    actions = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Extract concrete action items as a bullet list:\n{transcript}"}],
        temperature=0
    ).choices[0].message.content
    output_box.insert(tk.END, "\n--- Action Items ---\n" + actions + "\n")

    # 5) RAG QA
    output_box.insert(tk.END, "\n🔹 Building Vector DB...\n")
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
    answer_text = qa_result["result"] if isinstance(qa_result, dict) and "result" in qa_result else str(qa_result)
    output_box.insert(tk.END, "\n--- Query ---\n" + query + "\n")
    output_box.insert(tk.END, "\n--- Answer ---\n" + answer_text + "\n")

    # 6) Hallucination Guard
    claims = extract_claims_for_guard(summary, actions, answer_text)
    sum_res = check_claims(transcript, claims["summary"])
    act_res = check_claims(transcript, claims["actions"])
    qa_res  = check_claims(transcript, claims["qa"])

    insert_guard_section("Summary", sum_res, output_box)
    insert_guard_section("Action Items", act_res, output_box)
    insert_guard_section("QA Answer", qa_res, output_box)

def process_audio():
    path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if not path:
        return
    threading.Thread(target=analyze_audio, args=(path,), daemon=True).start()

root = tk.Tk()
root.title("Intellecta - ASR + NLP + RAG Demo (with Hallucination Guard)")
root.geometry("860x680")

btn = tk.Button(root, text="🎤 Select Audio & Analyze", command=process_audio, font=("Arial", 14))
btn.pack(pady=10)

output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=32)
output_box.tag_config("warn", foreground="#b22222")
output_box.tag_config("ok", foreground="#1a7f37")
output_box.pack(pady=10)

root.mainloop()
