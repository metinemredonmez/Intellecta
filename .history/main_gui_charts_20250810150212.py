# main_gui_charts.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # set BEFORE importing whisper/faiss

import whisper
import tkinter as tk
from tkinter import filedialog, scrolledtext
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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

def pie_in_frame(frame, labels, values, title):
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct="%1.0f%%", startangle=140)
    ax.set_title(title)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=5)

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

def analyze_audio_with_charts(audio_path: str):
    output_box.delete(1.0, tk.END)
    charts_frame.destroy(); create_charts_frame()

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

    # 3) Sentiment (single word)
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
    vstore = FAISS.from_documents(docs, embeddings)_
