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

load_dotenv()
client = OpenAIClient()

def pie_in_frame(frame, labels, values, title):
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct="%1.0f%%", startangle=140)
    ax.set_title(title)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=5)

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
    vstore = FAISS.from_documents(docs, embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        chain_type="stuff",
        retriever=vstore.as_retriever()
    )
    query = "What action items were decided during the meeting?"
    answer = qa.invoke(query)  # updated
    output_box.insert(tk.END, "\n--- Query ---\n" + query + "\n")
    output_box.insert(tk.END, "\n--- Answer ---\n" + str(answer) + "\n")

    # ---- Charts ----
    # Sentiment chart (from single word)
    labels = ["Positive", "Negative", "Neutral"]
    vals = [1 if "positive" in sentiment.lower() else 0,
            1 if "negative" in sentiment.lower() else 0,
            1 if "neutral"  in sentiment.lower() else 0]
    if sum(vals) == 0: vals[2] = 1
    pie_in_frame(charts_frame, labels, vals, "Sentiment")

    # Topic chart (use summary bullets as pseudo-topics)
    topics = [t.strip("-• ").strip() for t in summary.split("\n") if t.strip()]
    if not topics: topics = ["General"]
    counts = [1] * len(topics)
    pie_in_frame(charts_frame, topics, counts, "Topics")

def process_audio():
    path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if not path:
        return
    threading.Thread(target=analyze_audio_with_charts, args=(path,), daemon=True).start()

def create_charts_frame():
    global charts_frame
    charts_frame = tk.Frame(root)
    charts_frame.pack(pady=5, fill="x")

root = tk.Tk()
root.title("Intellecta - ASR + NLP + RAG (Charts)")
root.geometry("980x900")

btn = tk.Button(root, text="🎤 Select Audio & Analyze", command=process_audio, font=("Arial", 14))
btn.pack(pady=10)

output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=120, height=28)
output_box.pack(pady=10)

create_charts_frame()
root.mainloop()
