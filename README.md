Intellecta – ASR + NLP + RAG Demo
Intellecta is a Python-based demo application that combines Automatic Speech Recognition (ASR), Natural Language Processing (NLP), and Retrieval-Augmented Generation (RAG) to process meeting audio files, extract insights, and visualize results.

🚀 Features
🎙 ASR (Speech-to-Text): Converts audio into text using OpenAI Whisper.
📝 Summarization: Generates concise bullet-point summaries.
😊 Sentiment Analysis: Detects tone (Positive / Negative / Neutral).
✅ Action Item Extraction: Identifies actionable tasks.
📚 RAG (Retrieval-Augmented Generation): FAISS + LangChain for Q&A over transcripts.
📊 Charts: Optional sentiment/topic pie charts.
🖥 GUI Versions: Tkinter interfaces for easy interaction.
📂 Project Structure
File	Description
main.py	CLI version: Processes audio and prints results.
main_gui.py	GUI version: File selection + results display.
main_gui_charts.py	GUI + Charts version: Adds sentiment/topic charts.
sample_audio.py	Generates sample_audio.wav using macOS say or pyttsx3.
requirements.txt	Python dependencies.
.env	Stores OPENAI_API_KEY.
⚙️ Installation
# 1️⃣ Clone
git clone https://github.com/yourusername/Intellecta.git
cd Intellecta

# 2️⃣ Create venv

# Intellecta – ASR + NLP + RAG Demo

**Intellecta** is a Python-based demo application that combines Automatic Speech Recognition (ASR), Natural Language Processing (NLP), and Retrieval-Augmented Generation (RAG) to process meeting audio files, extract insights, and visualize results.

---

## 🚀 Features

- **🎙 ASR (Speech-to-Text):** Converts audio into text using OpenAI Whisper.
- **📝 Summarization:** Generates concise bullet-point summaries.
- **😊 Sentiment Analysis:** Detects tone (Positive / Negative / Neutral).
- **✅ Action Item Extraction:** Identifies actionable tasks.
- **📚 RAG (Retrieval-Augmented Generation):** FAISS + LangChain for Q&A over transcripts.
- **📊 Charts:** Optional sentiment/topic pie charts.
- **🖥 GUI Versions:** Tkinter interfaces for easy interaction.

---

## 📂 Project Structure

| File                | Description                                         |
|---------------------|-----------------------------------------------------|
| `main.py`           | CLI version: Processes audio and prints results.    |
| `main_gui.py`       | GUI version: File selection + results display.      |
| `main_gui_charts.py`| GUI + Charts version: Adds sentiment/topic charts.  |
| `sample_audio.py`   | Generates `sample_audio.wav` using macOS say or pyttsx3. |
| `requirements.txt`  | Python dependencies.                                |
| `.env`              | Stores `OPENAI_API_KEY`.                            |

---

## ⚙️ Installation

```bash
# 1️⃣ Clone
git clone https://github.com/yourusername/Intellecta.git
cd Intellecta

# 2️⃣ Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt
requirements.txt
openai>=1.30.0
langchain
langchain-community
langchain-openai
faiss-cpu
openai-whisper
tiktoken
matplotlib
python-dotenv
pyttsx3
🔑 API Key
Create a .env file in the project root:

OPENAI_API_KEY=your_api_key_here
🎙️ Sample Audio
Generate a sample audio file:

python sample_audio.py
On macOS: Uses say + ffmpeg (preferred).
On other systems: Falls back to pyttsx3.
▶️ Run
# CLI
python main.py

# GUI
python main_gui.py

# GUI + Charts
python main_gui_charts.py
🛠 Tech Stack
Python 3.9+
Whisper (ASR)
OpenAI GPT-4o-mini (NLP)
LangChain + FAISS (RAG)
Tkinter + Matplotlib (GUI/Charts)
python-dotenv (Environment Variables)
📌 Workflow
flowchart TD
    A[Audio File] --> B[Whisper ASR]
    B --> C[Transcript]
    C --> D[Summarization - GPT-4o-mini]
    C --> E[Sentiment Analysis - GPT-4o-mini]
    C --> F[Action Items - GPT-4o-mini]
    C --> G[Vector DB - FAISS]
    G --> H[RAG QA]
    D & E & F & H --> I[GUI / CLI Output]
    I --> J[Optional Charts: Sentiment + Topics]
📜 License
MIT License. See LICENSE for details. A[Audio File] --> B[Whisper ASR] B --> C[Transcript] C --> D[Summarization - GPT-4o-mini] C --> E[Sentiment Analysis - GPT-4o-mini] C --> F[Action Items - GPT-4o-mini] C --> G[Vector DB - FAISS] G --> H[RAG QA] D & E & F & H --> I[GUI / CLI Output] I --> J[Optional Charts: Sentiment + Topics]


## 📜 License
MIT License. See LICENSE for details.
