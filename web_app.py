import os
from pathlib import Path
import json
import re
import uuid
from flask import Flask, render_template, request, jsonify, session, send_file
from werkzeug.utils import secure_filename

from src.chatbot_engine import (
    handle_personal_information, instant_casual_response, is_goodbye,
    route_knowledge_question, search_pakistan,
    answer_physics_with_question_forms, search_maths_with_question_forms,
    search_programming_with_question_forms, search_general_knowledge,
    pakistan_data, pakistan_questions, pakistan_embeddings, pakistan_vocabulary,
    general_knowledge_data, general_questions, general_embeddings, general_vocabulary,
    user_profile, save_user_profile,
)
from src.database import init_db, create_session, add_message, get_session_messages, get_history, set_title_if_new, add_uploaded_file, analytics, delete_session
from src.file_handler import extract_text, summarize_text, answer_from_file, SUPPORTED_EXTENSIONS

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "ai-knowledge-chatbot-local-key-change-me")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
init_db()


def get_session_id():
    if "session_id" not in session:
        session["session_id"] = create_session()
    return session["session_id"]


def knowledge_answer(question):
    personal = handle_personal_information(question)
    if personal:
        return personal, "personal"

    # Handle goodbye messages before knowledge routing.
    # In the GUI, "bye" should receive a normal response rather than
    # falling through to the unknown-question message.
    if is_goodbye(question):
        return "Goodbye! 👋 Take care and have a great day!", "casual"

    casual = instant_casual_response(question)
    if casual:
        return casual, "casual"
    domain = route_knowledge_question(question)
    answer = None
    try:
        if domain == "pakistan":
            answer = search_pakistan(question, pakistan_data, pakistan_questions, pakistan_embeddings, pakistan_vocabulary)
        elif domain == "physics":
            answer = answer_physics_with_question_forms(question)
        elif domain == "maths":
            answer = search_maths_with_question_forms(question)
        elif domain == "programming":
            from src.chatbot_engine import search_programming_with_question_forms
            answer = search_programming_with_question_forms(question)
        elif domain == "general_knowledge":
            answer = search_general_knowledge(question, general_knowledge_data, general_questions, general_embeddings, general_vocabulary)
    except Exception:
        answer = None
    if answer:
        return answer, domain or "knowledge"
    return "I don't have enough information in my knowledge base to answer that question yet.", "unknown"


@app.route("/")
def index():
    sid = get_session_id()
    messages = get_session_messages(sid)
    return render_template("index.html", messages=messages, profile=user_profile)


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = str(data.get("message", "")).strip()
    input_type = "voice" if data.get("input_type") == "voice" else "text"
    if not question:
        return jsonify({"ok": False, "error": "Please enter a message."}), 400
    sid = get_session_id()
    answer, domain = knowledge_answer(question)
    add_message(sid, "user", question, domain, input_type)
    add_message(sid, "assistant", answer, domain, input_type)
    set_title_if_new(sid, question)
    return jsonify({"ok": True, "answer": answer, "domain": domain})


@app.get("/api/history")
def history():
    return jsonify(get_history())


@app.get("/api/history/<int:session_id>")
def history_session(session_id):
    session["session_id"] = session_id
    return jsonify({"messages": get_session_messages(session_id)})


@app.post("/api/new-chat")
def new_chat():
    session["session_id"] = create_session()
    return jsonify({"ok": True})


@app.delete("/api/history/<int:session_id>")
def delete_history(session_id):
    delete_session(session_id)
    if session.get("session_id") == session_id:
        session["session_id"] = create_session()
    return jsonify({"ok": True})


@app.get("/api/analytics")
def get_analytics():
    return jsonify(analytics())


@app.get("/api/profile")
def profile():
    return jsonify(user_profile)


@app.post("/api/profile")
def update_profile():
    data = request.get_json(silent=True) or {}
    allowed = {"name", "age", "studies", "university", "interests"}
    for key in allowed:
        if key in data:
            user_profile[key] = data[key]
    save_user_profile(user_profile)
    return jsonify({"ok": True, "profile": user_profile})


@app.post("/api/upload")
def upload_file():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "No file selected."}), 400
    filename = secure_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({"ok": False, "error": "Supported formats: PDF, DOCX, PPTX, XLSX, XLS, CSV, TXT, MD, JSON."}), 400
    saved = UPLOAD_DIR / filename
    file.save(saved)
    try:
        text = extract_text(saved)
        if not text.strip():
            raise ValueError("No readable text found in the file.")
    except Exception as exc:
        saved.unlink(missing_ok=True)
        return jsonify({"ok": False, "error": str(exc)}), 400
    sid = get_session_id()
    add_uploaded_file(sid, filename, ext.lstrip("."))

    # Keep the extracted text available for optional follow-up questions
    # about the uploaded file. The summary itself is generated immediately
    # so the user does not have to press another button or answer a prompt.
    text_id = uuid.uuid4().hex
    text_path = UPLOAD_DIR / f"{text_id}.txt"
    text_path.write_text(text, encoding="utf-8")
    session["file_text_path"] = str(text_path)
    session["file_name"] = filename

    summary = summarize_text(text)
    add_message(sid, "user", f"Uploaded file: {filename}", "file", "text")
    add_message(
        sid,
        "assistant",
        f"Summary of {filename}:\n\n{summary}",
        "file",
        "text",
    )

    return jsonify({
        "ok": True,
        "filename": filename,
        "characters": len(text),
        "session": str(sid),
        "summary": summary,
    })


@app.post("/api/file/summary")
def file_summary():
    text_path = Path(session.get("file_text_path", ""))
    text = text_path.read_text(encoding="utf-8", errors="ignore") if text_path.exists() else ""
    if not text:
        return jsonify({"ok": False, "error": "Upload a file first."}), 400
    summary = summarize_text(text)
    sid = get_session_id()
    add_message(sid, "assistant", f"Summary of {session.get('file_name','uploaded file')}:\n\n{summary}", "file", "text")
    return jsonify({"ok": True, "filename": session.get("file_name"), "summary": summary})


@app.post("/api/file/ask")
def file_ask():
    question = str((request.get_json(silent=True) or {}).get("question", "")).strip()
    text_path = Path(session.get("file_text_path", ""))
    text = text_path.read_text(encoding="utf-8", errors="ignore") if text_path.exists() else ""
    if not text:
        return jsonify({"ok": False, "error": "Upload a file first."}), 400
    if not question:
        return jsonify({"ok": False, "error": "Ask a question about the file."}), 400
    answer = answer_from_file(question, text)
    return jsonify({"ok": True, "answer": answer})


@app.get("/api/export/current")
def export_current():
    sid = get_session_id()
    return export_chat(sid)


@app.get("/api/export/<int:session_id>")
def export_chat(session_id):
    messages = get_session_messages(session_id)
    history = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    out = Path("database") / f"chat_{session_id}.txt"
    out.write_text(history, encoding="utf-8")
    return send_file(out, as_attachment=True, download_name=f"chat_{session_id}.txt", mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)