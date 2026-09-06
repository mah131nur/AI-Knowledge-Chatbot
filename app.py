# ============================================================
#                  AI KNOWLEDGE CHATBOT
#                  Terminal version
# ============================================================

from src.chatbot_engine import (
    handle_personal_information,
    instant_casual_response,
    route_knowledge_question,
    search_pakistan,
    answer_physics_with_question_forms,
    search_maths_with_question_forms,
    search_programming_with_question_forms,
    search_general_knowledge,
    pakistan_data, pakistan_questions, pakistan_embeddings, pakistan_vocabulary,
    general_knowledge_data, general_questions, general_embeddings, general_vocabulary,
    is_goodbye, looks_like_casual_conversation, casual_fallback,
    user_profile,
    maths_data, maths_questions, maths_embeddings, maths_vocabulary,
)
from src.chatbot_engine import (
    programming_data, programming_questions, programming_embeddings, programming_vocabulary,
)

print("\n=======================================================")
print("             🤖 AI KNOWLEDGE CHATBOT")
print("=======================================================")
print("🇵🇰 Pakistan | 🧮 Maths | ⚛️ Physics | 💻 Programming | 🌍 General Knowledge")
print("🧠 Personal information is saved in user_profile.json")
print("Type 'bye' to exit.")
print("=======================================================")

while True:
    try:
        user_question = input("\n👤 You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n🤖 Goodbye! 👋")
        break

    if not user_question:
        continue
    if is_goodbye(user_question):
        print("\n🤖 Goodbye! 👋 Take care and have a great day! 😊")
        break

    personal_answer = handle_personal_information(user_question)
    if personal_answer:
        print(f"\n🤖 Bot: {personal_answer}")
        continue

    instant_answer = instant_casual_response(user_question)
    if instant_answer:
        print(f"\n🤖 Bot: {instant_answer}")
        continue

    domain = route_knowledge_question(user_question)
    answer = None
    try:
        if domain == "pakistan":
            answer = search_pakistan(user_question, pakistan_data, pakistan_questions, pakistan_embeddings, pakistan_vocabulary)
        elif domain == "physics":
            answer = answer_physics_with_question_forms(user_question)
        elif domain == "maths":
            answer = search_maths_with_question_forms(user_question)
        elif domain == "programming":
            answer = search_programming_with_question_forms(user_question)
        elif domain == "general_knowledge":
            answer = search_general_knowledge(user_question, general_knowledge_data, general_questions, general_embeddings, general_vocabulary)
    except Exception as e:
        print(f"⚠️ {domain or 'Knowledge'} module error: {e}")

    if answer:
        print(f"\n🤖 Bot: {answer}")
        continue

    if looks_like_casual_conversation(user_question):
        answer = instant_casual_response(user_question) or casual_fallback()
        print(f"\n🤖 Bot: {answer}")
        continue

    print("\n🤖 I don't have enough information in my knowledge base to answer that question yet.")
