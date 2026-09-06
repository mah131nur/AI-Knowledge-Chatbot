import csv
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rapidfuzz.fuzz import ratio
from rapidfuzz import process


# ============================================================
# LOAD GENERAL KNOWLEDGE DATASET
# ============================================================

def load_general_knowledge_data():

    data = []

    with open(
        "data/general_knowledge.csv",
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            data.append(row)

    return data


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    text = str(
        text
    ).lower()

    text = text.replace(
        "-",
        " "
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# BUILD VOCABULARY (for spelling correction only)
# ============================================================

def build_vocabulary(data):

    vocabulary = set()

    for row in data:

        question = preprocess_text(
            row["question"]
        )

        for word in question.split():

            if len(word) >= 3:

                vocabulary.add(
                    word
                )

    return sorted(
        vocabulary
    )


# ============================================================
# SPELLING CORRECTION
# ============================================================

def correct_spelling(
    text,
    vocabulary
):

    words = text.split()

    corrected_words = []

    for word in words:

        if len(word) <= 2:

            corrected_words.append(
                word
            )

            continue


        if word in vocabulary:

            corrected_words.append(
                word
            )

            continue


        match = process.extractOne(
            word,
            vocabulary,
            scorer=ratio
        )


        if match is None:

            corrected_words.append(
                word
            )

            continue


        matched_word = match[0]
        similarity = match[1]


        # ----------------------------------------------------
        # More conservative thresholds
        # ----------------------------------------------------

        if len(word) <= 4:

            threshold = 90

        elif len(word) <= 6:

            threshold = 84

        else:

            threshold = 80


        if similarity >= threshold:

            corrected_words.append(
                matched_word
            )

        else:

            corrected_words.append(
                word
            )


    return " ".join(
        corrected_words
    )


# ============================================================
# STOP WORDS
# ============================================================

def get_keywords(text):

    stop_words = {

        "what",
        "is",
        "the",
        "a",
        "an",

        "of",
        "in",
        "on",
        "at",

        "to",
        "from",

        "tell",
        "me",
        "about",

        "who",
        "was",
        "were",

        "where",
        "which",

        "are",

        "called",

        "does",
        "do",

        "when",

        "how",
        "many",

        "can",

        "for",

        "with",

        "and",

        "or",

        "did",

        "has",
        "have",

        "please",

        "give",

        "explain"
    }


    words = preprocess_text(
        text
    ).split()


    keywords = []

    for word in words:

        if word not in stop_words:

            keywords.append(
                word
            )


    return set(
        keywords
    )


# ============================================================
# PREPARE DATA
# ============================================================
#
# NOTE: This module used to build embeddings with a
# SentenceTransformer model ("all-MiniLM-L6-v2"). That model
# is now replaced with a TfidfVectorizer, matching the same
# approach already used in pakistan.py / maths.py / programming.py.
#
# "question_embeddings" below is now a TF-IDF sparse matrix
# instead of a dense neural embedding matrix, and "vocabulary"
# is now the fitted TfidfVectorizer object itself (not a plain
# word list) so it can be reused to transform the user's
# question at search time. The word-list vocabulary that used
# to power correct_spelling() is now built separately and
# returned alongside it.
# ============================================================

def prepare_general_knowledge_data(
    data
):

    questions = []

    for row in data:

        question = preprocess_text(
            row["question"]
        )

        questions.append(
            question
        )


    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True
    )


    question_embeddings = tfidf_vectorizer.fit_transform(
        questions
    )


    spelling_vocabulary = build_vocabulary(
        data
    )


    vocabulary = {
        "vectorizer": tfidf_vectorizer,
        "spelling_words": spelling_vocabulary
    }


    return (
        questions,
        question_embeddings,
        vocabulary
    )


# ============================================================
# SEARCH GENERAL KNOWLEDGE
# ============================================================

def search_general_knowledge(
    question,
    data,
    questions,
    question_embeddings,
    vocabulary
):

    if not question.strip():

        return None


    # ========================================================
    # CLEAN QUESTION
    # ========================================================

    question_clean = preprocess_text(
        question
    )


    # ========================================================
    # SPELLING CORRECTION
    # ========================================================

    spelling_words = vocabulary["spelling_words"]

    corrected_question = correct_spelling(
        question_clean,
        spelling_words
    )


    if corrected_question != question_clean:

        print(
            f"✏️ GK Corrected: "
            f"{corrected_question}"
        )


    # ========================================================
    # USER VECTOR (TF-IDF instead of a neural embedding)
    # ========================================================

    tfidf_vectorizer = vocabulary["vectorizer"]

    user_embedding = tfidf_vectorizer.transform(
        [corrected_question]
    )


    if user_embedding.nnz == 0:

        return None


    # ========================================================
    # SEMANTIC SIMILARITY (TF-IDF cosine similarity)
    # ========================================================

    similarities = cosine_similarity(
        user_embedding,
        question_embeddings
    )[0]


    # ========================================================
    # TOP CANDIDATES
    # ========================================================

    top_indices = (
        similarities
        .argsort()[-10:][::-1]
    )


    # ========================================================
    # USER INFORMATION
    # ========================================================

    user_keywords = get_keywords(
        corrected_question
    )

    user_words = set(
        corrected_question.split()
    )


    best_index = None
    best_score = 0


    # ========================================================
    # COMPARE CANDIDATES
    # ========================================================

    for index in top_indices:

        candidate_question = questions[
            index
        ]


        semantic_score = similarities[
            index
        ]


        fuzzy_score = ratio(
            corrected_question,
            candidate_question
        ) / 100


        candidate_keywords = get_keywords(
            candidate_question
        )


        # ----------------------------------------------------
        # Keyword overlap
        # ----------------------------------------------------

        if len(user_keywords) > 0:

            common_keywords = (
                user_keywords
                &
                candidate_keywords
            )

            keyword_score = (
                len(common_keywords)
                /
                len(user_keywords)
            )

        else:

            keyword_score = 0


        # ----------------------------------------------------
        # Direct word matching
        # ----------------------------------------------------

        candidate_words = set(
            candidate_question.split()
        )

        common_words = (
            user_words
            &
            candidate_words
        )


        if len(user_words) > 0:

            direct_word_match = (
                len(common_words)
                /
                len(user_words)
            )

        else:

            direct_word_match = 0


        # ----------------------------------------------------
        # Word count
        # ----------------------------------------------------

        word_count = len(
            corrected_question.split()
        )


        # ====================================================
        # BASE SCORE
        # ====================================================

        if word_count <= 2:

            final_score = (

                semantic_score * 0.35

                +

                fuzzy_score * 0.20

                +

                keyword_score * 0.30

                +

                direct_word_match * 0.15
            )

        elif word_count <= 5:

            final_score = (

                semantic_score * 0.40

                +

                fuzzy_score * 0.20

                +

                keyword_score * 0.25

                +

                direct_word_match * 0.15
            )

        else:

            final_score = (

                semantic_score * 0.50

                +

                fuzzy_score * 0.15

                +

                keyword_score * 0.20

                +

                direct_word_match * 0.15
            )


        # ====================================================
        # DIRECT WORD BOOST
        # ====================================================

        if direct_word_match >= 1.0:

            final_score += 0.15

        elif direct_word_match >= 0.75:

            final_score += 0.08


        # ====================================================
        # LIMIT SCORE
        # ====================================================

        if final_score > 1.0:

            final_score = 1.0


        # ====================================================
        # SELECT BEST
        # ====================================================

        if final_score > best_score:

            best_score = final_score

            best_index = index


    # ========================================================
    # FINAL DECISION
    # ========================================================

    if best_index is not None:

        if best_score >= 0.50:

            return data[
                best_index
            ]["answer"]


    return None