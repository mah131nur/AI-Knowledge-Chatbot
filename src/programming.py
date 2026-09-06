import os
import re
import pandas as pd

from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# File Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROGRAMMING_FILE = os.path.join(BASE_DIR, "data", "programming.csv")

MIN_CONFIDENCE = 0.45


# ============================================================
# Text Preprocessing
# ============================================================

def preprocess_text(text):
    """
    Normalize text for searching and matching.

    This function does not contain any hard-coded knowledge.
    It only cleans the user's text.
    """

    if text is None:
        return ""

    text = str(text).lower().strip()

    # Normalize common apostrophe variations
    text = text.replace("’", "'")
    text = text.replace("`", "'")

    # Normalize common written variations
    text = re.sub(r"\bwhats\b", "what is", text)
    text = re.sub(r"\bwhats\b", "what is", text)
    text = re.sub(r"\bwhat're\b", "what are", text)
    text = re.sub(r"\bwhats\b", "what is", text)

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# Load Programming Dataset
# ============================================================

def load_programming_data():
    """
    Load Programming questions and answers from programming.csv.

    Expected columns:
        question
        answer
    """

    if not os.path.exists(PROGRAMMING_FILE):
        raise FileNotFoundError(
            f"Programming dataset not found: {PROGRAMMING_FILE}"
        )

    data = pd.read_csv(PROGRAMMING_FILE)

    # Normalize column names
    data.columns = [
        str(column).strip().lower()
        for column in data.columns
    ]

    required_columns = {"question", "answer"}

    if not required_columns.issubset(set(data.columns)):
        raise ValueError(
            "programming.csv must contain these columns: "
            "question, answer"
        )

    # Remove rows with missing values
    data = data.dropna(
        subset=["question", "answer"]
    ).copy()

    # Convert everything to string
    data["question"] = data["question"].astype(str)
    data["answer"] = data["answer"].astype(str)

    # Remove completely empty questions/answers
    data = data[
        (data["question"].str.strip() != "") &
        (data["answer"].str.strip() != "")
    ].reset_index(drop=True)

    return data


# ============================================================
# Build Vocabulary
# ============================================================

def build_programming_vocabulary(data):
    """
    Build vocabulary automatically from programming.csv.

    No Programming terms are hard-coded here.
    """

    vocabulary = set()

    if data is None or data.empty:
        return vocabulary

    for question in data["question"]:
        cleaned = preprocess_text(question)

        for word in cleaned.split():
            if len(word) >= 2:
                vocabulary.add(word)

    return vocabulary


# ============================================================
# Spelling Correction
# ============================================================

def correct_spelling(text, vocabulary):
    """
    Correct likely spelling mistakes using the vocabulary
    extracted from the Programming dataset.

    Answers are never generated here.
    """

    cleaned = preprocess_text(text)

    if not cleaned:
        return ""

    words = cleaned.split()
    corrected_words = []

    vocabulary_list = list(vocabulary)

    if not vocabulary_list:
        return cleaned

    for word in words:

        # Already known
        if word in vocabulary:
            corrected_words.append(word)
            continue

        # Very short words are usually not worth correcting
        if len(word) <= 2:
            corrected_words.append(word)
            continue

        match = process.extractOne(
            word,
            vocabulary_list,
            scorer=fuzz.ratio
        )

        if match:
            matched_word, score, _ = match

            # Higher tolerance for longer words
            if len(word) >= 7:
                required_score = 68
            elif len(word) >= 5:
                required_score = 72
            else:
                required_score = 78

            if score >= required_score:
                corrected_words.append(matched_word)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)


# ============================================================
# Prepare Programming Data
# ============================================================

def prepare_programming_data(data):
    """
    Prepare Programming dataset for TF-IDF matching.

    Returns:
        questions
        embeddings
        vocabulary
    """

    if data is None or data.empty:
        return [], None, set()

    questions = [
        preprocess_text(question)
        for question in data["question"].tolist()
    ]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        sublinear_tf=True,
        lowercase=True
    )

    embeddings = vectorizer.fit_transform(questions)

    vocabulary = build_programming_vocabulary(data)

    return questions, (vectorizer, embeddings), vocabulary


# ============================================================
# Question Similarity
# ============================================================

def calculate_similarity(
    user_question,
    dataset_questions,
    vectorizer,
    embeddings
):
    """
    Calculate TF-IDF semantic similarity between the
    user's question and all dataset questions.
    """

    if not user_question:
        return []

    if not dataset_questions:
        return []

    user_vector = vectorizer.transform(
        [user_question]
    )

    similarities = cosine_similarity(
        user_vector,
        embeddings
    )[0]

    return similarities


# ============================================================
# Fuzzy Matching
# ============================================================

def calculate_fuzzy_score(user_question, dataset_question):
    """
    Calculate multiple fuzzy scores.

    This helps with:
    - spelling mistakes
    - short questions
    - different wording
    - singular/plural differences
    """

    token_score = fuzz.token_set_ratio(
        user_question,
        dataset_question
    ) / 100.0

    sort_score = fuzz.token_sort_ratio(
        user_question,
        dataset_question
    ) / 100.0

    partial_score = fuzz.partial_ratio(
        user_question,
        dataset_question
    ) / 100.0

    ratio_score = fuzz.ratio(
        user_question,
        dataset_question
    ) / 100.0

    # Token similarity is generally strongest for questions
    fuzzy_score = (
        token_score * 0.40
        + sort_score * 0.25
        + partial_score * 0.20
        + ratio_score * 0.15
    )

    return fuzzy_score


# ============================================================
# Search Programming Dataset
# ============================================================

def search_programming(
    question,
    data,
    questions,
    embeddings,
    vocabulary
):
    """
    Search Programming dataset and return the answer
    stored in programming.csv.

    No answer is generated or hard-coded here.
    """

    if data is None or data.empty:
        return None

    if not questions:
        return None

    # --------------------------------------------------------
    # Clean original question
    # --------------------------------------------------------

    original_question = preprocess_text(question)

    if not original_question:
        return None

    # --------------------------------------------------------
    # Correct spelling using dataset vocabulary
    # --------------------------------------------------------

    corrected_question = correct_spelling(
        original_question,
        vocabulary
    )

    # --------------------------------------------------------
    # TF-IDF matching
    # --------------------------------------------------------

    if (
        embeddings is not None
        and isinstance(embeddings, tuple)
    ):
        vectorizer, tfidf_embeddings = embeddings

        semantic_scores = calculate_similarity(
            corrected_question,
            questions,
            vectorizer,
            tfidf_embeddings
        )
    else:
        semantic_scores = [0.0] * len(questions)

    # --------------------------------------------------------
    # Fuzzy matching
    # --------------------------------------------------------

    fuzzy_scores = []

    for dataset_question in questions:
        score = calculate_fuzzy_score(
            corrected_question,
            dataset_question
        )

        fuzzy_scores.append(score)

    # --------------------------------------------------------
    # Combine semantic + fuzzy scores
    # --------------------------------------------------------

    combined_scores = []

    for semantic, fuzzy in zip(
        semantic_scores,
        fuzzy_scores
    ):
        combined = (
            float(semantic) * 0.60
            + float(fuzzy) * 0.40
        )

        combined_scores.append(combined)

    if not combined_scores:
        return None

    # Best result
    best_index = max(
        range(len(combined_scores)),
        key=lambda index: combined_scores[index]
    )

    best_score = combined_scores[best_index]

    # --------------------------------------------------------
    # Adaptive confidence
    # --------------------------------------------------------

    question_word_count = len(
        corrected_question.split()
    )

    # Short questions need stronger confidence
    if question_word_count <= 2:
        required_confidence = 0.52
    elif question_word_count <= 4:
        required_confidence = 0.47
    else:
        required_confidence = MIN_CONFIDENCE

    if best_score < required_confidence:
        return None

    # --------------------------------------------------------
    # Return ONLY the answer from CSV
    # --------------------------------------------------------

    return data.iloc[best_index]["answer"]


# ============================================================
# Programming Question Detection
# ============================================================

def is_programming_question(
    question,
    programming_data
):
    """
    Determine whether a question is likely related to
    Programming by comparing it with the Programming dataset.

    No Programming knowledge is hard-coded here.
    """

    if programming_data is None:
        return False

    if programming_data.empty:
        return False

    user_question = preprocess_text(question)

    if not user_question:
        return False

    vocabulary = build_programming_vocabulary(
        programming_data
    )

    corrected_question = correct_spelling(
        user_question,
        vocabulary
    )

    dataset_questions = [
        preprocess_text(q)
        for q in programming_data["question"].tolist()
    ]

    # --------------------------------------------------------
    # Exact normalized question
    # --------------------------------------------------------

    if corrected_question in dataset_questions:
        return True

    # --------------------------------------------------------
    # Compare against all dataset questions
    # --------------------------------------------------------

    best_score = 0.0

    for dataset_question in dataset_questions:

        fuzzy_score = calculate_fuzzy_score(
            corrected_question,
            dataset_question
        )

        if fuzzy_score > best_score:
            best_score = fuzzy_score

    # --------------------------------------------------------
    # Use dataset size-independent threshold
    # --------------------------------------------------------

    words = corrected_question.split()

    if len(words) <= 2:
        return best_score >= 0.78

    if len(words) <= 4:
        return best_score >= 0.68

    return best_score >= 0.58


# ============================================================
# Main Programming Answer Function
# ============================================================

def answer_programming(question):
    """
    Main Programming function.

    Loads programming.csv, prepares the dataset,
    searches it, and returns the stored answer.
    """

    programming_data = load_programming_data()

    questions, embeddings, vocabulary = (
        prepare_programming_data(
            programming_data
        )
    )

    return search_programming(
        question,
        programming_data,
        questions,
        embeddings,
        vocabulary
    )


# ============================================================
# Safe Helper
# ============================================================

def answer_programming_safe(question):
    """
    Safe wrapper so an unexpected dataset/search error
    does not crash the entire chatbot.
    """

    try:
        return answer_programming(question)

    except FileNotFoundError:
        return None

    except ValueError:
        return None

    except Exception:
        return None