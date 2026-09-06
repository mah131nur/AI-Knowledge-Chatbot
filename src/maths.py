# ============================================================
# 🧮 MATHEMATICS KNOWLEDGE MODULE
# ============================================================
# BASIC / CONCEPTUAL Mathematics only.
#
# This module:
# ✅ Reads data/maths.csv
# ✅ Handles exact questions
# ✅ Handles spelling mistakes
# ✅ Handles short questions
# ✅ Uses TF-IDF similarity
# ✅ Detects whether a question belongs to Maths
#
# It DOES NOT solve numerical Maths problems.
# ============================================================

import os
import re
import difflib

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "maths.csv"
)

MIN_CONFIDENCE = 0.50


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    if text is None:
        return ""

    text = str(text).lower().strip()

    # --------------------------------------------------------
    # Common Mathematics spelling mistakes
    # --------------------------------------------------------

    replacements = {

        # Mathematics
        "mathematcs": "mathematics",
        "mathemetics": "mathematics",
        "mathematic": "mathematics",

        # Number system
        "numbr": "number",
        "naturall": "natural",
        "integar": "integer",
        "positve": "positive",
        "negativ": "negative",
        "evenn": "even",
        "primme": "prime",
        "compostie": "composite",
        "facotr": "factor",
        "multipel": "multiple",

        # Fractions
        "fraciton": "fraction",
        "numrator": "numerator",
        "denomintor": "denominator",
        "impropr": "improper",

        # Percentage / ratio
        "percentge": "percentage",
        "percntage": "percentage",
        "raito": "ratio",
        "proporton": "proportion",

        # Algebra
        "exponant": "exponent",
        "algera": "algebra",
        "varible": "variable",
        "constnt": "constant",
        "expresson": "expression",
        "equaton": "equation",
        "linar": "linear",

        # Geometry
        "geometery": "geometry",
        "trianlge": "triangle",
        "equilaterl": "equilateral",
        "quadrilaterl": "quadrilateral",
        "rectangel": "rectangle",
        "parallelogrm": "parallelogram",
        "circel": "circle",
        "raduis": "radius",
        "diamter": "diameter",
        "circumfrence": "circumference",
        "perimter": "perimeter",
        "volum": "volume",

        # Trigonometry
        "trigonmetery": "trigonometry",
        "trignometry": "trigonometry",
        "sinee": "sine",
        "cosinee": "cosine",
        "tangentt": "tangent",

        # Statistics
        "statstics": "statistics",
        "statitics": "statistics",
        "medain": "median",

        # Probability
        "probablity": "probability",

        # Sets / Functions
        "functon": "function",
        "coordinte": "coordinate"
    }

    for wrong, correct in sorted(
        replacements.items(),
        key=lambda x: len(x[0]),
        reverse=True
    ):

        pattern = (
            r"\b"
            + re.escape(wrong)
            + r"\b"
        )

        text = re.sub(
            pattern,
            correct,
            text
        )

    # --------------------------------------------------------
    # Remove punctuation
    # --------------------------------------------------------

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # --------------------------------------------------------
    # Remove extra spaces
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# LOAD MATHS DATA
# ============================================================

def load_maths_data(
    file_path=DATA_FILE
):

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Maths dataset not found: {file_path}"
        )

    data = pd.read_csv(
        file_path
    )

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    if "question" not in data.columns:

        raise ValueError(
            "Maths CSV must contain a 'question' column."
        )

    if "answer" not in data.columns:

        raise ValueError(
            "Maths CSV must contain an 'answer' column."
        )

    # --------------------------------------------------------
    # Remove incomplete rows
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            "question",
            "answer"
        ]
    ).copy()

    data["question"] = (
        data["question"]
        .astype(str)
    )

    data["answer"] = (
        data["answer"]
        .astype(str)
    )

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    if "category" not in data.columns:

        data["category"] = "basic"

    data["category"] = (
        data["category"]
        .fillna("basic")
        .astype(str)
    )

    # --------------------------------------------------------
    # Normalized questions
    # --------------------------------------------------------

    data["normalized_question"] = (
        data["question"]
        .apply(preprocess_text)
    )

    return data.reset_index(
        drop=True
    )


# ============================================================
# PREPARE MATHS DATA
# ============================================================

def prepare_maths_data(
    data
):

    if data is None or len(data) == 0:

        return (
            [],
            np.array([]),
            None
        )

    questions = (
        data["normalized_question"]
        .tolist()
    )

    vocabulary = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    question_embeddings = (
        vocabulary.fit_transform(
            questions
        )
    )

    return (
        questions,
        question_embeddings,
        vocabulary
    )


# ============================================================
# KEYWORD SCORE
# ============================================================

def keyword_score(
    user_question,
    dataset_question
):

    user_words = set(
        preprocess_text(
            user_question
        ).split()
    )

    data_words = set(
        preprocess_text(
            dataset_question
        ).split()
    )

    if not user_words or not data_words:
        return 0.0

    common = (
        user_words
        .intersection(
            data_words
        )
    )

    return (
        len(common)
        /
        len(user_words)
    )


# ============================================================
# EXACT MATCH
# ============================================================

def exact_match(
    user_question,
    data
):

    normalized = preprocess_text(
        user_question
    )

    if not normalized:
        return None

    for index, row in data.iterrows():

        candidate = (
            row["normalized_question"]
        )

        if normalized == candidate:

            return index

    return None


# ============================================================
# SPELLING CORRECTION
# ============================================================

def correct_spelling(
    question,
    vocabulary
):

    words = question.split()

    if not words:
        return question

    vocabulary_words = set()

    if vocabulary is not None:

        try:

            vocabulary_words = set(
                vocabulary
                .get_feature_names_out()
            )

        except Exception:

            vocabulary_words = set()

    corrected_words = []

    for word in words:

        # ----------------------------------------------------
        # Keep very short words
        # ----------------------------------------------------

        if len(word) <= 2:

            corrected_words.append(
                word
            )

            continue

        # ----------------------------------------------------
        # Already exists
        # ----------------------------------------------------

        if word in vocabulary_words:

            corrected_words.append(
                word
            )

            continue

        # ----------------------------------------------------
        # Fuzzy correction
        # ----------------------------------------------------

        matches = difflib.get_close_matches(
            word,
            vocabulary_words,
            n=1,
            cutoff=0.82
        )

        if matches:

            corrected_words.append(
                matches[0]
            )

        else:

            corrected_words.append(
                word
            )

    return " ".join(
        corrected_words
    )


# ============================================================
# HELPER: TERM DETECTION
# ============================================================
# IMPORTANT:
# We use word boundaries.
#
# This prevents:
#
# "acceleration"
# from matching
# "ratio"
#
# merely because "ratio" appears inside another word.
# ============================================================

def contains_term(
    text,
    term
):

    text = preprocess_text(text)
    term = preprocess_text(term)

    if not text or not term:
        return False

    pattern = (
        r"(?<!\w)"
        + re.escape(term)
        + r"(?!\w)"
    )

    return re.search(
        pattern,
        text
    ) is not None


# ============================================================
# MATHS DOMAIN DETECTION
# ============================================================
#
# This function is extremely important for app.py.
#
# It prevents Physics from stealing Maths questions.
#
# Example:
#
# "What is a fraction?"
# -> Maths
#
# "What is a constant?"
# -> Maths
#
# "What is an equation?"
# -> Maths
#
# "What is an angle?"
# -> Maths
#
# "What is a function?"
# -> Maths
#
# But:
#
# "What is force?"
# -> NOT Maths
#
# "What is acceleration?"
# -> NOT Maths
#
# "Newton first law?"
# -> NOT Maths
# ============================================================

def is_maths_question(
    question,
    data
):

    if not question:
        return False

    normalized = preprocess_text(
        question
    )

    if not normalized:
        return False

    # ========================================================
    # 1. EXACT DATASET MATCH
    # ========================================================
    #
    # This is the strongest check.
    #
    # If the exact question exists in maths.csv,
    # it MUST belong to Maths.
    # ========================================================

    if data is not None and len(data) > 0:

        exact_index = exact_match(
            normalized,
            data
        )

        if exact_index is not None:
            return True

    # ========================================================
    # 2. EXPLICIT MATHS CONTEXT
    # ========================================================

    explicit_math_terms = {

        "math",
        "maths",
        "mathematics",

        "in math",
        "in maths",
        "in mathematics",

        "math question",
        "maths question",
        "mathematics question",

        "mathematical",
        "mathematical concept"
    }

    for term in explicit_math_terms:

        if contains_term(
            normalized,
            term
        ):
            return True

    # ========================================================
    # 3. STRONG MATHS CONCEPTS
    # ========================================================

    strong_math_terms = {

        "natural number",
        "whole number",
        "integer",
        "positive number",
        "negative number",
        "even number",
        "odd number",
        "prime number",
        "composite number",

        "factor",
        "multiple",
        "hcf",
        "lcm",

        "fraction",
        "numerator",
        "denominator",
        "proper fraction",
        "improper fraction",
        "mixed fraction",

        "decimal",
        "percentage",
        "ratio",
        "proportion",

        "exponent",
        "square root",
        "algebra",
        "variable",
        "algebraic expression",
        "linear equation",

        "geometry",
        "point",
        "line segment",
        "triangle",
        "equilateral",
        "isosceles",
        "scalene",
        "quadrilateral",
        "parallelogram",

        "circle",
        "radius",
        "diameter",
        "circumference",
        "perimeter",
        "area",
        "volume",

        "pythagorean",
        "trigonometry",
        "sine",
        "cosine",
        "tangent",

        "statistics",
        "mean",
        "median",
        "mode",
        "probability",

        "set",
        "coordinate geometry"
    }

    for term in strong_math_terms:

        if contains_term(
            normalized,
            term
        ):
            return True

    # ========================================================
    # 4. AMBIGUOUS TERMS WITH MATH CONTEXT
    # ========================================================
    #
    # Words such as:
    #
    # power
    # constant
    # equation
    # angle
    # function
    #
    # also exist in Physics.
    #
    # Therefore we DO NOT classify them as Maths alone.
    #
    # They become Maths when:
    #
    # "power in maths"
    # "constant in mathematics"
    # etc.
    # ========================================================

    ambiguous_math_terms = {
        "power",
        "constant",
        "equation",
        "angle",
        "function",
        "square",
        "mean",
        "line"
    }

    math_context_terms = {
        "math",
        "maths",
        "mathematics",
        "mathematical"
    }

    has_ambiguous = any(
        contains_term(
            normalized,
            term
        )
        for term in ambiguous_math_terms
    )

    has_math_context = any(
        contains_term(
            normalized,
            term
        )
        for term in math_context_terms
    )

    if has_ambiguous and has_math_context:
        return True

    return False


# ============================================================
# MAIN MATHS SEARCH
# ============================================================

def search_maths(
    question,
    data,
    questions,
    question_embeddings,
    vocabulary
):

    try:

        # ====================================================
        # EMPTY DATA
        # ====================================================

        if data is None or len(data) == 0:
            return None

        if not question:
            return None

        # ====================================================
        # NORMALIZE
        # ====================================================

        user_question = preprocess_text(
            question
        )

        if not user_question:
            return None

        # ====================================================
        # EXACT MATCH
        # ====================================================

        exact_index = exact_match(
            user_question,
            data
        )

        if exact_index is not None:

            return data.iloc[
                exact_index
            ]["answer"]

        # ====================================================
        # SPELLING CORRECTION
        # ====================================================

        corrected_question = correct_spelling(
            user_question,
            vocabulary
        )

        if corrected_question != user_question:

            print(
                "✏️ Maths Corrected: "
                f"{corrected_question}"
            )

        user_question = corrected_question

        # ====================================================
        # EXACT MATCH AFTER CORRECTION
        # ====================================================

        exact_index = exact_match(
            user_question,
            data
        )

        if exact_index is not None:

            return data.iloc[
                exact_index
            ]["answer"]

        # ====================================================
        # VECTOR CHECK
        # ====================================================

        if vocabulary is None:
            return None

        user_vector = vocabulary.transform(
            [user_question]
        )

        if user_vector.nnz == 0:
            return None

        # ====================================================
        # TF-IDF SIMILARITY
        # ====================================================

        similarities = cosine_similarity(
            user_vector,
            question_embeddings
        )[0]

        # ====================================================
        # SCORE CANDIDATES
        # ====================================================

        candidate_scores = []

        user_words = set(
            user_question.split()
        )

        for index, row in data.iterrows():

            candidate_question = (
                row["normalized_question"]
            )

            semantic = float(
                similarities[index]
            )

            fuzzy = difflib.SequenceMatcher(
                None,
                user_question,
                candidate_question
            ).ratio()

            keyword = keyword_score(
                user_question,
                candidate_question
            )

            candidate_words = set(
                candidate_question.split()
            )

            common_words = (
                user_words
                .intersection(
                    candidate_words
                )
            )

            common_count = len(
                common_words
            )

            score = (
                semantic * 0.50
                +
                fuzzy * 0.20
                +
                keyword * 0.30
            )

            # Small boost for short questions

            if common_count >= 2:

                score += 0.10

            elif (
                common_count == 1
                and
                len(user_words) <= 3
            ):

                score += 0.05

            candidate_scores.append(
                score
            )

        # ====================================================
        # NO RESULT
        # ====================================================

        if not candidate_scores:
            return None

        # ====================================================
        # BEST RESULT
        # ====================================================

        best_index = int(
            np.argmax(
                candidate_scores
            )
        )

        best_score = float(
            candidate_scores[
                best_index
            ]
        )

        # ====================================================
        # CONFIDENCE CHECK
        # ====================================================

        if best_score < MIN_CONFIDENCE:
            return None

        # ====================================================
        # RETURN ANSWER FROM CSV
        # ====================================================

        return data.iloc[
            best_index
        ]["answer"]

    except Exception as e:

        print(
            f"⚠️ Maths search error: {e}"
        )

        return None


# ============================================================
# STANDALONE HELPER
# ============================================================

def get_maths_answer(
    question
):

    data = load_maths_data()

    (
        questions,
        question_embeddings,
        vocabulary
    ) = prepare_maths_data(
        data
    )

    return search_maths(
        question,
        data,
        questions,
        question_embeddings,
        vocabulary
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print(
        "🧮 Testing Mathematics module..."
    )

    data = load_maths_data()

    (
        questions,
        question_embeddings,
        vocabulary
    ) = prepare_maths_data(
        data
    )

    print(
        f"📚 Loaded {len(data)} Maths questions."
    )

    tests = [

        # Exact Maths
        "what is a fraction?",
        "what is a power in mathematics?",
        "what is a constant?",
        "what is an equation?",
        "what is an angle?",
        "what is a function?",

        # Spelling mistakes
        "what is a trianlge?",
        "what is geometery?",
        "what is probablity?",
        "what is an integar?",

        # Short
        "prime number?",
        "mean?",
        "circle?",

        # Explicit Maths
        "what is a power in maths?",
        "what is a constant in maths?",
        "what is an equation in maths?",
        "what is an angle in maths?"
    ]

    for question in tests:

        print()
        print(
            "Q:",
            question
        )

        print(
            "Is Maths:",
            is_maths_question(
                question,
                data
            )
        )

        answer = search_maths(
            question,
            data,
            questions,
            question_embeddings,
            vocabulary
        )

        print(
            "A:",
            answer
        )