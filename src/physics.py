# ============================================================
# ⚛️ PHYSICS KNOWLEDGE MODULE
# ============================================================

import os
import re
import pandas as pd

from rapidfuzz import fuzz
from rapidfuzz import process


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PHYSICS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "physics.csv"
)


# ============================================================
# LOAD PHYSICS DATA
# ============================================================

def load_physics_data():

    if not os.path.exists(
        PHYSICS_FILE
    ):

        print(
            "⚠️ physics.csv not found:"
        )

        print(
            PHYSICS_FILE
        )

        return pd.DataFrame(
            columns=[
                "topic",
                "definition",
                "formula",
                "variables",
                "answer_unit"
            ]
        )

    try:

        data = pd.read_csv(
            PHYSICS_FILE
        )

        data.columns = [
            str(column)
            .strip()
            .lower()
            for column in data.columns
        ]

        required_columns = [
            "topic",
            "definition",
            "formula",
            "variables",
            "answer_unit"
        ]

        for column in required_columns:

            if column not in data.columns:

                data[column] = ""

        data["topic"] = (
            data["topic"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        data = data[
            data["topic"] != ""
        ].reset_index(
            drop=True
        )

        return data

    except Exception as e:

        print(
            "⚠️ Error loading physics.csv:",
            e
        )

        return pd.DataFrame(
            columns=[
                "topic",
                "definition",
                "formula",
                "variables",
                "answer_unit"
            ]
        )


# ============================================================
# PREPARE PHYSICS DATA
# ============================================================

def prepare_physics_data(
    data=None
):

    if data is None:

        data = load_physics_data()

    return data


# ============================================================
# NORMALIZE
# ============================================================

def normalize_text(text):

    text = str(
        text
    ).lower()

    text = text.replace(
        "’",
        "'"
    )

    text = text.replace(
        "‘",
        "'"
    )

    symbol_map = {

        "ρ": "rho",
        "λ": "lambda",
        "μ": "mu",
        "τ": "tau",
        "ω": "omega",
        "α": "alpha",
        "β": "beta",
        "θ": "theta",
        "Δ": "delta",
        "ε": "epsilon",
        "Φ": "phi",
        "φ": "phi",
        "π": "pi"
    }

    for symbol, replacement in symbol_map.items():

        text = text.replace(
            symbol,
            replacement
        )

    text = text.replace(
        "'",
        ""
    )

    text = text.replace(
        "-",
        " "
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# QUESTION WORDS
# ============================================================

QUESTION_WORDS = {

    "what",
    "is",
    "the",

    "a",
    "an",

    "of",
    "for",

    "give",
    "me",
    "tell",

    "about",

    "define",
    "definition",

    "explain",
    "explanation",

    "please",

    "can",
    "you",

    "does",
    "do",

    "mean",

    "show",

    "find",
    "calculate",

    "formula",
    "formulas",

    "unit",
    "units",

    "called",

    "known",
    "as"
}


# ============================================================
# REMOVE QUESTION WORDS
# ============================================================

def remove_question_words(
    text
):

    words = normalize_text(
        text
    ).split()

    useful_words = []

    for word in words:

        if word not in QUESTION_WORDS:

            useful_words.append(
                word
            )

    return " ".join(
        useful_words
    )


# ============================================================
# GET TOPICS
# ============================================================

def get_physics_topics(
    data
):

    topics = []

    if (
        data is None
        or
        data.empty
        or
        "topic" not in data.columns
    ):

        return topics

    for topic in data[
        "topic"
    ].astype(str):

        topic_clean = normalize_text(
            topic
        )

        if topic_clean:

            topics.append(
                topic_clean
            )

    return topics


# ============================================================
# VOCABULARY
# ============================================================

def get_physics_vocabulary(
    data
):

    vocabulary = set()

    topics = get_physics_topics(
        data
    )

    for topic in topics:

        vocabulary.update(
            topic.split()
        )

    return vocabulary


# ============================================================
# EXACT TOPIC
# ============================================================

def find_exact_topic(
    question,
    data
):

    normalized_question = normalize_text(
        question
    )

    if not normalized_question:

        return None

    question_core = remove_question_words(
        normalized_question
    )

    topics = get_physics_topics(
        data
    )

    # Exact complete topic
    for topic in sorted(
        set(topics),
        key=lambda x: len(
            x.split()
        ),
        reverse=True
    ):

        if question_core == topic:

            return topic

    # Topic inside question
    for topic in sorted(
        set(topics),
        key=lambda x: len(
            x.split()
        ),
        reverse=True
    ):

        pattern = (
            r"\b"
            +
            re.escape(topic)
            +
            r"\b"
        )

        if re.search(
            pattern,
            normalized_question
        ):

            return topic

    return None


# ============================================================
# SPELLING CORRECTION
# ============================================================

def correct_physics_spelling(
    question,
    data
):

    normalized = normalize_text(
        question
    )

    if not normalized:

        return normalized

    topics = get_physics_topics(
        data
    )

    if not topics:

        return normalized

    # Do not modify valid topic
    exact_topic = find_exact_topic(
        normalized,
        data
    )

    if exact_topic is not None:

        return normalized

    core = remove_question_words(
        normalized
    )

    if not core:

        return normalized

    # Phrase match
    phrase_match = process.extractOne(
        core,
        list(set(topics)),
        scorer=fuzz.ratio
    )

    if phrase_match is not None:

        matched_topic = phrase_match[0]
        similarity = phrase_match[1]

        if similarity >= 82:

            original_words = normalized.split()

            prefix = []

            for word in original_words:

                if word in QUESTION_WORDS:

                    prefix.append(
                        word
                    )

                else:

                    break

            if prefix:

                return (
                    " ".join(prefix)
                    +
                    " "
                    +
                    matched_topic
                ).strip()

            return matched_topic

    # Individual correction
    vocabulary = get_physics_vocabulary(
        data
    )

    words = normalized.split()

    corrected_words = []

    for word in words:

        if len(word) <= 3:

            corrected_words.append(
                word
            )

            continue

        if word in QUESTION_WORDS:

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
            list(vocabulary),
            scorer=fuzz.ratio
        )

        if match is None:

            corrected_words.append(
                word
            )

            continue

        matched_word = match[0]
        similarity = match[1]

        if len(word) <= 5:

            threshold = 88

        elif len(word) <= 7:

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
# TOPIC MATCH SCORE
# ============================================================

def topic_match_score(
    question,
    topic
):

    user = normalize_text(
        question
    )

    topic_clean = normalize_text(
        topic
    )

    if not user or not topic_clean:

        return 0.0

    user_core = remove_question_words(
        user
    )

    if not user_core:

        return 0.0

    # Exact
    if user_core == topic_clean:

        return 1.0

    # Topic appears in question
    pattern = (
        r"\b"
        +
        re.escape(topic_clean)
        +
        r"\b"
    )

    if re.search(
        pattern,
        user
    ):

        if len(
            topic_clean.split()
        ) >= 2:

            return 0.99

        return 0.95

    user_words = set(
        user_core.split()
    )

    topic_words = set(
        topic_clean.split()
    )

    if not topic_words:

        return 0.0

    common_words = (
        user_words
        &
        topic_words
    )

    topic_coverage = (
        len(common_words)
        /
        len(topic_words)
    )

    user_coverage = (
        len(common_words)
        /
        max(
            len(user_words),
            1
        )
    )

    fuzzy_score = (
        fuzz.ratio(
            user_core,
            topic_clean
        )
        /
        100
    )

    token_score = (
        fuzz.token_set_ratio(
            user_core,
            topic_clean
        )
        /
        100
    )

    partial_score = (
        fuzz.partial_ratio(
            user_core,
            topic_clean
        )
        /
        100
    )

    if len(topic_words) >= 2:

        score = (

            topic_coverage * 0.30

            +
            user_coverage * 0.20

            +
            fuzzy_score * 0.25

            +
            token_score * 0.20

            +
            partial_score * 0.05
        )

    else:

        score = (

            topic_coverage * 0.30

            +
            user_coverage * 0.15

            +
            fuzzy_score * 0.30

            +
            token_score * 0.20

            +
            partial_score * 0.05
        )

    # --------------------------------------------------------
    # Important protection:
    # If the user has several words and the Physics topic
    # is only one generic word, don't let it steal the query.
    # --------------------------------------------------------

    if (
        len(user_words) >= 2
        and
        len(topic_words) == 1
    ):

        if topic_clean in user_words:

            score *= 0.70

        else:

            score *= 0.35

    return score


# ============================================================
# BEST TOPIC
# ============================================================

def find_best_topic(
    question,
    data
):

    if (
        data is None
        or
        data.empty
        or
        "topic" not in data.columns
    ):

        return None

    # ========================================================
    # EXACT TOPIC
    # ========================================================

    exact_topic = find_exact_topic(
        question,
        data
    )

    if exact_topic is not None:

        for _, row in data.iterrows():

            row_topic = normalize_text(
                row["topic"]
            )

            if row_topic == exact_topic:

                return row

    # ========================================================
    # SCORE TOPICS
    # ========================================================

    best_row = None
    best_score = 0.0
    best_topic_length = 0

    for _, row in data.iterrows():

        topic = str(
            row["topic"]
        ).strip()

        if not topic:

            continue

        score = topic_match_score(
            question,
            topic
        )

        topic_length = len(
            normalize_text(
                topic
            ).split()
        )

        if score > best_score:

            best_score = score
            best_row = row
            best_topic_length = topic_length

        elif (
            score >= best_score - 0.03
            and
            topic_length > best_topic_length
        ):

            best_score = score
            best_row = row
            best_topic_length = topic_length

    if best_row is None:

        return None

    # ========================================================
    # ONE WORD TOPIC
    # ========================================================

    if best_topic_length == 1:

        normalized_question = normalize_text(
            question
        )

        question_core = remove_question_words(
            normalized_question
        )

        topic = normalize_text(
            str(best_row["topic"])
        )

        # Topic must actually be present
        # as a word in the question.
        if topic not in question_core.split():

            return None

        # Strong confidence
        if best_score >= 0.72:

            return best_row

        return None

    # ========================================================
    # MULTI WORD TOPIC
    # ========================================================

    if best_score >= 0.68:

        return best_row

    return None


# ============================================================
# IS PHYSICS QUESTION
# ============================================================

def is_physics_question(
    question,
    data
):

    normalized = normalize_text(
        question
    )

    if not normalized:

        return False

    # Exact topic
    exact_topic = find_exact_topic(
        normalized,
        data
    )

    if exact_topic is not None:

        return True

    # Correct spelling
    corrected = correct_physics_spelling(
        normalized,
        data
    )

    corrected_core = remove_question_words(
        corrected
    )

    topics = get_physics_topics(
        data
    )

    # Exact corrected topic
    for topic in topics:

        if corrected_core == topic:

            return True

    # Best fuzzy score. Compare the corrected question too, so a short
    # spelling mistake such as "currnt" can still be recognized as the
    # Physics topic "current".
    best_score = 0.0

    for topic in topics:

        score = max(
            topic_match_score(normalized, topic),
            topic_match_score(corrected, topic)
        )

        if score > best_score:

            best_score = score

    if best_score >= 0.68:

        return True

    return False


# ============================================================
# FORMAT ANSWER
# ============================================================

def format_physics_answer(
    row
):

    topic = str(
        row.get(
            "topic",
            ""
        )
    ).strip()

    definition = str(
        row.get(
            "definition",
            ""
        )
    ).strip()

    formula = str(
        row.get(
            "formula",
            ""
        )
    ).strip()

    variables = str(
        row.get(
            "variables",
            ""
        )
    ).strip()

    unit = str(
        row.get(
            "answer_unit",
            ""
        )
    ).strip()

    # NaN protection
    values = [
        "topic",
        "definition",
        "formula",
        "variables",
        "answer_unit"
    ]

    for column in values:

        value = str(
            row.get(
                column,
                ""
            )
        ).strip()

        if value.lower() == "nan":

            if column == "topic":
                topic = ""

            elif column == "definition":
                definition = ""

            elif column == "formula":
                formula = ""

            elif column == "variables":
                variables = ""

            elif column == "answer_unit":
                unit = ""

    output = []

    output.append(
        "⚛️ Physics"
    )

    output.append("")

    if topic:

        output.append(
            f"Topic: {topic.title()}"
        )

    if (
        definition
        and
        definition != "—"
    ):

        output.append("")

        output.append(
            f"Definition: {definition}"
        )

    if (
        formula
        and
        formula != "—"
    ):

        output.append("")

        output.append(
            f"Formula: {formula}"
        )

    if (
        variables
        and
        variables != "—"
    ):

        output.append("")

        output.append(
            f"Variables: {variables}"
        )

    if (
        unit
        and
        unit != "—"
    ):

        output.append("")

        output.append(
            f"SI Unit: {unit}"
        )

    return "\n".join(
        output
    )


# ============================================================
# ALL FORMULAS
# ============================================================

def format_all_formulas(
    data
):

    if (
        data is None
        or
        data.empty
    ):

        return None

    output = []

    output.append(
        "⚛️ Physics Formulas"
    )

    output.append("")

    counter = 1

    for _, row in data.iterrows():

        topic = str(
            row.get(
                "topic",
                ""
            )
        ).strip()

        formula = str(
            row.get(
                "formula",
                ""
            )
        ).strip()

        unit = str(
            row.get(
                "answer_unit",
                ""
            )
        ).strip()

        if (
            not topic
            or
            not formula
            or
            formula == "—"
            or
            formula.lower() == "nan"
        ):

            continue

        output.append(
            f"{counter}. {topic.title()}"
        )

        output.append(
            f"Formula: {formula}"
        )

        if (
            unit
            and
            unit != "—"
            and
            unit.lower() != "nan"
        ):

            output.append(
                f"SI Unit: {unit}"
            )

        output.append("")

        counter += 1

    return "\n".join(
        output
    ).strip()


# ============================================================
# ALL TOPICS
# ============================================================

def format_all_topics(
    data
):

    if (
        data is None
        or
        data.empty
        or
        "topic" not in data.columns
    ):

        return None

    output = []

    output.append(
        "⚛️ Physics Topics"
    )

    output.append("")

    seen = set()

    counter = 1

    for topic in data[
        "topic"
    ].astype(str):

        topic_clean = topic.strip()

        key = normalize_text(
            topic_clean
        )

        if (
            key
            and
            key not in seen
        ):

            seen.add(
                key
            )

            output.append(
                f"{counter}. {topic_clean.title()}"
            )

            counter += 1

    return "\n".join(
        output
    )


# ============================================================
# MAIN PHYSICS ANSWER
# ============================================================

def answer_physics(
    question
):

    data = load_physics_data()

    normalized_question = normalize_text(
        question
    )

    if not normalized_question:

        return None

    # ========================================================
    # ALL FORMULAS
    # ========================================================

    formula_requests = {

        "all physics formulas",
        "all physics formula",
        "list all physics formulas",
        "tell me all physics formulas",
        "important physics formulas"
    }

    if normalized_question in formula_requests:

        return format_all_formulas(
            data
        )

    if (
        "all physics formulas"
        in normalized_question
        or
        "list all physics formulas"
        in normalized_question
        or
        "important physics formulas"
        in normalized_question
    ):

        return format_all_formulas(
            data
        )

    # ========================================================
    # ALL TOPICS
    # ========================================================

    if (
        "all physics topics"
        in normalized_question
        or
        "list all physics topics"
        in normalized_question
        or
        "tell me all physics topics"
        in normalized_question
    ):

        return format_all_topics(
            data
        )

    # ========================================================
    # PHYSICS DETECTION
    # ========================================================

    if not is_physics_question(
        normalized_question,
        data
    ):

        return None

    # ========================================================
    # SPELLING CORRECTION
    # ========================================================

    corrected_question = correct_physics_spelling(
        normalized_question,
        data
    )

    if (
        corrected_question
        !=
        normalized_question
    ):

        print(
            f"✏️ Physics corrected: "
            f"{corrected_question}"
        )

    # ========================================================
    # FIND BEST TOPIC
    # ========================================================

    best_row = find_best_topic(
        corrected_question,
        data
    )

    if best_row is None:

        return None

    # ========================================================
    # RETURN DATASET ANSWER
    # ========================================================

    return format_physics_answer(
        best_row
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=========================================="
    )

    print(
        "        PHYSICS MODULE TEST"
    )

    print(
        "=========================================="
    )

    tests = [

        "force",
        "speed",
        "density",
        "displacement",
        "acceleration",
        "velocity",
        "wavelength",

        "vector quantity",
        "luminous intensity",
        "magnetic field",
        "magnetic flux",
        "electric current",
        "electric power",
        "parallel resistance",
        "series resistance",

        "what is force",
        "what is speed",
        "what is displacement",
        "what is acceleration",
        "what is vector quantity",

        "angular displacement",
        "angular acceleration",
        "centripetal acceleration",
        "centripetal force",
        "elastic potential energy",

        "wave amplitude",
        "transverse wave",
        "longitudinal wave",
        "electromagnetic wave",

        "reflection of light",
        "refractive index",
        "critical angle",
        "mirror formula",
        "lens formula",

        "specific heat capacity",
        "thermal expansion",
        "charles law",

        "ohms law",
        "electric current",
        "electric power",

        "electromagnetic induction",
        "faradays law",
        "lenzs law",

        "dimensional analysis",
        "photoelectric effect",
        "mass energy equivalence",

        "newton secod law",
        "lenz law",
        "faradays low",
        "electromagnetic inducton",
        "magnetic flx",
        "paralel resistance",
        "dimensional analisis"
    ]

    for question in tests:

        print()
        print(
            "👤",
            question
        )

        answer = answer_physics(
            question
        )

        print()

        if answer:

            print(
                "🤖",
                answer
            )

        else:

            print(
                "❌ No Physics answer found"
            )