# ============================================================
# 🇵🇰 PAKISTAN KNOWLEDGE MODULE
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

DATA_FILE = os.path.join("data", "pakistan.csv")

MIN_CONFIDENCE = 0.55


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    if text is None:
        return ""

    text = str(text).lower().strip()

    replacements = {

        # ----------------------------------------------------
        # Pakistan
        # ----------------------------------------------------

        "pakiatan": "pakistan",
        "paksitan": "pakistan",
        "pakitsan": "pakistan",
        "pakistn": "pakistan",

        # ----------------------------------------------------
        # Pakistan / common typos
        # ----------------------------------------------------

        "minsiter": "minister",

        "parliments": "parliament",
        "parlimen": "parliament",
        "parlimant": "parliament",
        "parliment": "parliament",
        "parlimanets": "parliament",

        "frst": "first",
        "fst": "first",

        "winng": "winning",

        "ststes": "states",

        "turky": "turkey",
        "turkei": "turkey",
        "trkiye": "turkiye",

        "indai": "india",

        "afganistan": "afghanistan",
        "afgan": "afghanistan",

        "sudi": "saudi",
        "sudia": "saudi",
        "sudai": "saudi",

        "chin": "china",

        "champions": "champions",
        "troph": "trophy",

        # ----------------------------------------------------
        # National
        # ----------------------------------------------------

        "mational": "national",
        "natioanl": "national",
        "natioan": "national",

        # ----------------------------------------------------
        # Official
        # ----------------------------------------------------

        "offical": "official",
        "officail": "official",

        # ----------------------------------------------------
        # Largest / mountain
        # ----------------------------------------------------

        "largset": "largest",
        "lagest": "largest",

        "mountim": "mountain",

        # ----------------------------------------------------
        # Capital
        # ----------------------------------------------------

        "capiatl": "capital",

        # ----------------------------------------------------
        # Punjab
        # ----------------------------------------------------

        "punajb": "punjab",

        # ----------------------------------------------------
        # Mosque
        # ----------------------------------------------------

        "moque": "mosque",
        "moquse": "mosque",
        "mosqe": "mosque",

        # ----------------------------------------------------
        # Jinnah
        # ----------------------------------------------------

        "jannah": "jinnah",
        "jinaah": "jinnah",

        # ----------------------------------------------------
        # Kashmir
        # ----------------------------------------------------

        "kashmir dispte": "kashmir dispute",
        "kashmir disptue": "kashmir dispute",
        "kashmir disput": "kashmir dispute",

        # ----------------------------------------------------
        # Other useful spellings
        # ----------------------------------------------------

        "mohenjo daro": "mohenjo daro",
        "mohenjo-daro": "mohenjo daro",

        "minar-e-pakistan": "minar e pakistan",

        "rohtas fort": "rohtas fort",

        "makli necropolis": "makli necropolis"
    }


    # --------------------------------------------------------
    # Apply replacements safely using word boundaries
    # --------------------------------------------------------

    for wrong, correct in sorted(
        replacements.items(),
        key=lambda x: len(x[0]),
        reverse=True
    ):

        pattern = r"\b" + re.escape(wrong) + r"\b"

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
# LOAD DATA
# ============================================================

def load_pakistan_data(file_path=DATA_FILE):

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Pakistan dataset not found: {file_path}"
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


    if "question" not in data.columns:

        raise ValueError(
            "Pakistan CSV must contain a 'question' column."
        )


    if "answer" not in data.columns:

        raise ValueError(
            "Pakistan CSV must contain an 'answer' column."
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

        data["category"] = "general"


    data["category"] = (
        data["category"]
        .fillna("general")
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
# PREPARE DATA
# ============================================================

def prepare_pakistan_data(data):

    if data is None or len(data) == 0:

        return (
            [],
            np.array([]),
            None
        )


    # --------------------------------------------------------
    # Questions
    # --------------------------------------------------------

    questions = (
        data["normalized_question"]
        .tolist()
    )


    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

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
# ENTITY GROUPS
# ============================================================

ENTITY_GROUPS = {

    # --------------------------------------------------------
    # Pakistan
    # --------------------------------------------------------

    "pakistan": {
        "pakistan",
        "pakistani",
        "pakistanis"
    },


    # --------------------------------------------------------
    # Provinces
    # --------------------------------------------------------

    "punjab": {
        "punjab"
    },

    "sindh": {
        "sindh"
    },

    "khyber pakhtunkhwa": {
        "khyber pakhtunkhwa",
        "kpk"
    },

    "balochistan": {
        "balochistan"
    },

    "gilgit baltistan": {
        "gilgit baltistan",
        "gilgit"
    },


    # --------------------------------------------------------
    # Major cities
    # --------------------------------------------------------

    "islamabad": {
        "islamabad"
    },

    "lahore": {
        "lahore"
    },

    "karachi": {
        "karachi"
    },

    "peshawar": {
        "peshawar"
    },

    "quetta": {
        "quetta"
    },

    "rawalpindi": {
        "rawalpindi"
    },

    "larkana": {
        "larkana"
    },

    "thatta": {
        "thatta"
    },

    "jhelum": {
        "jhelum"
    },


    # --------------------------------------------------------
    # Landmarks
    # --------------------------------------------------------

    "badshahi mosque": {
        "badshahi mosque",
        "badshahi"
    },

    "faisal mosque": {
        "faisal mosque"
    },

    "minar e pakistan": {
        "minar e pakistan",
        "minar pakistan"
    },

    "lahore fort": {
        "lahore fort"
    },

    "pakistan monument": {
        "pakistan monument"
    },

    "lok virsa museum": {
        "lok virsa museum"
    },

    "mohenjo daro": {
        "mohenjo daro",
        "mohenjo-daro"
    },

    "taxila": {
        "taxila"
    },

    "rohtas fort": {
        "rohtas fort"
    },

    "derawar fort": {
        "derawar fort"
    },

    "makli necropolis": {
        "makli necropolis"
    },


    # --------------------------------------------------------
    # Pakistan personalities
    # --------------------------------------------------------

    "muhammad ali jinnah": {
        "muhammad ali jinnah",
        "ali jinnah",
        "jinnah"
    },

    "allama iqbal": {
        "allama iqbal",
        "allama muhammad iqbal",
        "iqbal"
    },

    "liaquat ali khan": {
        "liaquat ali khan"
    },

    "imran khan": {
        "imran khan"
    },

    "fatima jinnah": {
        "fatima jinnah"
    },

    "benazir bhutto": {
        "benazir bhutto"
    },

    "zia ul haq": {
        "zia ul haq"
    },

    "ayub khan": {
        "ayub khan"
    },


    # --------------------------------------------------------
    # Countries
    # --------------------------------------------------------

    "china": {
        "china",
        "chinese"
    },

    "india": {
        "india",
        "indian"
    },

    "iran": {
        "iran",
        "iranian"
    },

    "afghanistan": {
        "afghanistan",
        "afghan"
    },

    "turkey": {
        "turkey",
        "turkish",
        "turkiye"
    },

    "saudi arabia": {
        "saudi arabia",
        "saudi"
    },

    "united states": {
        "united states",
        "usa",
        "america",
        "american"
    },


    # --------------------------------------------------------
    # Important topics
    # --------------------------------------------------------

    "prime minister": {
        "prime minister"
    },

    "world cup": {
        "world cup",
        "worldcup"
    },

    "champions trophy": {
        "champions trophy",
        "champion trophy"
    },

    "parliament": {
        "parliament"
    },

    "election commission": {
        "election commission"
    },

    "federal territories": {
        "federal territories",
        "federal territory"
    }
}


# ============================================================
# GET ENTITIES
# ============================================================

def get_entities(text):

    text = preprocess_text(
        text
    )

    found = set()


    for entity, aliases in ENTITY_GROUPS.items():

        for alias in aliases:

            pattern = (
                r"\b"
                + re.escape(alias)
                + r"\b"
            )


            if re.search(
                pattern,
                text
            ):

                found.add(
                    entity
                )

                break


    return found


# ============================================================
# QUESTION INTENT
# ============================================================

def get_intent(text):

    text = preprocess_text(
        text
    )


    # --------------------------------------------------------
    # Specific intents first
    # --------------------------------------------------------

    if "prime minister" in text:

        return "prime_minister"


    if "world cup" in text:

        return "world_cup"


    if "champions trophy" in text:

        return "champions_trophy"


    if "election commission" in text:

        return "election_commission"


    if "parliament" in text:

        return "parliament"


    if "federal territories" in text:

        return "federal_territories"


    # --------------------------------------------------------
    # Kashmir
    # --------------------------------------------------------

    if (
        "kashmir dispute" in text
        or "kashmir issue" in text
    ):

        return "kashmir_dispute"


    # --------------------------------------------------------
    # Capital
    # --------------------------------------------------------

    if "capital" in text:

        return "capital"


    # --------------------------------------------------------
    # Historical dates
    # --------------------------------------------------------

    if "born" in text:

        return "born"


    if "died" in text or "death" in text:

        return "died"


    # --------------------------------------------------------
    # National symbols
    # --------------------------------------------------------

    if "national anthem" in text:

        return "national_anthem"


    if "anthem" in text:

        return "national_anthem"


    if "flag" in text:

        return "flag"


    if "mosque" in text:

        return "mosque"


    if "flower" in text:

        return "flower"


    if "bird" in text:

        return "bird"


    if "animal" in text:

        return "animal"


    if "sport" in text:

        return "sport"


    if "national poet" in text:

        return "national_poet"


    if "national tree" in text:

        return "national_tree"


    if "national fruit" in text:

        return "national_fruit"


    if "national dress" in text:

        return "national_dress"


    # --------------------------------------------------------
    # Geography
    # --------------------------------------------------------

    if "largest province" in text:

        return "largest_province"


    if "smallest province" in text:

        return "smallest_province"


    if "highest mountain" in text:

        return "highest_mountain"


    if "largest desert" in text:

        return "largest_desert"


    if "longest river" in text:

        return "longest_river"


    # --------------------------------------------------------
    # International relations
    # --------------------------------------------------------

    if (
        "relationship" in text
        or "relation" in text
        or "relations" in text
        or "ties" in text
    ):

        return "relationship"


    return None


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

        candidate = preprocess_text(
            row["question"]
        )


        if normalized == candidate:

            return index


    return None


# ============================================================
# SAFE SPELLING CORRECTION
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
                vocabulary.get_feature_names_out()
            )

        except Exception:

            vocabulary_words = set()


    corrected_words = []


    for word in words:

        if len(word) <= 2:

            corrected_words.append(
                word
            )

            continue


        if word in vocabulary_words:

            corrected_words.append(
                word
            )

            continue


        matches = difflib.get_close_matches(
            word,
            vocabulary_words,
            n=1,
            cutoff=0.88
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
# MAIN SEARCH
# ============================================================

def search_pakistan(
    question,
    data,
    questions,
    question_embeddings,
    vocabulary
):

    try:

        # ====================================================
        # EMPTY INPUT
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
                f"✏️ Pakistan Corrected: "
                f"{corrected_question}"
            )

        else:

            print(
                f"✏️ Pakistan Corrected: "
                f"{user_question}"
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
        # USER ENTITIES
        # ====================================================

        user_entities = get_entities(
            user_question
        )


        # ====================================================
        # USER INTENT
        # ====================================================

        user_intent = get_intent(
            user_question
        )


        # ====================================================
        # SCORE CANDIDATES
        # ====================================================

        candidate_scores = []


        for index, row in data.iterrows():

            candidate_question = row[
                "normalized_question"
            ]


            # ------------------------------------------------
            # Semantic
            # ------------------------------------------------

            semantic = float(
                similarities[index]
            )


            # ------------------------------------------------
            # Fuzzy
            # ------------------------------------------------

            fuzzy = difflib.SequenceMatcher(
                None,
                user_question,
                candidate_question
            ).ratio()


            # ------------------------------------------------
            # Keyword
            # ------------------------------------------------

            keyword = keyword_score(
                user_question,
                candidate_question
            )


            # ------------------------------------------------
            # Candidate entities
            # ------------------------------------------------

            candidate_entities = get_entities(
                candidate_question
            )


            # ------------------------------------------------
            # Candidate intent
            # ------------------------------------------------

            candidate_intent = get_intent(
                candidate_question
            )


            # =================================================
            # ENTITY PROTECTION
            # =================================================

            if user_entities:

                if not user_entities.intersection(
                    candidate_entities
                ):

                    candidate_scores.append(
                        -1.0
                    )

                    continue


            # =================================================
            # SPECIAL PROTECTION FOR RELATIONSHIP QUESTIONS
            # =================================================

            if user_intent == "relationship":

                if candidate_intent != "relationship":

                    candidate_scores.append(
                        -1.0
                    )

                    continue


                relationship_countries = {
                    "china",
                    "india",
                    "iran",
                    "afghanistan",
                    "turkey",
                    "saudi arabia",
                    "united states"
                }


                user_countries = (
                    user_entities
                    .intersection(
                        relationship_countries
                    )
                )


                candidate_countries = (
                    candidate_entities
                    .intersection(
                        relationship_countries
                    )
                )


                if user_countries:

                    if not user_countries.intersection(
                        candidate_countries
                    ):

                        candidate_scores.append(
                            -1.0
                        )

                        continue


            # =================================================
            # INTENT PROTECTION
            # =================================================

            if user_intent is not None:

                if candidate_intent != user_intent:

                    candidate_scores.append(
                        -1.0
                    )

                    continue


            # =================================================
            # BASE SCORE
            # =================================================

            score = (

                semantic * 0.50

                +

                fuzzy * 0.20

                +

                keyword * 0.30
            )


            # =================================================
            # ENTITY BOOST
            # =================================================

            if user_entities:

                if user_entities.intersection(
                    candidate_entities
                ):

                    score += 0.20


            # =================================================
            # INTENT BOOST
            # =================================================

            if (
                user_intent is not None
                and
                candidate_intent == user_intent
            ):

                score += 0.20


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
        # FINAL ENTITY SAFETY
        # ====================================================

        best_question = data.iloc[
            best_index
        ]["question"]


        best_entities = get_entities(
            best_question
        )


        if user_entities:

            if not user_entities.intersection(
                best_entities
            ):

                return None


        # ====================================================
        # FINAL RELATIONSHIP SAFETY
        # ====================================================

        if user_intent == "relationship":

            best_intent = get_intent(
                best_question
            )


            if best_intent != "relationship":

                return None


            relationship_countries = {
                "china",
                "india",
                "iran",
                "afghanistan",
                "turkey",
                "saudi arabia",
                "united states"
            }


            user_countries = (
                user_entities
                .intersection(
                    relationship_countries
                )
            )


            best_countries = (
                best_entities
                .intersection(
                    relationship_countries
                )
            )


            if user_countries:

                if not user_countries.intersection(
                    best_countries
                ):

                    return None


        # ====================================================
        # FINAL INTENT SAFETY
        # ====================================================

        if user_intent is not None:

            best_intent = get_intent(
                best_question
            )


            if best_intent != user_intent:

                return None


        # ====================================================
        # RETURN ANSWER FROM CSV
        # ====================================================

        return data.iloc[
            best_index
        ]["answer"]


    except Exception as e:

        print(
            f"⚠️ Pakistan search error: {e}"
        )

        return None


# ============================================================
# STANDALONE HELPER
# ============================================================

def get_pakistan_answer(question):

    data = load_pakistan_data()


    (
        questions,
        question_embeddings,
        vocabulary
    ) = prepare_pakistan_data(
        data
    )


    return search_pakistan(
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
        "🇵🇰 Testing Pakistan module..."
    )


    data = load_pakistan_data()


    (
        questions,
        question_embeddings,
        vocabulary
    ) = prepare_pakistan_data(
        data
    )


    tests = [

        # ----------------------------------------------------
        # Basic
        # ----------------------------------------------------

        "capital of pakistan",

        "capital of punjab",

        "capital of sindh",

        "capital of kpk",

        "capital of balochistan",

        # ----------------------------------------------------
        # National symbols
        # ----------------------------------------------------

        "national flower",

        "national animal",

        "national bird",

        "national sport",

        "national poet",

        "national anthem",

        "pakistan flag",

        # ----------------------------------------------------
        # Jinnah
        # ----------------------------------------------------

        "who was ali jinnah",

        "ali jinnah born",

        "ali jinnah die",

        # ----------------------------------------------------
        # Iqbal
        # ----------------------------------------------------

        "allama iqbal",

        "allama iqbal born",

        "iqbal day",

        # ----------------------------------------------------
        # Government
        # ----------------------------------------------------

        "first prime minsiter of pakistan",

        "parliments of paksitan",

        "what is the senate of pakistan",

        "what is the election commission of pakistan",

        # ----------------------------------------------------
        # Geography
        # ----------------------------------------------------

        "largest desert of pakistan",

        "highest mountain of pakistan",

        "provinces of pakistan",

        # ----------------------------------------------------
        # Landmarks
        # ----------------------------------------------------

        "what is badshahi mosque",

        "what is faisal mosque",

        "what is minar e pakistan",

        "what is pakistan monument",

        "what is mohenjo daro",

        "what is taxila",

        "what is rohtas fort",

        "what is makli necropolis",

        # ----------------------------------------------------
        # Culture
        # ----------------------------------------------------

        "what is basant",

        "what is truck art in pakistan",

        # ----------------------------------------------------
        # Kashmir
        # ----------------------------------------------------

        "what is the kashmir dispute",

        "kashmir dispute",

        "what is kashmir issue",

        "what is kashmir dispte",

        # ----------------------------------------------------
        # Relationships
        # ----------------------------------------------------

        "relationship between pakistan and china",

        "relationship between pakistan and india",

        "relationship between pakistan and iran",

        "relationship between pakistan and afghanistan",

        "relationship between pakistan and turkey",

        "relationship between pakistan and saudi arabia",

        "relationship between pakistan and the united states",

        # ----------------------------------------------------
        # Short relationship forms
        # ----------------------------------------------------

        "relation with china",

        "relation with india",

        "relation with iran",

        "relation with afghanistan",

        "relation with turkey",

        "relation with saudi arabia",

        # ----------------------------------------------------
        # Typos
        # ----------------------------------------------------

        "relationship between pakitsan and chin",

        "relationship between pakistan and turky",

        "relationship between pakistan and sudi arabia",

        "relationship between pakistan and afganistan"
    ]


    for question in tests:

        print()

        print(
            "Q:",
            question
        )


        answer = search_pakistan(
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