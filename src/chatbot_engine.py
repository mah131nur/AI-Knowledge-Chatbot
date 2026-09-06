# ============================================================
#                  AI KNOWLEDGE CHATBOT
# ============================================================

import os
import json
import re
import difflib
import pandas as pd
from rapidfuzz import fuzz

# ------------------------------------------------------------
# Pakistan
# ------------------------------------------------------------
from src.pakistan import (
    load_pakistan_data,
    prepare_pakistan_data,
    search_pakistan,
)

# ------------------------------------------------------------
# Maths
# ------------------------------------------------------------
from src.maths import (
    load_maths_data,
    prepare_maths_data,
    search_maths,
    is_maths_question,
)

# ------------------------------------------------------------
# Physics
# IMPORTANT:
# Your physics.py uses answer_physics().
# Do NOT import search_physics.
# ------------------------------------------------------------
from src.physics import (
    load_physics_data,
    prepare_physics_data,
    answer_physics,
    is_physics_question,
)

# ------------------------------------------------------------
# Programming
# ------------------------------------------------------------
from src.programming import (
    load_programming_data,
    prepare_programming_data,
    search_programming,
    is_programming_question,
)

try:

    programming_data = (
        load_programming_data()
    )

    (
        programming_questions,
        programming_embeddings,
        programming_vocabulary,
    ) = prepare_programming_data(
        programming_data
    )

    print(
        "✅ Programming dataset loaded."
    )

except Exception as e:

    print(
        f"❌ Programming dataset error: {e}"
    )

    programming_data = pd.DataFrame()

    programming_questions = []

    programming_embeddings = None

    programming_vocabulary = set()


# ------------------------------------------------------------
# General Knowledge
# ------------------------------------------------------------
from src.general_knowledge import (
    load_general_knowledge_data,
    prepare_general_knowledge_data,
    search_general_knowledge,
)


# ============================================================
#                       CONFIGURATION
# ============================================================

PROFILE_FILE = "user_profile.json"


# ============================================================
#                    LOAD USER PROFILE
# ============================================================

def load_user_profile():

    if not os.path.exists(PROFILE_FILE):
        return {}

    try:

        with open(
            PROFILE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, dict):
                return data

    except Exception:
        pass

    return {}


def save_user_profile(profile):

    try:

        with open(
            PROFILE_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                profile,
                file,
                indent=4,
                ensure_ascii=False
            )

    except Exception as e:

        print(
            f"⚠️ Could not save profile: {e}"
        )


user_profile = load_user_profile()


# ============================================================
#                    TEXT HELPERS
# ============================================================

def normalize_text(text):

    text = str(text).lower().strip()

    # Common spelling variations of question words.  These are language
    # normalization only; they do not contain domain knowledge.
    text = re.sub(r"\bwat\s+is\b", "what is", text)
    text = re.sub(r"\bwht\s+is\b", "what is", text)
    text = re.sub(r"\bwhats\b", "what is", text)
    text = re.sub(r"\bwhat\s+is\s+called\b", "what is", text)
    text = re.sub(r"\bdefin\b", "define", text)
    text = re.sub(r"\bexplaine\b", "explain", text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def name_prefix():

    name = user_profile.get("name")

    if name:
        return f", {name}"

    return ""


# ============================================================
#                       GREETINGS
# ============================================================

def is_greeting(text):

    text = normalize_text(text)

    greetings = {
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "helo",
        "good morning",
        "good afternoon",
        "good evening",
        "howdy",
    }

    return text in greetings


# ============================================================
#                        THANKS
# ============================================================

def is_thanks(text):

    text = normalize_text(text)

    thanks_words = {
        "thanks",
        "thank you",
        "thankyou",
        "thx",
        "ty",
    }

    return text in thanks_words


# ============================================================
#                       GOODBYE
# ============================================================

def is_goodbye(text):

    text = normalize_text(text)

    goodbye_words = {
        "bye",
        "goodbye",
        "good bye",
        "exit",
        "quit",
    }

    return text in goodbye_words


# ============================================================
#                 PERSONAL INFORMATION
# ============================================================

def handle_personal_information(question):

    text = normalize_text(question)

    # --------------------------------------------------------
    # WHAT DO YOU KNOW ABOUT ME?
    # Check this BEFORE normal personal statements.
    # --------------------------------------------------------

    if (
        "what do you know about me" in text
        or "what you know about me" in text
        or "tell me what you know about me" in text
    ):

        facts = []

        if user_profile.get("name"):

            facts.append(
                f"your name is {user_profile['name']}"
            )

        if user_profile.get("age"):

            facts.append(
                f"you are {user_profile['age']} years old"
            )

        if user_profile.get("studies"):

            facts.append(
                f"you are studying "
                f"{user_profile['studies']}"
            )

        if user_profile.get("university"):

            facts.append(
                f"you study at "
                f"{user_profile['university']}"
            )

        interests = user_profile.get(
            "interests",
            []
        )

        if interests:

            facts.append(
                "you are interested in "
                + " and ".join(interests)
            )

        if not facts:

            return (
                "I don't know much about you yet. 😊 "
                "You can tell me your name, age, studies, "
                "university, or interests."
            )

        return (
            "Here's what I remember about you: "
            + ", ".join(facts)
            + ". 😊"
        )

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    name_match = re.search(
        r"\bmy name is\s+([a-zA-Z]+)",
        text,
        re.IGNORECASE
    )

    if name_match:

        name = (
            name_match
            .group(1)
            .strip()
            .capitalize()
        )

        user_profile["name"] = name

        save_user_profile(
            user_profile
        )

        return (
            f"Nice to meet you, {name}! 😊 "
            f"I'll remember your name."
        )

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    age_match = re.search(
        r"\bmy age is\s+(\d+)",
        text,
        re.IGNORECASE
    )

    if not age_match:

        age_match = re.search(
            r"\bi am\s+(\d+)\s*"
            r"(?:years?\s*old)?\b",
            text,
            re.IGNORECASE
        )

    if age_match:

        age = int(
            age_match.group(1)
        )

        user_profile["age"] = age

        save_user_profile(
            user_profile
        )

        return (
            f"Got it{name_prefix()}! 😊 "
            f"I'll remember that you're "
            f"{age} years old."
        )

    # --------------------------------------------------------
    # BSCS
    # --------------------------------------------------------

    if re.search(
        r"\b("
        r"i am studying|"
        r"i study|"
        r"i'm studying|"
        r"studying"
        r")\s+(bscs|bs cs)\b",
        text,
        re.IGNORECASE
    ):

        user_profile["studies"] = "BSCS"

        save_user_profile(
            user_profile
        )

        return (
            f"Got it{name_prefix()}! 📚 "
            f"I'll remember that you're studying BSCS."
        )

    # --------------------------------------------------------
    # UNIVERSITY
    # --------------------------------------------------------

    university_match = re.search(
        r"\b(?:"
        r"i study in|"
        r"i am studying in|"
        r"i studied in|"
        r"i study at|"
        r"i am studying at|"
        r"i studied at"
        r")\s+(.+)",
        text,
        re.IGNORECASE
    )

    if university_match:

        university = (
            university_match
            .group(1)
            .strip()
        )

        university = re.sub(
            r"\b("
            r"unicersity|"
            r"univercity|"
            r"universaty"
            r")\b",
            "university",
            university,
            flags=re.IGNORECASE
        )

        if "stmu" in university.lower():

            user_profile["university"] = "STMU"

            save_user_profile(
                user_profile
            )

            return (
                f"Got it{name_prefix()}! 🎓 "
                f"I'll remember that you study at STMU."
            )

    # --------------------------------------------------------
    # CODING / VIDEO EDITING
    # --------------------------------------------------------

    if (
        "like coding" in text
        or "like to do coding" in text
        or "interested in coding" in text
        or "coding and video editing" in text
        or "video editing" in text
    ):

        interests = user_profile.get(
            "interests",
            []
        )

        if not isinstance(
            interests,
            list
        ):

            interests = []

        if "coding" not in interests:

            interests.append(
                "coding"
            )

        if "video editing" not in interests:

            interests.append(
                "video editing"
            )

        user_profile["interests"] = interests

        save_user_profile(
            user_profile
        )

        return (
            f"Nice{name_prefix()}! 😊 "
            f"I'll remember that you're interested "
            f"in coding and video editing."
        )

    return None


# ============================================================
#                  PAKISTAN DETECTION
# ============================================================

def is_pakistan_question(question):

    text = normalize_text(question)

    pakistan_words = [

        "pakistan",
        "pakistani",

        "punjab",
        "sindh",
        "balochistan",

        "kpk",
        "khyber",
        "khyber pakhtunkhwa",

        "islamabad",
        "lahore",
        "karachi",
        "quetta",
        "peshawar",

        "parliament of pakistan",
        "national assembly",
        "senate of pakistan",

        "faisal mosque",
        "badshahi mosque",

        "cpec",
        "kashmir dispute",

        "ali jinnah",
        "jinnah",

        "allama iqbal",

        "markhor",
    ]

    return any(
        word in text
        for word in pakistan_words
    )


# ============================================================
#                 MATHS PROTECTION
# ============================================================

def is_obviously_not_maths(question):

    text = normalize_text(question)

    non_maths_words = [

        # Pakistan
        "pakistan",
        "pakistani",
        "punjab",
        "sindh",
        "balochistan",
        "khyber",
        "islamabad",
        "lahore",
        "karachi",

        # Physics
        "physics",
        "force",
        "velocity",
        "acceleration",
        "mass",
        "density",
        "photon",
        "proton",
        "neutron",
        "newton",
        "work function",
        "ideal gas",
        "voltage",
        "current",
        "resistance",

        # General science
        "biology",
        "human body",
        "dna",
        "cell",
        "solar system",
        "planet",
        "organ",
    ]

    programming_words = [
        "programming", "programming language", "source code",
        "machine code", "algorithm", "pseudocode", "flowchart",
        "compiler", "interpreter", "debugging", "syntax error",
        "runtime error", "array", "linked list", "stack", "queue",
        "binary tree", "binary search tree", "graph traversal",
        "bfs", "dfs", "oop", "class", "object", "pointer",
        "api", "ide", "git", "version control", "unit testing",
        "functional programming", "procedural programming",
        "modular programming", "structured programming", "variable",
        "variables", "data type", "data types", "operator", "operators",
        "function", "functions", "parameter", "parameters",
        "argument", "arguments", "return value", "recursion",
        "input", "output", "expression", "statement", "syntax"
    ]

    if any(
        re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text)
        for word in programming_words
    ):
        return True

    return any(
        word in text
        for word in non_maths_words
    )


# ============================================================
#                  MATHS ROUTING HELPER
# ============================================================

def is_math_routing_question(question):
    """
    Detect Maths questions for routing only.

    Actual Maths answers always come from maths.csv through
    search_maths().
    """

    text = normalize_text(question)

    if not text:
        return False

    # Explicit domain context has highest priority.
    if re.search(r"\b(math|maths|mathematics|mathematical)\b", text):
        return True

    # Do not claim concepts that clearly belong to another domain.
    programming_context = [
        "programming", "programming language", "computer programming",
        "in programming", "source code", "machine code", "compiler",
        "interpreter", "debugging", "syntax error", "runtime error",
        "algorithm", "pseudocode", "flowchart", "array", "linked list",
        "stack", "queue", "binary tree", "binary search tree", "graph",
        "bfs", "dfs", "oop", "object oriented programming", "class",
        "object", "encapsulation", "inheritance", "polymorphism",
        "abstraction", "pointer", "api", "ide", "git", "version control",
        "modular programming", "structured programming",
        "procedural programming", "functional programming"
    ]

    physics_context = [
        "physics", "force", "mass", "density", "velocity",
        "acceleration", "momentum", "photon", "proton", "neutron",
        "newton", "work function", "ideal gas", "voltage", "current",
        "resistance", "power", "energy", "uncertainty"
    ]

    if any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text)
           for term in programming_context):
        return False

    if any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text)
           for term in physics_context):
        return False

    # Strong Maths concepts, including plural forms.
    maths_terms = [
        "number", "numbers",
        "natural number", "natural numbers",
        "whole number", "whole numbers",
        "integer", "integers",
        "positive number", "positive numbers",
        "negative number", "negative numbers",
        "even number", "even numbers",
        "odd number", "odd numbers",
        "prime number", "prime numbers",
        "composite number", "composite numbers",
        "factor", "factors", "multiple", "multiples",
        "hcf", "lcm", "fraction", "fractions",
        "numerator", "denominator", "decimal", "decimals",
        "percentage", "percentages", "ratio", "ratios",
        "proportion", "proportions", "exponent", "exponents",
        "square root", "square roots", "algebra",
        "algebraic expression", "algebraic expressions",
        "linear equation", "linear equations", "geometry",
        "triangle", "triangles", "rectangle", "rectangles",
        "square", "squares", "circle", "circles", "radius", "radii",
        "diameter", "diameters", "circumference", "perimeter",
        "perimeters", "pythagorean", "trigonometry", "sine", "cosine",
        "tangent", "statistics", "mean", "median", "mode",
        "probability", "set", "sets", "coordinate geometry",
        "function in maths", "functions in maths",
        "function in math", "functions in math"
    ]

    if any(
        re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text)
        for term in maths_terms
    ):
        return True

    # Small spelling mistakes such as "rectange" should still route to
    # Maths. This is routing only; the actual answer still comes from
    # maths.csv.
    for token in text.split():
        token = re.sub(r"[^a-z0-9]+", "", token)
        if len(token) < 5:
            continue
        for term in maths_terms:
            if " " in term:
                continue
            if len(term) < 5:
                continue
            if difflib.SequenceMatcher(None, token, term).ratio() >= 0.84:
                return True

    return False


# ============================================================
#              PROGRAMMING ROUTING HELPER
# ============================================================

def is_programming_routing_question(question):
    """
    Detect Programming questions for routing only.

    Actual answers always come from programming.csv through
    search_programming().
    """

    text = normalize_text(question)

    if not text:
        return False

    # Explicit Maths context must stay in Maths.
    if re.search(r"\b(math|maths|mathematics|mathematical)\b", text):
        return False

    # Explicit Physics context and Physics-specific concepts must stay in Physics.
    physics_terms = [
        "physics", "force", "mass", "density", "velocity",
        "acceleration", "momentum", "photon", "proton", "neutron",
        "newton", "work function", "ideal gas", "voltage", "current",
        "resistance", "power", "energy", "uncertainty", "equation of motion"
    ]

    if any(
        re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text)
        for term in physics_terms
    ):
        return False

    # Explicit Programming context.
    programming_context = [
        "programming", "programming question", "programming concept",
        "programming language", "computer programming", "in programming",
        "in python", "in c++", "in cpp", "in java", "in javascript"
    ]

    if any(
        re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text)
        for term in programming_context
    ):
        return True

    # Strong Programming concepts. These are routing signals only.
    programming_terms = [
        "program", "programs", "source code", "machine code",
        "algorithm", "algorithms", "pseudocode", "flowchart",
        "debugging", "syntax error", "logical error", "runtime error",
        "compiler", "interpreter", "data type", "data types",
        "operator", "operators", "conditional statement", "if statement",
        "if else", "nested if", "loop", "loops", "for loop",
        "while loop", "do while", "infinite loop", "break statement",
        "continue statement", "function", "functions", "parameter",
        "parameters", "argument", "arguments", "return value",
        "recursion", "recursive", "array", "arrays", "index in an array",
        "multidimensional array", "linked list", "linked lists", "node",
        "nodes", "singly linked list", "doubly linked list", "stack",
        "stacks", "lifo", "push", "pop", "queue", "queues", "fifo",
        "enqueue", "dequeue", "circular queue", "priority queue",
        "tree traversal", "binary tree", "binary search tree", "bst",
        "inorder", "preorder", "postorder", "graph traversal", "bfs", "dfs",
        "linear search", "binary search", "sorting", "bubble sort",
        "selection sort", "insertion sort", "merge sort", "quick sort",
        "time complexity", "space complexity", "big o", "o(1)", "o(n)",
        "o(log n)", "o(n2)", "object oriented programming", "oop",
        "class", "classes", "object", "objects", "encapsulation",
        "inheritance", "polymorphism", "abstraction", "constructor",
        "destructor", "method overloading", "method overriding", "pointer",
        "pointers", "reference", "dynamic memory allocation",
        "memory management", "memory leak", "exception", "exception handling",
        "module", "modules", "library", "libraries", "api", "ide",
        "version control", "git", "repository", "bug", "software testing",
        "unit testing", "input", "output", "expression", "statement",
        "syntax", "comment", "modular programming", "structured programming",
        "procedural programming", "functional programming", "data structure",
        "data structures", "variable", "variables", "constant", "constants",
        "floating point", "floating-point"
    ]

    if any(
        re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text)
        for term in programming_terms
    ):
        return True

    # Tolerate small spelling mistakes in common Programming concepts.
    for token in text.split():
        token = re.sub(r"[^a-z0-9]+", "", token)
        if len(token) < 5:
            continue
        for term in programming_terms:
            if " " in term or len(term) < 5:
                continue
            if difflib.SequenceMatcher(None, token, term).ratio() >= 0.84:
                return True

    return False


# ============================================================
#                  PHYSICS PROTECTION
# ============================================================

def is_safe_physics_question(question):

    text = normalize_text(question)

    # Explicit Programming context must win over Physics terms.
    # For example, "algorithm's efficiency in programming" contains
    # the Physics concept "efficiency", but the user is asking about
    # Programming.
    programming_context = [
        "programming", "in programming", "programming language",
        "computer programming", "source code", "machine code",
        "algorithm", "algorithms", "pseudocode", "flowchart",
        "compiler", "interpreter", "debugging", "syntax error",
        "array", "arrays", "linked list", "stack", "queue",
        "binary tree", "binary search tree", "graph", "bfs", "dfs",
        "oop", "object oriented programming", "class", "object",
        "api", "ide", "git", "version control", "software testing",
        "unit testing", "data structure", "data structures",
        "variable", "variables", "function", "functions",
        "module", "modules", "library", "libraries", "statement",
        "input", "output", "recursion", "sorting", "searching",
        "time complexity", "space complexity", "big o",
        "program", "programs"
    ]

    if any(
        re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text)
        for term in programming_context
    ):
        return False

    # A bare "function/functions" is ambiguous. Do not let the
    # Physics fuzzy matcher turn it into "work function" unless
    # the user explicitly gives Physics context.
    if re.search(r"\bfunctions?\b", text):
        physics_context = [
            "physics", "work function", "photoelectric",
            "electron", "photon", "energy", "quantum"
        ]
        if not any(
            re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text)
            for term in physics_context
        ):
            return False

    # --------------------------------------------------------
    # Maths phrases should NEVER enter Physics
    # --------------------------------------------------------

    maths_phrases = [

        "in maths",
        "in math",
        "mathematics",
        "mathematical",

        "algebra",
        "geometry",

        "mean",
        "median",
        "mode",

        "denominator",
        "numerator",

        "triangle",
        "right angle triangle",
        "right angled triangle",

        "rectangle",
        "square",

        "natural number",
        "whole number",
        "negative number",

        "function in maths",
        "functions in maths",

        "function in math",
        "functions in math",
    ]

    if any(
        phrase in text
        for phrase in maths_phrases
    ):

        return False

    # --------------------------------------------------------
    # Pakistan must NEVER enter Physics
    # --------------------------------------------------------

    if is_pakistan_question(
        question
    ):

        return False

    # --------------------------------------------------------
    # Let physics.py decide
    # --------------------------------------------------------

    try:

        return is_physics_question(
            question,
            physics_data
        )

    except Exception:

        return False


# ============================================================
#             QUESTION-FORM NORMALIZATION
# ============================================================

def build_question_variants(question):
    """Return clean concept-first variants for CSV matching.

    Question words such as "what is", "define", and "explain" are
    intentionally removed first.  This makes short forms such as
    "define API" and "explain API" behave like "what is API".
    """
    text = normalize_text(question)
    text = re.sub(r"[?!.]+$", "", text).strip()
    if not text:
        return []

    patterns = [
        r"^(?:can you|could you|please)\s+",
        r"^(?:what is|what are)\s+(?:an?|the)\s+",
        r"^(?:what is|what are)\s+",
        r"^(?:define|explain|describe)\s+(?:an?|the)\s+",
        r"^(?:define|explain|describe)\s+",
        r"^(?:tell me about|tell me|meaning of)\s+(?:an?|the)\s+",
        r"^(?:tell me about|tell me|meaning of)\s+",
    ]

    core = text
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            match = re.match(pattern, core)
            if match:
                core = core[match.end():].strip()
                changed = True
                break

    for suffix in (
        " in mathematics", " in programming", " in physics",
        " in general knowledge", " in maths", " in math"
    ):
        if core.endswith(suffix):
            core = core[:-len(suffix)].strip()
            break

    variants = []
    def add(value):
        value = re.sub(r"\s+", " ", value).strip(" ?!.,")
        if value and value not in variants:
            variants.append(value)

    # Concept-first is critical: it prevents "define API" from matching
    # an unrelated long question before the actual API row is considered.
    add(core)

    words = core.split()
    if words:
        last = words[-1]
        if last.endswith("ies") and len(last) > 3:
            words[-1] = last[:-3] + "y"
            add(" ".join(words))
        elif last.endswith("s") and not last.endswith("ss") and len(last) > 3:
            words[-1] = last[:-1]
            add(" ".join(words))

    add("what is " + core)
    add("what are " + core)
    add(text)  # original question is deliberately last
    return variants

def correct_question_spelling(question, vocabulary):
    """Correct small spelling mistakes using a module's vocabulary.

    Supports both Programming's set vocabulary and Maths/Pakistan's
    TfidfVectorizer vocabulary.
    """
    if vocabulary is None:
        return normalize_text(question)

    try:
        if hasattr(vocabulary, "get_feature_names_out"):
            raw_vocab = vocabulary.get_feature_names_out()
        else:
            raw_vocab = vocabulary
        vocab = {
            re.sub(r"[^a-z0-9]+", "", str(word).lower())
            for word in raw_vocab
        }
    except Exception:
        return normalize_text(question)

    vocab = {word for word in vocab if len(word) >= 4}
    if not vocab:
        return normalize_text(question)

    stop_words = {
        "what", "whatis", "is", "are", "was", "were", "the", "a", "an",
        "in", "of", "to", "for", "and", "or", "on", "with", "define",
        "explain", "describe", "tell", "me", "about", "how", "why",
        "does", "do", "can", "you", "please", "meaning"
    }

    corrected = []
    for token in normalize_text(question).split():
        clean = re.sub(r"[^a-z0-9]+", "", token)
        if len(clean) < 4 or clean in stop_words or clean in vocab:
            corrected.append(token)
            continue

        best = clean
        best_score = 0.0
        for word in vocab:
            if abs(len(clean) - len(word)) > 2:
                continue
            score = difflib.SequenceMatcher(None, clean, word).ratio()
            if score > best_score:
                best_score = score
                best = word

        threshold = 0.86 if len(clean) <= 5 else 0.80
        corrected.append(best if best_score >= threshold else token)

    return " ".join(corrected)


def search_programming_with_question_forms(question):
    """Search programming.csv using equivalent question forms."""

    corrected = correct_question_spelling(
        question,
        programming_vocabulary
    )

    candidate_questions = []
    if corrected != normalize_text(question):
        candidate_questions.append(corrected)
    candidate_questions.append(question)

    variants = []
    for candidate in candidate_questions:
        for variant in build_question_variants(candidate):
            if variant not in variants:
                variants.append(variant)

    for variant in variants:
        try:
            answer = search_programming(
                variant,
                programming_data,
                programming_questions,
                programming_embeddings,
                programming_vocabulary,
            )
        except Exception:
            answer = None

        if answer:
            return answer

    return None


def search_maths_with_question_forms(question):
    """Search maths.csv using equivalent question forms."""

    corrected = correct_question_spelling(
        question,
        maths_vocabulary
    )

    candidate_questions = []
    if corrected != normalize_text(question):
        candidate_questions.append(corrected)
    candidate_questions.append(question)

    variants = []
    for candidate in candidate_questions:
        for variant in build_question_variants(candidate):
            if variant not in variants:
                variants.append(variant)

    for variant in variants:
        try:
            answer = search_maths(
                variant,
                maths_data,
                maths_questions,
                maths_embeddings,
                maths_vocabulary,
            )
        except Exception:
            answer = None

        if answer:
            return answer

    return None


def answer_physics_with_question_forms(question):
    """Answer Physics questions using equivalent question forms."""

    for variant in build_question_variants(question):
        try:
            answer = answer_physics(variant)
        except Exception:
            answer = None

        if answer:
            return answer

    return None


# ============================================================
#               CASUAL CONVERSATION DETECTION
# ============================================================

def looks_like_casual_conversation(question):

    text = normalize_text(question)

    # --------------------------------------------------------
    # PERSONAL / EMOTIONAL
    # --------------------------------------------------------

    casual_patterns = [

        r"\bi am sad\b",
        r"\bi'm sad\b",

        r"\bi am happy\b",
        r"\bi'm happy\b",

        r"\bi am worried\b",
        r"\bi'm worried\b",

        r"\bi am stressed\b",
        r"\bi'm stressed\b",

        r"\bi am stress\b",

        r"\bi am upset\b",
        r"\bi'm upset\b",

        r"\bi am angry\b",
        r"\bi'm angry\b",

        r"\bi feel\b",

        r"\bi like someone\b",
        r"\bi like somebody\b",

        r"\bi love someone\b",
        r"\bi love somebody\b",

        r"\bi am in love\b",
        r"\bi'm in love\b",

        r"\bi got into a fight\b",
        r"\bi got in a fight\b",

        r"\bi fought with\b",

        r"\bi don't feel good\b",

        r"\bi feel better\b",

        r"\bi feel shy\b",

        r"\bhe flirts with me\b",
        r"\bshe flirts with me\b",

        r"\bi am confused\b",
        r"\bi'm confused\b",

        r"\bi am scared\b",
        r"\bi'm scared\b",

        r"\bi am tired\b",
        r"\bi'm tired\b",

        r"\bi am bored\b",
        r"\bi'm bored\b",

        r"\bi am excited\b",
        r"\bi'm excited\b",

        r"\bi am lonely\b",
        r"\bi'm lonely\b",

        r"\bi hate\b",

        r"\bi don't know what to do\b",

        r"\bhow can i improve myself\b",
        r"\bhow can i improve\b",

        r"\bi think my emotions\b",

        r"\bi want to talk\b",
        r"\bi want to talk to you\b",

        r"\bcan i talk to you\b",
        r"\bcan we talk\b",

        r"\bi need advice\b",
        r"\bgive me advice\b",

        r"\bi need help\b",

        r"\bi am making my chatbot\b",
        r"\bi'm making my chatbot\b",

        r"\bi am making a chatbot\b",
        r"\bi'm making a chatbot\b",

        r"\bi am studying\b",
        r"\bi'm studying\b",

        r"\bi have to make\b",

        r"\bi am overwhelmed\b",
        r"\bi'm overwhelmed\b",
    ]

    for pattern in casual_patterns:

        if re.search(
            pattern,
            text
        ):

            return True

    # --------------------------------------------------------
    # CONVERSATIONAL QUESTIONS
    # --------------------------------------------------------

    conversational_patterns = [

        r"\bhow are you\b",
        r"\bhow are u\b",

        r"\bwhat are you doing\b",

        r"\bwhat do you think\b",

        r"\btell me something\b",

        r"\bwhat should i do\b",

        r"\bcan you help me\b",

        r"\bplease help me\b",

        r"\bhelp me with\b",

        r"\bmake me a\b",

        r"\bhow can i\b",
    ]

    for pattern in conversational_patterns:

        if re.search(
            pattern,
            text
        ):

            # Important:
            # Do NOT classify obvious knowledge
            # questions as casual.

            if (
                text.startswith("what is ")
                or text.startswith("who is ")
                or text.startswith("who was ")
                or text.startswith("where is ")
                or text.startswith("when was ")
            ):

                return False

            return True

    return False


# ============================================================
#               FAST CASUAL RESPONSES
# ============================================================
# These do NOT use external language model.
# This makes very common messages instant.

def instant_casual_response(question):

    text = normalize_text(question)

    name = name_prefix()

    # --------------------------------------------------------
    # HELLO
    # --------------------------------------------------------

    if is_greeting(text):

        return (
            f"Hello{name}! 👋 "
            f"How are you doing today?"
        )

    # --------------------------------------------------------
    # THANKS
    # --------------------------------------------------------

    if is_thanks(text):

        return (
            f"You're very welcome{name}! 💛"
        )

    # --------------------------------------------------------
    # HAPPY
    # --------------------------------------------------------

    if re.search(
        r"\b(i am happy|i'm happy)\b",
        text
    ):

        return (
            f"I'm glad to hear that{name}! 😊 "
            f"What's making you happy today?"
        )

    # --------------------------------------------------------
    # SAD
    # --------------------------------------------------------

    if re.search(
        r"\b(i am sad|i'm sad)\b",
        text
    ):

        return (
            f"I'm sorry you're feeling sad{name}. 💛 "
            f"I'm here if you want to talk about it."
        )

    # --------------------------------------------------------
    # WORRIED
    # --------------------------------------------------------

    if re.search(
        r"\b(i am worried|i'm worried)\b",
        text
    ):

        return (
            f"I understand{name}. 💛 "
            f"What's worrying you?"
        )

    # --------------------------------------------------------
    # STRESSED
    # --------------------------------------------------------

    if re.search(
        r"\b(i am stressed|i'm stressed|"
        r"i am stress)\b",
        text
    ):

        return (
            f"That sounds stressful{name}. 💛 "
            f"Let's take it one thing at a time."
        )

    # --------------------------------------------------------
    # TIRED
    # --------------------------------------------------------

    if re.search(
        r"\b(i am tired|i'm tired)\b",
        text
    ):

        return (
            f"You sound tired{name}. 💛 "
            f"Maybe take a short break and breathe."
        )

    # --------------------------------------------------------
    # WANT TO TALK
    # --------------------------------------------------------

    if (
        "i want to talk" in text
        or "can i talk to you" in text
        or "can we talk" in text
    ):

        return (
            f"Of course{name}. 💛 "
            f"I'm here to listen."
        )

    return None


# ============================================================
#                  CASUAL FALLBACK
# ============================================================

def casual_fallback():
    name = name_prefix()
    return (
        f"I can help with Pakistan, Maths, Physics, Programming, "
        f"and General Knowledge{name}. "
        f"Please ask me a question from one of these areas."
    )

# ============================================================
#                    LOAD DATASETS
# ============================================================

print(
    "\nLoading knowledge datasets..."
)


# ------------------------------------------------------------
# Pakistan
# ------------------------------------------------------------

try:

    pakistan_data = (
        load_pakistan_data()
    )

    (
        pakistan_questions,
        pakistan_embeddings,
        pakistan_vocabulary,
    ) = prepare_pakistan_data(
        pakistan_data
    )

    print(
        "✅ Pakistan dataset loaded."
    )

except Exception as e:

    print(
        f"❌ Pakistan dataset error: {e}"
    )

    pakistan_data = []

    pakistan_questions = []

    pakistan_embeddings = None

    pakistan_vocabulary = set()


# ------------------------------------------------------------
# Maths
# ------------------------------------------------------------

try:

    maths_data = (
        load_maths_data()
    )

    (
        maths_questions,
        maths_embeddings,
        maths_vocabulary,
    ) = prepare_maths_data(
        maths_data
    )

    print(
        "✅ Maths dataset loaded."
    )

except Exception as e:

    print(
        f"❌ Maths dataset error: {e}"
    )

    maths_data = []

    maths_questions = []

    maths_embeddings = None

    maths_vocabulary = set()


# ------------------------------------------------------------
# Physics
# ------------------------------------------------------------

try:

    physics_data = (
        load_physics_data()
    )

    physics_data = (
        prepare_physics_data(
            physics_data
        )
    )

    print(
        "✅ Physics dataset loaded."
    )

except Exception as e:

    print(
        f"❌ Physics dataset error: {e}"
    )

    physics_data = []


# ------------------------------------------------------------
# General Knowledge
# ------------------------------------------------------------

try:

    general_knowledge_data = (
        load_general_knowledge_data()
    )

    (
        general_questions,
        general_embeddings,
        general_vocabulary,
    ) = prepare_general_knowledge_data(
        general_knowledge_data
    )

    print(
        "✅ General Knowledge dataset loaded."
    )

except Exception as e:

    print(
        f"❌ General Knowledge dataset error: {e}"
    )

    general_knowledge_data = []

    general_questions = []

    general_embeddings = None

    general_vocabulary = set()



# ============================================================
#                 DATA-DRIVEN DOMAIN ROUTER
# ============================================================

QUESTION_PREFIX_WORDS = {
    "what", "is", "are", "a", "an", "the", "of", "for", "give",
    "me", "tell", "about", "who", "was", "were", "where", "which",
    "called", "does", "do", "when", "how", "many", "can", "with",
    "and", "or", "did", "has", "have", "please", "explain", "define",
    "definition", "describe", "meaning", "show", "known", "as"
}


def router_core(text):
    """Normalize a question to its knowledge-bearing concept."""
    variants = build_question_variants(text)
    if not variants:
        return ""
    return variants[0]


def build_router_candidates():
    """Build routing candidates directly from the five CSV datasets.

    No Physics/Maths/Programming answer or concept list is hard-coded here.
    The CSV files themselves define what each domain knows.
    """
    candidates = {}

    def clean(value):
        value = normalize_text(value)
        value = re.sub(r"[^a-z0-9\s]", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def question_core(value):
        value = clean(value)
        variants = build_question_variants(value)
        return variants[0] if variants else value

    # Pakistan
    pakistan_items = []
    if hasattr(pakistan_data, "iterrows"):
        for _, row in pakistan_data.iterrows():
            q = str(row.get("question", ""))
            c = question_core(q)
            if c:
                pakistan_items.append(c)
    candidates["pakistan"] = sorted(set(pakistan_items), key=len, reverse=True)

    # Maths
    maths_items = []
    if hasattr(maths_data, "iterrows"):
        for _, row in maths_data.iterrows():
            q = str(row.get("question", ""))
            c = question_core(q)
            if c:
                maths_items.append(c)
    candidates["maths"] = sorted(set(maths_items), key=len, reverse=True)

    # Programming
    programming_items = []
    if hasattr(programming_data, "iterrows"):
        for _, row in programming_data.iterrows():
            q = str(row.get("question", ""))
            c = question_core(q)
            if c:
                programming_items.append(c)
    candidates["programming"] = sorted(set(programming_items), key=len, reverse=True)

    # General Knowledge
    general_items = []
    for row in general_knowledge_data if isinstance(general_knowledge_data, list) else []:
        c = question_core(str(row.get("question", "")))
        if c:
            general_items.append(c)
    candidates["general_knowledge"] = sorted(set(general_items), key=len, reverse=True)

    # Physics uses topic instead of question.
    physics_items = []
    if hasattr(physics_data, "iterrows"):
        for _, row in physics_data.iterrows():
            c = clean(str(row.get("topic", "")))
            if c:
                physics_items.append(c)
    candidates["physics"] = sorted(set(physics_items), key=len, reverse=True)

    return candidates


def correct_router_core(core, candidates):
    """Correct spelling only against concepts present in the candidate domain."""
    words = core.split()
    if not words:
        return core

    vocabulary = set()
    for candidate in candidates:
        vocabulary.update(candidate.split())

    corrected = []
    for word in words:
        if len(word) <= 3 or word in vocabulary:
            corrected.append(word)
            continue

        best_word = word
        best_score = 0
        for vocab_word in vocabulary:
            if abs(len(word) - len(vocab_word)) > 2:
                continue
            score = fuzz.ratio(word, vocab_word)
            if score > best_score:
                best_score = score
                best_word = vocab_word

        threshold = 88
        corrected.append(best_word if best_score >= threshold else word)

    return " ".join(corrected)


def score_router_candidate(core, candidate):
    """Score one concept using exact, token, fuzzy and containment evidence."""
    if not core or not candidate:
        return 0.0

    if core == candidate:
        return 1.0

    # A corrected short concept such as "current" should match a CSV
    # concept such as "electric current" without needing the word
    # "electric" in the user's question.
    if len(core.split()) == 1 and core in candidate.split():
        return 0.90

    if re.search(r"(?<!\w)" + re.escape(candidate) + r"(?!\w)", core):
        return 0.97

    token_score = fuzz.token_set_ratio(core, candidate) / 100.0
    ratio_score = fuzz.ratio(core, candidate) / 100.0
    partial_score = fuzz.partial_ratio(core, candidate) / 100.0

    core_words = set(core.split())
    candidate_words = set(candidate.split())
    overlap = len(core_words & candidate_words) / max(1, len(candidate_words))

    return (
        token_score * 0.45
        + ratio_score * 0.30
        + partial_score * 0.10
        + overlap * 0.15
    )


def explicit_domain_boost(question):
    """Only explicit domain names get a small routing boost."""
    text = normalize_text(question)
    boost = {d: 0.0 for d in (
        "pakistan", "physics", "maths", "programming", "general_knowledge"
    )}

    if re.search(r"\b(pakistan|pakistani)\b", text):
        boost["pakistan"] = 0.15
    if re.search(r"\b(physics|physical science)\b", text):
        boost["physics"] = 0.15
    if re.search(r"\b(math|maths|mathematics)\b", text):
        boost["maths"] = 0.15
    if re.search(r"\b(programming|programming language|coding)\b", text):
        boost["programming"] = 0.15
    if re.search(r"\b(general knowledge|general knowledge question)\b", text):
        boost["general_knowledge"] = 0.15

    return boost


def route_knowledge_question(question):
    """Return the single most likely domain, or None if confidence is low."""
    core = router_core(question)
    if not core:
        return None

    boosts = explicit_domain_boost(question)
    domain_scores = {}

    for domain, domain_candidates in ROUTER_CANDIDATES.items():
        corrected_core = correct_router_core(core, domain_candidates)
        best = 0.0

        # Compare both original and corrected forms. This preserves valid
        # words while still recovering from typos such as "rectange".
        for candidate in domain_candidates:
            best = max(
                best,
                score_router_candidate(core, candidate),
                score_router_candidate(corrected_core, candidate),
            )

        domain_scores[domain] = best + boosts[domain]

    ranked = sorted(
        domain_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    if not ranked:
        return None

    best_domain, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    # Short concepts require stronger evidence. This is important for
    # ambiguous words such as "function", "variable", and "power".
    word_count = len(core.split())
    if word_count <= 2:
        minimum = 0.78
    elif word_count <= 4:
        minimum = 0.64
    else:
        minimum = 0.56

    if best_score < minimum:
        return None

    # If two domains are genuinely close, use explicit context if present;
    # otherwise prefer the established conceptual order rather than allowing
    # a random module to win.
    if second_score >= best_score - 0.025:
        priority = {
            "pakistan": 0,
            "physics": 1,
            "maths": 2,
            "programming": 3,
            "general_knowledge": 4,
        }
        close = [d for d, s in ranked if s >= best_score - 0.025]
        best_domain = min(close, key=lambda d: priority[d])

    return best_domain


# ============================================================
#                       STARTUP
# ============================================================

ROUTER_CANDIDATES = build_router_candidates()

print(
    """
=======================================================
             🤖 AI KNOWLEDGE CHATBOT
=======================================================
🇵🇰 Pakistan | 🧮 Maths | ⚛️ Physics | 💻 Programming | 🌍 General Knowledge
🧠 Personal information can be saved in user_profile.json
Type 'bye' to exit.
=======================================================
"""
)
