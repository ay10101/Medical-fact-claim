"""
Health Claim Classifier Module

Uses a two-stage approach:
  Stage 1 (Fast): lightweight keyword pre-check to ALLOW obvious health queries
                  and REJECT obvious non-health topics (sports, finance, etc.)
  Stage 2 (Semantic RAG): if a RAG engine is provided, the claim is embedded and
                          matched against the evidence dataset. If the best cosine
                          similarity exceeds SEMANTIC_THRESHOLD, it is classified
                          as health-related – regardless of whether any keyword matched.

This avoids hardcoding and makes classification data-driven. A query like
"Pineapple reduces acidity" will score high similarity against acidity/gut-health
records and will correctly pass.
"""

import re
from typing import Optional

# ── SEMANTIC threshold ───────────────────────────────────────────────────────
# Any query whose best evidence-dataset cosine similarity >= this value
# is considered health-related.
SEMANTIC_THRESHOLD = 0.30

# ── HIGH-CONFIDENCE health keywords ─────────────────────────────────────────
# These alone are strong enough to flag a claim as health-related
STRONG_HEALTH_KEYWORDS = {
    # Diseases / conditions
    "cancer", "diabetes", "hypertension", "autism", "dementia", "alzheimer",
    "asthma", "arthritis", "osteoarthritis", "tuberculosis", "malaria",
    "cholera", "typhoid", "dengue", "chikungunya", "hepatitis", "hiv", "aids",
    "covid", "covid-19", "sars", "ebola", "influenza", "pneumonia",
    "stroke", "seizure", "epilepsy", "parkinson", "sclerosis",
    "cystitis", "eczema", "psoriasis", "acne", "migraine",
    "syndrome", "disorder", "tumor", "carcinoma", "lymphoma", "leukemia",
    "ulcer", "hernia", "cataract", "glaucoma", "meningitis",
    # Symptoms and body signals
    "acidity", "nausea", "vomiting", "diarrhea", "diarrhoea", "constipation",
    "bloating", "indigestion", "heartburn", "reflux", "gerd",
    "fatigue", "lethargy", "dizziness", "headache",
    "swelling", "edema", "rash", "allergy", "allergies", "allergic",
    "insomnia", "anxiety", "depression",
    "breathlessness", "dyspnea", "palpitation", "palpitations",
    # Body systems, organs, functions
    "digestion", "digestive", "metabolism", "metabolic",
    "immunity", "immune", "inflammation", "inflammatory",
    "cardiovascular", "cardiac", "respiratory", "neurological", "hormonal",
    "thyroid", "pancreas", "liver", "kidney", "kidneys", "lungs", "lung",
    "colon", "bowel", "gut", "intestine", "intestinal",
    "bone", "bones", "joint", "joints", "muscle", "muscles",
    "blood", "heart", "brain", "spine", "nerve", "nerves",
    "urinary", "lymph", "prostate", "ovarian", "uterine", "cervical", "spleen",
    # Medical interventions & products
    "vaccine", "vaccines", "vaccination", "antibiotic", "antibiotics",
    "chemotherapy", "chemo", "dialysis", "paxlovid", "insulin",
    "medicine", "medicines", "medication", "dosage", "drug", "drugs",
    "supplement", "supplements", "vitamin", "vitamins", "probiotic",
    "therapy", "treatment", "surgery", "transplant", "physiotherapy",
    # Clinical / medical terms
    "clinical", "diagnosis", "prognosis", "biopsy", "mri", "xray",
    "ecg", "blood pressure", "cholesterol", "hemoglobin", "platelet",
    "antibody", "antigen", "pathogen", "bacteria", "viral", "virus",
    "infection", "infections", "symptom", "symptoms", "mortality", "death", "physical activity", "sedentary",
    "chronic", "acute", "benign", "malignant",
    # Pregnancy / reproductive
    "pregnancy", "pregnant", "prenatal", "postnatal", "miscarriage",
    "fertility", "contraceptive",
    # Professionals & facilities
    "doctor", "doctors", "physician", "surgeon", "nurse", "pharmacist",
    "hospital", "hospitals", "clinic", "icu", "emergency",
    "patient", "patients",
    # Foods / substances inherently health-focused
    "probiotic", "prebiotic", "antioxidant", "omega",
    "turmeric", "curcumin", "ginger",
    # Hinglish (strong medical terms)
    "bimari", "bimaari", "ilaj", "ilaaj", "dawa", "dawai",
    "aspatal", "khansi", "bukhar", "sehat", "dard",
    # Devanagari Hindi
    "कैंसर", "वैक्सीन", "टीका", "टीकाकरण", "दवा", "दवाई", "डॉक्टर", "अस्पताल",
    "बीमारी", "मधुमेह", "डायबिटीज", "बुखार", "एंटीबायोटिक", "कोविड", "स्वास्थ्य",
    "इलाज", "उपचार", "वायरस", "खांसी", "संक्रमण", "गर्भावस्था", "रक्त", "दिल",
    "पाचन", "अम्लता", "मतली",
}

# ── AMBIGUOUS keywords ──────────────────────────────────────────────────────
# These are health-relevant BUT too generic to stand alone.
# They only count if a HEALTH_CONTEXT_VERB or second ambiguous/strong keyword is also present.
# Uncertain cases without a keyword match fall through to semantic RAG stage.
AMBIGUOUS_HEALTH_KEYWORDS = {
    "sugar", "pain", "cold", "cough", "sore", "sores", "blister", "blisters",
    "skin", "stomach", "knee", "back",
    "mask", "masks", "soap", "hygiene", "sanitizer", "sanitize", "handwash",
    "herbal", "remedy", "remedies", "heal", "healing", "prevent", "prevention",
    "preventive", "relieve", "alleviate", "fever", "health", "disease", "diseases",
    "herpes", "lysine", "cbd", "cannabis", "cannabidiol",
    "tobacco", "smoking", "cigarette",
    "haldi", "doodh", "chot",  # Hinglish
    # Common foods/substances in potential health context
    "milk", "water", "lemon", "honey", "pineapple", "apple", "garlic",
    "green tea", "coffee", "salt", "orange",
    # Lifestyle and wellness terms
    "weight", "diet", "nutrition", "nutritional", "calorie", "calories",
    "exercise", "workout", "sleep", "stress", "mental",
    "fiber", "fibre",
}

# ── Context verbs that signal a health claim ────────────────────────────────
# If an ambiguous keyword appears with one of these, the claim is health-related
HEALTH_CONTEXT_VERBS = {
    "cause", "causes", "caused", "causing",
    "treat", "treats", "treated", "treating", "treatment",
    "cure", "cures", "cured", "curing",
    "prevent", "prevents", "prevented", "preventing",
    "reduce", "reduces", "reduced", "reducing",
    "increase", "increases", "increased", "increasing",
    "diagnose", "diagnosed", "diagnosis",
    "affect", "affects", "affected",
    "help", "helps", "helped", "helping",
    "fight", "fights", "fighting",
    "risk", "risks",
    "effective", "ineffective", "beneficial", "harmful",
    "safe", "unsafe", "dangerous",
    "improve", "improves", "improved",
    "spread", "spreads",
    "kill", "kills",
    "absorb", "absorbs",
    "boost", "boosts",
    "lower", "lowers",
    "raise", "raises",
    "protect", "protects",
    "linked", "associated", "correlation",
    "symptom", "symptoms",
    "good", "bad",
    "heals", "soothes", "relieves",
    "good for", "bad for",
    # Hindi/Hinglish verbs
    "fayda", "nuksan", "thik",
}

# ── Hard nonsense / non-health signal words ─────────────────────────────────
# If the claim contains these words alongside ambiguous keywords, it's not health-related
NONSENSE_SIGNALS = {
    "nasa", "rocket", "spaceship", "alien", "ufo", "planet", "galaxy",
    "cricket", "basketball", "baseball",
    "ipl", "fifa", "nba", "nfl", "premier",  # sports leagues
    "movie", "film", "cinema", "bollywood", "hollywood", "netflix",
    "bank", "money", "loan", "finance", "stock", "crypto", "bitcoin",
    "recipe", "cooking", "bake", "baking", "restaurant",
    "weather", "rain", "snow", "hurricane", "tornado",
    "politics", "government", "election", "vote", "parliament",
    "car", "drive", "driving", "traffic", "accident",
    "school", "college", "university", "exam",
    "song", "concert", "dance",
}

# ── Celebrities, Public Figures & Named Individuals ─────────────────────────
# Clinical databases (PubMed, Lancet, WHO) evaluate medical hypotheses and treatments;
# they do NOT verify personal medical diagnoses or celebrity health gossip.
CELEBRITY_AND_INDIVIDUAL_SIGNALS = {
    # Cricket & Sports personalities
    "virat", "kohli", "rohit", "sharma", "dhoni", "sachin", "tendulkar",
    "hardik", "pandya", "bumrah", "shubman", "gill", "kl rahul", "pant",
    "babar", "azam", "ronaldo", "messi", "neymar", "mbappe", "lebron",
    "federer", "nadal", "djokovic", "cricketer", "footballer",
    # Film & Entertainment celebrities
    "salman", "shah rukh", "srk", "amitabh", "bachchan", "deepika",
    "alia", "katrina", "ranbir", "ranveer", "kareena", "akshay",
    "tom cruise", "taylor swift", "selena", "actor", "actress", "celebrity",
    # Politicians & Public figures
    "modi", "rahul gandhi", "amit shah", "kejriwal", "yogi",
    "trump", "biden", "putin", "obama", "politician", "minister", "president",
}


def _keyword_stage(text_lower: str, tokens: set) -> Optional[bool]:
    """
    Stage 1 fast keyword pre-check.
    Returns True  → definitely health-related
    Returns False → definitely NOT health-related
    Returns None  → uncertain; let semantic stage decide
    """
    # Add multi-word ambiguous keywords to token set
    extended_tokens = set(tokens)
    for kw in AMBIGUOUS_HEALTH_KEYWORDS:
        if " " in kw and kw in text_lower:
            extended_tokens.add(kw)

    # 1. Celebrity Gossip / Personal Claims are strictly rejected as non-verifiable clinical claims
    celebrity_matches = extended_tokens.intersection(CELEBRITY_AND_INDIVIDUAL_SIGNALS)
    for cs in CELEBRITY_AND_INDIVIDUAL_SIGNALS:
        if " " in cs and cs in text_lower:
            celebrity_matches.add(cs)
    if celebrity_matches:
        return False

    # 2. Detect personal health diagnosis grammar: e.g. "[Name] ko [Disease] hai / tha / hua"
    if re.search(r"^[a-zA-Z\u0900-\u097F\s]{2,25}\s+(?:ko|ke)\s+[a-zA-Z\u0900-\u097F\s]+(?:\s+hai|\s+tha|\s+hua|\s+ho\s+gaya)\b", text_lower):
        generic_subjects = {
            "bacho", "bachon", "bachche", "patient", "patients", "mariz", "mareez",
            "insan", "log", "logon", "mahila", "mahilaon", "pregnant", "elderly",
            "adults", "children", "women", "men", "sugar", "haldi", "doodh", "pani",
            "khansi", "bukhar", "dawa", "vaccine", "exercise", "walk", "sleeping"
        }
        first_word = text_lower.split()[0] if text_lower.split() else ""
        if first_word not in generic_subjects:
            return False

    # 3. Hard nonsense signals (NASA, crypto, movie, etc.)
    nonsense_matches = extended_tokens.intersection(NONSENSE_SIGNALS)
    if nonsense_matches:
        return False

    # Check strong health keywords (multi-word too)
    strong_matches: set = tokens.intersection(STRONG_HEALTH_KEYWORDS)
    for kw in STRONG_HEALTH_KEYWORDS:
        if " " in kw and kw in text_lower:
            strong_matches.add(kw)

    ambiguous_matches = extended_tokens.intersection(AMBIGUOUS_HEALTH_KEYWORDS)
    health_verb_matches = extended_tokens.intersection(HEALTH_CONTEXT_VERBS)
    for hv in HEALTH_CONTEXT_VERBS:
        if " " in hv and hv in text_lower:
            health_verb_matches.add(hv)

    if strong_matches:
        generic_strong = {
            "health", "disease", "diseases", "fever", "cough",
            "infection", "infections",
        }
        non_generic_strong = strong_matches - generic_strong
        if non_generic_strong:
            return True
        if not nonsense_matches or health_verb_matches:
            return True

    # Pure nonsense with no health words → reject immediately
    if nonsense_matches and not ambiguous_matches and not strong_matches:
        return False

    # Ambiguous + health verb → pass
    if ambiguous_matches and health_verb_matches:
        return True

    # Multiple ambiguous keywords alone → pass
    if len(ambiguous_matches) >= 2:
        return True

    # Uncertain — let semantic stage handle it
    return None


def is_health_related(text: str, rag_engine=None) -> bool:
    """
    Evaluates whether the input claim is related to health or medicine.

    Two-stage pipeline:
    1. Fast keyword check (O(1)) — returns immediately when confident.
    2. Semantic RAG check — only if rag_engine is provided and Stage 1 is uncertain.
       Embeds the claim and retrieves the best match from the evidence dataset.
       If cosine similarity >= SEMANTIC_THRESHOLD → health-related.

    Parameters
    ----------
    text       : The user's input claim string.
    rag_engine : Optional HealthcareRAGEngine instance. When provided, Stage 2
                 semantic matching runs on uncertain cases instead of defaulting
                 to False. This catches food/lifestyle/gut-health queries like
                 "Pineapple reduces acidity" that have no strong keyword.
    """
    if not text or not isinstance(text, str):
        return False

    text_lower = text.lower().strip()

    # If the input is or contains a medical or clinical URL, accept it for retrieval
    if re.search(r"https?://\S+", text_lower):
        if any(domain in text_lower for domain in [
            "pubmed", "ncbi", "nih.gov", "who.int", "cdc.gov", "cochrane",
            "lancet", "nejm", "bmj", "jama", "mayoclinic", "health", "medical",
            "webmd", "healthline", "nature.com", "cell.com", "sciencedirect"
        ]):
            return True
    tokens = set(re.findall(r"[\w\u0900-\u097F]+", text_lower))

    result = _keyword_stage(text_lower, tokens)
    if result is not None:
        return result

    # ── Stage 2: Semantic RAG classification ────────────────────────────────
    if rag_engine is not None:
        try:
            hits = rag_engine.retrieve_evidence(text, top_k=1)
            if hits:
                top_score = float(hits[0].get("similarity_score", 0.0))
                return top_score >= SEMANTIC_THRESHOLD
        except Exception as e:
            print(f"[HealthClassifier] Semantic stage error: {e}")

    # No RAG engine and keyword stage was uncertain → conservative reject
    return False
