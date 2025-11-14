"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project,
which identifies the language of each word in a code-switched
Filipino-English text. The function is designed to be called with a list
of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

The trained model is saved as a pickle file and loaded here for prediction.
"""

import os
import joblib
from scipy import sparse
from typing import List

# =========================
# CONFIG
# =========================
MODEL_FILE = "pinoybot_model.pkl"
NUMERIC_FEATURE_NAMES = []  # will be filled when loading the model


# =========================
# Feature Extraction
# =========================
def extract_numeric_features(word: str) -> dict:
    """Extract numeric features from a word for the model."""
    vowels = set("aeiouAEIOU")
    w = word if isinstance(word, str) else ""
    w_lower = w.lower()

    prefixes_fil = ['mag', 'nag', 'pin', 'pa', 'ka', 'i', 'in', 'um', 'ma', 'na']
    prefixes_eng = ['un', 're', 'pre', 'dis', 'mis', 'non', 'an']
    suffixes_eng = ['ness', 'ship', 'able', 'ment', 'tion', 'ing', 'ed', 'ly', 'ify', 'ance', 'ence', 'ible']

    features = {
        'wordLength': len(w),
        'vowelCount': sum(1 for ch in w if ch in vowels),
        'vowelRatio': sum(1 for ch in w if ch in vowels)/len(w) if len(w) > 0 else 0.0,
        'consonantCount': sum(1 for ch in w if ch.isalpha() and ch.lower() not in vowels),
        'isCapitalized': int(w[0].isupper()) if len(w) > 0 else 0,
        'isPropNoun': int(w.istitle()),
        'hasDigits': int(any(ch.isdigit() for ch in w)),
        'hasMixedString': int(any(ch.isdigit() for ch in w) and any(ch.isalpha() for ch in w)),
        'hasHyphen': int('-' in w),
        'endingVowel': int(len(w) > 0 and w[-1] in vowels),
        'endingConsonant': int(len(w) > 0 and w[-1].isalpha() and w[-1].lower() not in vowels),
        'hasPrefixFil': int(any(w_lower.startswith(p) for p in prefixes_fil)),
        'hasPrefixEng': int(any(w_lower.startswith(p) for p in prefixes_eng)),
        'hasSuffixEng': int(any(w_lower.endswith(s) for s in suffixes_eng)),
        'numNonAlpha': sum(1 for ch in w if not ch.isalpha()),
        'hasMixedCase': int(any(ch.isupper() for ch in w[1:]) and w[0].isupper()),
        'isAllCaps': int(w.isupper()),
        'isTitleWithLower': int(w.istitle() and w_lower not in prefixes_fil)
    }
    return features


# =========================
# Model Handling
# =========================
def load_model(model_file: str = MODEL_FILE):
    """Load the trained model, label encoder, vectorizer, and numeric feature order."""
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Trained model file not found: {model_file}")

    saved_data = joblib.load(model_file)
    model = saved_data['model']
    le = saved_data['label_encoder']
    vectorizer = saved_data['char_vectorizer']
    global NUMERIC_FEATURE_NAMES
    NUMERIC_FEATURE_NAMES = saved_data['num_featureNames']

    return model, le, vectorizer


# =========================
# Tagging Function
# =========================
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Returns a list of tags ("ENG", "FIL", "OTH") for each token.
    """
    # Convert all tokens to string (safety)
    tokens = [str(t) if t is not None else "" for t in tokens]

    model, le, vectorizer = load_model()

    # Numeric features
    X_num = [extract_numeric_features(w) for w in tokens]
    X_num_sparse = sparse.csr_matrix([[d[f] for f in NUMERIC_FEATURE_NAMES] for d in X_num])

    # Character n-gram features
    X_char = vectorizer.transform(tokens)

    # Combine features
    X_combined = sparse.hstack([X_num_sparse, X_char])

    # Predict
    preds_encoded = model.predict(X_combined)
    preds = le.inverse_transform(preds_encoded)

    # Convert to list
    tags = [str(p) for p in preds]

    # Post-processing: likely names or proper nouns
    for i, token in enumerate(tokens):
        if token[0].isupper() and i != 0 and tags[i] in ["FIL", "ENG"]:
            if token.isalpha() and len(token) > 1:
                tags[i] = "OTH"

    return tags


# =========================
# Example / Test
# =========================
if __name__ == "__main__":
    example_tokens = [
        "Ibinibenta","ng","grupo","ng","mga","suspek","ang","Tocilizumab",
        "400mg","sa","halagang","P85",",","000","tatlong","doble","sa","orihinal",
        "na","halaga","nitong","Php28",",","830",".","84",",","habang","ang",
        "Tocilizumab","80","mg","naman","ay","ibinibenta","ng","mga","suspek",
        "sa","halagang","P25",",","000","kumpara","sa","suggested","retail",
        "price","na","P8",",","811","lang","."
    ]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print("\nPredicted tags:", tags)
