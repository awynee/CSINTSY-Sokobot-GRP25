"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
import joblib
import numpy as np
from scipy import sparse
from typing import List


model_file = "pinoybot_model.pkl"

# Feature extraction
def extractNumericFeatures(word: str):
    vowels = set("aeiouAEIOU")
    w = word if isinstance(word, str) else ""
    w_lower = w.lower()
    prefixes_fil = ['mag', 'nag', 'pin', 'pa', 'ka', 'i', 'in', 'um', 'ma', 'na']
    prefixes_eng = ['un', 're', 'pre', 'dis', 'mis', 'non', 'an']
    suffixes_eng = ['ness', 'ship', 'able', 'ment', 'tion', 'ing', 'ed', 'ly', 'ify', 'ance','ence', 'ible']

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
        'numNonAlpha': sum(1 for ch in w if not ch.isalpha())
    }
    return features




# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:

    
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", or "OTH"), one per token.
    """
    # 1. Load your trained model from disk (e.g., using pickle or joblib)
    #    Example: with open('trained_model.pkl', 'rb') as f: model = pickle.load(f)
    #    (Replace with your actual model loading code)

    # 2. Extract features from the input tokens to create the feature matrix
    #    Example: features = ... (your feature extraction logic here)

    # 3. Use the model to predict the tags for each token
    #    Example: predicted = model.predict(features)

    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    #    Example: tags = [str(tag) for tag in predicted]

    # 5. Return the list of tags
    #    return tags

    # You can define other functions, import new libraries, or add other Python files as needed, as long as
    # the tag_language function is retained and correctly accomplishes the expected task.

    # Currently, the bot just tags every token as FIL. Replace this with your more intelligent predictions.

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Trained model file not found: {model_file}")

    saved_data = joblib.load(model_file)
    model = saved_data["model"]
    le = saved_data["label_encoder"]
    vectorizer = saved_data["char_vectorizer"]
    num_featureNames = saved_data["num_featureNames"]

    X_num = [extractNumericFeatures(w) for w in tokens]
    X_numSparse = sparse.csr_matrix([list(d.values()) for d in X_num])
    X_char = vectorizer.transform(tokens)
    X_combined = sparse.hstack([X_numSparse, X_char])

    preds_encoded = model.predict(X_combined)
    preds = le.inverse_transform(preds_encoded)

    tags = [str(p) for p in preds]

    return tags


if __name__ == "__main__":
    # Example usage
    example_tokens = ["ito", "ba", "yung", "restaurant", "na", "popular", "?"]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print("Predicted tags:", tags)

