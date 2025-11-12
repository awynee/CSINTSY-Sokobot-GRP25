import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier  # better than a single tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from scipy import sparse
import joblib
import os

RANDOM_STATE = 42
DATA_PATH = "dataset.csv"
MODEL_OUT = "pinoybot_model.pkl"

# Feature extraction
def extract_numeric_features(word: str):
    vowels = set("aeiouAEIOU")
    w = word if isinstance(word, str) else ""
    w_lower = w.lower()
    prefixes_fil = ['mag', 'nag', 'pin', 'pa', 'ka', 'i', 'in', 'um']
    prefixes_eng = ['un', 're', 'pre', 'dis', 'mis', 'non']
    suffixes_eng = ['ness', 'ship', 'able', 'ment', 'tion', 'ing', 'ed', 'ly']

    features = {
        'word_len': len(w),
        'num_vowels': sum(1 for ch in w if ch in vowels),
        'ratio_vowel': sum(1 for ch in w if ch in vowels)/len(w) if len(w) > 0 else 0.0,
        'num_consonants': sum(1 for ch in w if ch.isalpha() and ch.lower() not in vowels),
        'is_capitalized': int(w[0].isupper()) if len(w) > 0 else 0,
        'is_proper_noun': int(w.istitle()),
        'has_digit': int(any(ch.isdigit() for ch in w)),
        'has_mixed_alnum': int(any(ch.isdigit() for ch in w) and any(ch.isalpha() for ch in w)),
        'has_hyphen': int('-' in w),
        'ends_with_vowel': int(len(w) > 0 and w[-1] in vowels),
        'ends_with_consonant': int(len(w) > 0 and w[-1].isalpha() and w[-1].lower() not in vowels),
        'has_prefix_fil': int(any(w_lower.startswith(p) for p in prefixes_fil)),
        'has_prefix_eng': int(any(w_lower.startswith(p) for p in prefixes_eng)),
        'has_suffix_eng': int(any(w_lower.endswith(s) for s in suffixes_eng)),
        'num_nonalpha': sum(1 for ch in w if not ch.isalpha())
    }
    return features

# Label normalization
def normalize_label(raw_label: str) -> str:
    if not isinstance(raw_label, str):
        return "OTH"
    lab = raw_label.strip().lower()
    if lab.startswith("fil") or lab.startswith("cs"):
        return "FIL"
    if lab.startswith("eng"):
        return "ENG"
    return "OTH"

def main():
    # loading of the dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'word' not in df.columns or 'label' not in df.columns:
        raise ValueError("'word' and 'label' not found in columns")

    df['label_norm'] = df['label'].apply(normalize_label)
    print("Label distribution:")
    print(df['label_norm'].value_counts())

    #extracting numeric features
    X_numeric = [extract_numeric_features(w) for w in df['word']]
    X_numeric_sparse = sparse.csr_matrix([list(d.values()) for d in X_numeric])
    numeric_feature_names = list(X_numeric[0].keys())

    # extracting character n-grams
    vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2,4), lowercase=True)
    char_ngrams = vectorizer.fit_transform(df['word'].astype(str))

    # combining extractions
    X_combined = sparse.hstack([X_numeric_sparse, char_ngrams])
    feature_names = numeric_feature_names + vectorizer.get_feature_names_out().tolist()

    # encoding labels
    le = LabelEncoder()
    y = le.fit_transform(df['label_norm'])
    print("Label classes:", le.classes_)

    # splitting dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_combined, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"Data sizes: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

    # random forest classifier
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # evaluation function
    def evaluate(X_eval, y_eval, set_name):
        y_pred = clf.predict(X_eval)
        unique_labels = sorted(set(y_eval))
        target_names = [le.classes_[i] for i in unique_labels]
        print(f"\n--- {set_name} set performance ---")
        print("Accuracy:", accuracy_score(y_eval, y_pred))
        print(classification_report(y_eval, y_pred, labels=unique_labels, target_names=target_names))

    evaluate(X_val, y_val, "Validation")
    evaluate(X_test, y_test, "Test")

    # save model
    joblib.dump({
        'model': clf,
        'label_encoder': le,
        'char_vectorizer': vectorizer,
        'numeric_feature_names': numeric_feature_names
    }, MODEL_OUT)
    print(f"\nSaved model to {MODEL_OUT}")

if __name__ == "__main__":
    main()
