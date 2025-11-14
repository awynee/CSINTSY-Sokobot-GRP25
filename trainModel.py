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
def extractNumericFeatures(word: str):
    vowels = set("aeiouAEIOU")
    w = word if isinstance(word, str) else ""
    w_lower = w.lower()
    # add more prefixes, suffixes if there are more common ones but not too many
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
        'numNonAlpha': sum(1 for ch in w if not ch.isalpha()),
        'hasMixedCase': int(any(ch.isupper() for ch in w[1:]) and w[0].isupper()),  # uncommon in English words
        'isAllCaps': int(w.isupper()),
        'isTitleWithLower': int(w.istitle() and w_lower not in prefixes_fil)
    }
    return features

# Label normalization
def normalizeLabel(raw_label: str) -> str:
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
        raise ValueError("'word' and 'label' not found in the given columns")
    
    # deciding which column to use for the correct word labels
    label_col = "corrected_label" if "corrected_label" in df.columns else "label"
    print(f"Using '{label_col}' column for labels")

    # applying the normalization function to clean up the labels
    df['normedLabel'] = df['label'].apply(normalizeLabel)

    # displays how many words there are in each category after normalization
    print("Label distribution:")
    print(df['normedLabel'].value_counts())

    # extracting numeric features
    X_num = [extractNumericFeatures(w) for w in df['word']]
    X_numSparse = sparse.csr_matrix([list(d.values()) for d in X_num])
    num_featureNames = list(X_num[0].keys())

    # extracting character n-grams
    vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2,4), lowercase=True)
    charNgrams = vectorizer.fit_transform(df['word'].astype(str))

    # combining extractions
    X_combined = sparse.hstack([X_numSparse, charNgrams])
    featureNames = num_featureNames + vectorizer.get_feature_names_out().tolist()

    # encoding labels
    le = LabelEncoder()
    y = le.fit_transform(df['normedLabel'])
    print("Label classes:", le.classes_)

    # splitting dataset to 70-15-15
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
        uniqueLabels = sorted(set(y_eval))
        targetNames = [le.classes_[i] for i in uniqueLabels]
        print(f"\n--- {set_name} set performance ---")
        print("Accuracy:", accuracy_score(y_eval, y_pred))
        print(classification_report(y_eval, y_pred, labels= uniqueLabels, target_names = targetNames))

    evaluate(X_val, y_val, "Validation")
    evaluate(X_test, y_test, "Test")

    joblib.dump({
        'model': clf,
        'label_encoder': le,
        'char_vectorizer': vectorizer,
        'num_featureNames': num_featureNames
    }, MODEL_OUT)
    print(f"\nSaved model to {MODEL_OUT}")

if __name__ == "__main__":
    main()
