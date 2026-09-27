import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight

# Import functions of eda.py
from eda import (
    parse_period, 
    sentiment_from_rating, 
    add_text_features, 
    TEXT_COL, 
    PERIOD_COL, 
    RATING_COL
)

def normalize_text(text):
    """Paso 6: Normalization without lowercase and urls."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Eliminar URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    return text

def run_pipeline():
    # --- 0. charge data ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    route = os.path.join(current_dir, 'data', 'raw', 'DisneylandReviews.csv')
    df = pd.read_csv(route, encoding='latin-1')
    print(f"Data original shape: {df.shape}")

    # --- Phase 1:  Global clean data  ---
    
    # Step 1: 	Drop exact duplicate rows
    df_clean = df.drop_duplicates()
    
    # Step 2: Resolve duplicate texts
    df_clean = df_clean.drop_duplicates(subset=[TEXT_COL], keep='first')
    
    # Step 3: "missing" -> NaT
    df_clean['period'] = parse_period(df_clean[PERIOD_COL])
    
    # Step 4: three class from rating
    df_clean['sentiment'] = sentiment_from_rating(df_clean[RATING_COL], scheme='three_class')
    # Nos aseguramos de eliminar filas donde no se pudo extraer el sentimiento
    df_clean = df_clean.dropna(subset=['sentiment'])

    # --- PHASE 2: Features ---
    
    # get variables of text (n_words, n_sentences, unique_ratio
    df_clean = add_text_features(df_clean)

    # Paso 6: 	lowercase, strip URLs, keep negations
    df_clean['text_normalised'] = df_clean[TEXT_COL].apply(normalize_text)

    print(f"Data Transform: {df_clean.shape}")

    # --- PHASE 3: SPLIT ---

    X = df_clean.drop(columns=['sentiment'])
    y = df_clean['sentiment']
    
    # The recommendation is a stratified split after text deduplication, 70/15/15.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # --- PHASE 4:Learn parameters ---
    
    print(" Adjust the vectorizer y class balance")
    
    # Step 7: Adjust the vectorizer 
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train['text_normalised'])
    X_test_tfidf = vectorizer.transform(X_test['text_normalised'])

    # Step 8:  manage weigth imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, weights))

    print("\n¡Pipeline ejecutado exitosamente!")
    print(f"- Set de entrenamiento: {X_train.shape[0]} reseñas")
    print(f"- Set de prueba: {X_test.shape[0]} reseñas")

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer, class_weights_dict

if __name__ == "__main__":
    run_pipeline()