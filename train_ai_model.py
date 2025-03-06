import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# ✅ Load dataset
df = pd.read_csv("dataset.csv")

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42)

# ✅ Improve TF-IDF settings
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.95, min_df=2, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

# ✅ Use Random Forest Classifier
clf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
clf.fit(X_train_vec, y_train)

# ✅ Save model and vectorizer
joblib.dump(clf, "ticket_classifier_pipeline.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# ✅ Evaluate Model
X_test_vec = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_vec)
accuracy = (y_pred == y_test.values).mean()
print(f"Test Accuracy: {accuracy:.2f}")

# ✅ Print Classification Report
print(classification_report(y_test, y_pred))

# ✅ Test with a sample ticket
new_ticket = "Cannot connect to email, getting password incorrect error"
new_vec = vectorizer.transform([new_ticket])
pred_label = clf.predict(new_vec)[0]
print("Predicted category for new ticket:", pred_label)
