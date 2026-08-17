from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report)
messages = [
    "win free money now",
    "claim your free prize",
    "congratulations you won a prize",
    "free lottery winner",
    "click here to win money",
    "you won a free gift",
    "claim your reward now",
    "exclusive free offer",
    "meeting at 5 pm",
    "please send the report",
    "project meeting tomorrow",
    "call me when you reach home",
    "your appointment is tomorrow",
    "please submit the assignment",
    "let us discuss the project",
    "see you at college tomorrow",
    "free cash reward",
    "you are a lucky winner",
    "claim your free money",
    "please attend the meeting",
    "send me the project file",
    "your class starts at 9"
]
labels = [
    1, 1, 1, 1,
    1, 1, 1, 1,
    0, 0, 0, 0,
    0, 0, 0, 0,
    1, 1, 1,
    0, 0, 0]
X_train, X_test, y_train, y_test = train_test_split(
    messages,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
model = MultinomialNB(alpha=1.0)
model.fit(X_train_vectorized, y_train)
y_pred = model.predict(X_test_vectorized)
print("Predictions:")
print(y_pred)
print("\nActual:")
print(y_test)
probabilities = model.predict_proba(X_test_vectorized)
print("\nPrediction Probabilities:")
print(probabilities)
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))
print("\nPrecision:")
print(precision_score(y_test, y_pred))
print("\nRecall:")
print(recall_score(y_test, y_pred))
print("\nF1 Score:")
print(f1_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nVocabulary:")
print(vectorizer.get_feature_names_out())
new_message = ["congratulations you won free money"]
new_message_vectorized = vectorizer.transform(new_message)
prediction = model.predict(new_message_vectorized)
probability = model.predict_proba(new_message_vectorized)
print("\nNew Message:")
print(new_message[0])
print("\nPrediction:")
if prediction[0] == 1:
    print("SPAM")
else:
    print("NORMAL")
print("\nProbability:")
print(probability)
