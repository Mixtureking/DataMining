import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Dữ liệu huấn luyện mở rộng
text_data = [
    # English
    "Hello, how are you?", "Good morning!", "This is an English sentence.", "I love programming.",
    "Welcome to the world of AI!", "ChatGPT is amazing.", "Python is my favorite language.",
    "Have a great day!", "What are you doing?", "Machine learning is fun.",

    # French
    "Bonjour, comment ça va?", "C'est une belle journée.", "Je suis étudiant.",
    "La vie est belle.", "Il fait très chaud aujourd'hui.", "J'adore la musique française.",
    "Paris est une belle ville.", "Apprendre le français est intéressant.", "J'aime le café.",

    # Spanish
    "Hola, ¿cómo estás?", "Buenos días!", "Esto es una oración en español.",
    "Me gusta la comida mexicana.", "Estoy aprendiendo español.", "La casa es grande.",
    "España tiene una cultura increíble.", "El fútbol es muy popular en España.", "Voy al mercado.",

    # German
    "Hallo, wie geht's?", "Guten Morgen!", "Das ist ein deutscher Satz.",
    "Ich liebe Deutschland.", "Berlin ist die Hauptstadt von Deutschland.", "Das Wetter ist schön.",
    "Ich spreche ein bisschen Deutsch.", "Wie spät ist es?", "Die Sprache ist schwer.",

    # Italian
    "Ciao, come stai?", "Buongiorno!", "Questa è una frase italiana.",
    "Mi piace la pizza.", "L'Italia è un paese bellissimo.", "Roma è la capitale d'Italia.",
    "Sto imparando l'italiano.", "Come si dice in italiano?", "Mi piace viaggiare."
]

labels = ["English"] * 10 + ["French"] * 9 + ["Spanish"] * 9 + ["German"] * 9 + ["Italian"] * 9

# 2️⃣ Chuyển đổi văn bản thành vector TF-IDF (dùng n-grams 1-3 để tăng độ chính xác)
vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words=None)
X = vectorizer.fit_transform(text_data)

# 3️⃣ Chia tập dữ liệu train/test (giảm test_size để tránh mất cân bằng dữ liệu)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)

# 4️⃣ Kiểm tra số lượng mẫu train/test để tránh lỗi
print("Train distribution:", Counter(y_train))
print("Test distribution:", Counter(y_test))

# 5️⃣ Huấn luyện mô hình SVM Multiclass (kernel 'rbf' để tăng độ chính xác)
svm_model = SVC(kernel='rbf', C=10, decision_function_shape='ovr')
svm_model.fit(X_train, y_train)

# 6️⃣ Dự đoán trên tập test
y_pred = svm_model.predict(X_test)

# 7️⃣ Đánh giá mô hình
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, zero_division=1))

# 8️⃣ Dự đoán ngôn ngữ mới
def predict_language(text):
    text_vectorized = vectorizer.transform([text])
    prediction = svm_model.predict(text_vectorized)
    return prediction[0]

# Test với câu mới
sample_texts = [
    "Guten Abend, wie geht es Ihnen?",  # German
    "J'aime le fromage et le vin.",     # French
    "Estoy aprendiendo a programar.",   # Spanish
    "Good evening, how are you?",       # English
    "Roma è una città bellissima.",      # Italian
    "Halo niece and nephew it's Uncle Roger."
]

for text in sample_texts:
    print(f"'{text}' → Predicted: {predict_language(text)}")
