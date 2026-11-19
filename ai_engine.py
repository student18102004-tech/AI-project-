
from collections import deque
import time
import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# MODULE 1: BFS (Search Algorithm)
# ==========================================

CAREER_GRAPH = {

    "Computer Science": [
        "Software Development",
        "Data Science",
        "Cybersecurity",
        "AI/ML"
    ],

    "Software Development": [
        "Web Development",
        "App Development",
        "Backend Development"
    ],

    "Data Science": [
        "Data Analysis",
        "Machine Learning",
        "Business Intelligence"
    ],

    "AI/ML": [
        "Deep Learning",
        "NLP",
        "Computer Vision"
    ],

    "Cybersecurity": [
        "Network Security",
        "Ethical Hacking"
    ]
}


def bfs_career_exploration(start_category):

    visited = set()
    queue = deque([start_category])
    result = []

    while queue:

        node = queue.popleft()

        if node not in visited:

            visited.add(node)
            result.append(node)

            for neighbor in CAREER_GRAPH.get(node, []):

                if neighbor not in visited:
                    queue.append(neighbor)

    return result


# ==========================================
# MODULE 2: MACHINE LEARNING
# Logistic Regression + TF-IDF
# ==========================================

TRAINING_DATA = [

    # Data Science (6)

    {"skills": "python sql excel tableau data",
     "category": "Data Science"},

    {"skills": "python pandas numpy machine learning",
     "category": "Data Science"},

    {"skills": "sql power bi statistics",
     "category": "Data Science"},

    {"skills": "data cleaning visualization python",
     "category": "Data Science"},

    {"skills": "statistics excel r sql",
     "category": "Data Science"},

    {"skills": "pandas numpy matplotlib seaborn",
     "category": "Data Science"},


    # Web Development (6)

    {"skills": "html css javascript react web",
     "category": "Web Development"},

    {"skills": "javascript node.js backend api",
     "category": "Web Development"},

    {"skills": "html css php mysql",
     "category": "Web Development"},

    {"skills": "react angular frontend web",
     "category": "Web Development"},

    {"skills": "nodejs express mongodb api",
     "category": "Web Development"},

    {"skills": "typescript css html vue",
     "category": "Web Development"},


    # AI/ML (6)

    {"skills": "python tensorflow pytorch deep learning",
     "category": "AI/ML"},

    {"skills": "nlp text mining transformers python",
     "category": "AI/ML"},

    {"skills": "opencv computer vision python",
     "category": "AI/ML"},

    {"skills": "keras tensorflow neural networks",
     "category": "AI/ML"},

    {"skills": "machine learning scikit-learn python",
     "category": "AI/ML"},

    {"skills": "ai algorithms deep learning",
     "category": "AI/ML"},


    # Cybersecurity (6)

    {"skills": "networking linux wireshark security",
     "category": "Cybersecurity"},

    {"skills": "hacking python networking firewall",
     "category": "Cybersecurity"},

    {"skills": "penetration testing ethical hacking",
     "category": "Cybersecurity"},

    {"skills": "cryptography network security linux",
     "category": "Cybersecurity"},

    {"skills": "soc analyst siem incident response",
     "category": "Cybersecurity"},

    {"skills": "firewall linux security audit",
     "category": "Cybersecurity"},


    # Software Development (6)

    {"skills": "python oop git software engineering",
     "category": "Software Development"},

    {"skills": "java c++ algorithms data structures",
     "category": "Software Development"},

    {"skills": "agile scrum software testing",
     "category": "Software Development"},

    {"skills": "c sharp dotnet backend",
     "category": "Software Development"},

    {"skills": "rest api microservices java",
     "category": "Software Development"},

    {"skills": "unit testing git docker",
     "category": "Software Development"}
]


# ==========================================
# TRAIN ML MODEL
# ==========================================

def train_ml_model():

    df = pd.DataFrame(TRAINING_DATA)

    # Convert text into numerical TF-IDF vectors

    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(df["skills"])

    y = df["category"]

    # Split dataset into training and testing

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create Logistic Regression model

    model = LogisticRegression()

    model.fit(X_train, y_train)

    # Evaluate model

    y_pred = model.predict(X_test)

    print("\n=== ML MODEL EVALUATION METRICS ===")

    print(
        f"Accuracy:  {accuracy_score(y_test, y_pred) * 100:.2f}%"
    )

    print(
        f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100:.2f}%"
    )

    print(
        f"Recall:    {recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100:.2f}%"
    )

    print(
        f"F1-Score:  {f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100:.2f}%"
    )

    print("===================================\n")

    # Train final model on complete dataset

    model.fit(X, y)

    # Save model and vectorizer

    base_dir = os.path.dirname(os.path.abspath(__file__))

    joblib.dump(
        model,
        os.path.join(base_dir, "ml_model.pkl")
    )

    joblib.dump(
        vectorizer,
        os.path.join(base_dir, "vectorizer.pkl")
    )

    print("ML Model successfully trained and saved!")

    # IMPORTANT: Return both objects

    return model, vectorizer


# ==========================================
# PREDICT CAREER CATEGORY
# ==========================================

def predict_category(user_skills, model, vectorizer):

    # TEST 1: Empty input

    if not isinstance(user_skills, str) or not user_skills.strip():

        return "Please enter your skills."

    # Convert input into TF-IDF vector

    user_vector = vectorizer.transform([user_skills])

    # TEST 2: Unknown skills

    if user_vector.nnz == 0:

        return "No recognized skills found. Please enter valid skills."

    # Predict career category

    prediction = model.predict(user_vector)

    return prediction[0]


# ==========================================
# MODULE 3: JOB MATCHING
# TF-IDF + COSINE SIMILARITY
# ==========================================

def get_job_recommendations(user_skills, jobs_list):

    if not jobs_list:
        return []

    if not user_skills or not user_skills.strip():
        return []

    job_skills = [
        job["required_skills"].replace(";", " ")
        for job in jobs_list
    ]

    user_skills_clean = user_skills.replace(";", " ")

    vectorizer = TfidfVectorizer()

    all_text = job_skills + [user_skills_clean]

    tfidf_matrix = vectorizer.fit_transform(all_text)

    user_vector = tfidf_matrix[-1]

    jobs_vectors = tfidf_matrix[:-1]

    similarity_scores = cosine_similarity(
        user_vector,
        jobs_vectors
    )[0]

    scored_jobs = []

    for i, score in enumerate(similarity_scores):

        match_percentage = round(score * 100, 2)

        job = dict(jobs_list[i])

        job["match_score"] = match_percentage

        scored_jobs.append(job)

    # Sort jobs by highest match score

    scored_jobs.sort(
        key=lambda x: x["match_score"],
        reverse=True
    )

    return scored_jobs


# ==========================================
# MODULE 4: SKILL GAP ANALYSIS
# ==========================================

def get_skill_gaps(user_skills, job_skills):

    user_skills_list = [
        s.strip().lower()
        for s in user_skills.split(";")
        if s.strip()
    ]

    job_skills_list = [
        s.strip().lower()
        for s in job_skills.split(";")
        if s.strip()
    ]

    missing_skills = []

    for skill in job_skills_list:

        if skill not in user_skills_list:

            missing_skills.append(skill)

    return missing_skills


# ==========================================
# MODULE 5: COMPLETE TESTING
# ==========================================

if __name__ == "__main__":

    # --------------------------------------
    # 1. TESTING BFS + RUNTIME ANALYSIS
    # --------------------------------------

    print("\n--- 1. Testing BFS ---")

    start_node = "Computer Science"

    times = []

    result = []

    for i in range(10):

        start_time = time.perf_counter()

        result = bfs_career_exploration(start_node)

        end_time = time.perf_counter()

        execution_time = end_time - start_time

        times.append(execution_time)

        print(
            f"Run {i + 1}: {execution_time:.8f} seconds"
        )

    average_time = sum(times) / len(times)

    print("\nBFS Starting Category:", start_node)

    print("BFS Explored Categories:", result)

    print(
        f"Average BFS Execution Time: {average_time:.8f} seconds"
    )


    # --------------------------------------
    # 2. TESTING MACHINE LEARNING
    # --------------------------------------

    print("\n--- 2. Testing Machine Learning ---")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "ml_model.pkl")

    vectorizer_path = os.path.join(base_dir, "vectorizer.pkl")

    # Remove old model files

    if os.path.exists(model_path):
        os.remove(model_path)

    if os.path.exists(vectorizer_path):
        os.remove(vectorizer_path)

    # Train model and receive both objects

    model, vectorizer = train_ml_model()

    # Test prediction

    test_skills = "python sql data visualization"

    predicted = predict_category(
        test_skills,
        model,
        vectorizer
    )

    print("\nUser Skills:", test_skills)

    print("ML Model Prediction:", predicted)


    # --------------------------------------
    # 3. TESTING JOB MATCHING
    # --------------------------------------

    print("\n--- 3. Testing Job Matching ---")

    try:

        from database.db import get_all_jobs

        all_jobs = get_all_jobs()

        if all_jobs:

            recommendations = get_job_recommendations(
                test_skills,
                all_jobs
            )

            print("\nTop 3 Job Recommendations:")

            for job in recommendations[:3]:

                print(
                    f"- {job['title']} at {job['company']} "
                    f"(Match: {job['match_score']}%)"
                )

        else:

            print("Jobs database is empty.")

    except Exception as e:

        print("Job Matching Test Error:", e)


    # --------------------------------------
    # 4. TESTING CAREER PREDICTION
    # VALIDATION
    # --------------------------------------

    print("\n--- 4. Testing Career Prediction Validation ---")


    # TEST 1: Empty Input

    print("\nTest 1: Empty Input")

    result = predict_category(
        "",
        model,
        vectorizer
    )

    print("Input: Empty")

    print("Output:", result)


    # TEST 2: Unknown Skills

    print("\nTest 2: Unknown Skills")

    result = predict_category(
        "xyzabc qwerty",
        model,
        vectorizer
    )

    print("Input: xyzabc qwerty")

    print("Output:", result)


    # TEST 3: Valid Skills - Python, SQL

    print("\nTest 3: Valid Skills - Python, SQL")

    result = predict_category(
        "Python SQL",
        model,
        vectorizer
    )

    print("Input: Python SQL")

    print("Predicted Career:", result)


    # TEST 4: Valid Skills - Python, Machine Learning

    print("\nTest 4: Valid Skills - Python, Machine Learning")

    result = predict_category(
        "Python Machine Learning",
        model,
        vectorizer
    )

    print("Input: Python Machine Learning")

    print("Predicted Career:", result)


    print("\n--- All 4 Validation Tests Completed ---")

    print("\n--- AI ENGINE TESTING COMPLETED ---")
# ==========================================
# 5. TESTING SKILL GAP ANALYSIS
# ==========================================

# ==========================================
# 5. TESTING SKILL GAP ANALYSIS
# ==========================================

if __name__ == "__main__":

    print("\n--- 5. Testing Skill Gap Analysis ---")

    user_skills = "python;sql;excel"

    required_skills = "python;sql;excel;tableau;power bi"

    missing_skills = get_skill_gaps(
        user_skills,
        required_skills
    )

    print("User Skills:", user_skills)

    print("Required Skills:", required_skills)

    print("Missing Skills:", missing_skills)

    print("\n--- Skill Gap Testing Completed ---")