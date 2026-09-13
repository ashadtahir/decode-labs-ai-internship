# AI Recommendation Logic - Tech Stack Recommender
# Project 3 - DecodeLabs AI Internship

# Import required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the job role dataset
print("Loading job role dataset...")
df = pd.read_csv("raw_skills.csv")
print(f"Loaded {len(df)} job roles: {list(df['job_title'])}")
print("-" * 60)


# Step 2: Get user skills input
def get_user_skills():
    """Ask the user for their skills and validate at least 3 are given."""
    while True:
        skills_input = input("\nEnter your skills (separated by commas): ")
        # Split by comma and remove empty/blank entries
        user_skills = [s.strip() for s in skills_input.split(",") if s.strip()]

        if len(user_skills) < 3:
            print(f"You entered {len(user_skills)} skill(s). Please enter at least 3 skills.")
            continue

        return user_skills


user_skills = get_user_skills()
print("\nYour skills:", ", ".join(user_skills))
print("-" * 60)

# Step 3: Build the TF-IDF vectorizer
# Combine user skills into one text document
user_text = " ".join(user_skills)

# Put the user input together with the dataset for vectorization
all_text = pd.concat([df["skills"], pd.Series([user_text])], ignore_index=True)

# Create and fit the TF-IDF vectorizer
print("\nApplying TF-IDF vectorization...")
vectorizer = TfidfVectorizer(lowercase=True)
tfidf_matrix = vectorizer.fit_transform(all_text)

# User skills vector is the last row, job role vectors are the rest
user_vector = tfidf_matrix[-1]
job_vectors = tfidf_matrix[:-1]
print(f"TF-IDF matrix shape: {tfidf_matrix.shape} (documents x unique words)")

# Step 4: Calculate cosine similarity between user and every job role
print("\nCalculating cosine similarity...")
similarity_scores = cosine_similarity(user_vector, job_vectors).flatten()

# Step 5: Sort job roles by similarity score (highest to lowest)
ranked = sorted(
    zip(df["job_title"], similarity_scores),
    key=lambda x: x[1],
    reverse=True,
)

# Step 6: Display the Top 3 recommendations
print("\n" + "=" * 60)
print("            TOP 3 RECOMMENDED CAREER ROLES")
print("=" * 60)
print(f"{'#':<4} {'Job Role':<25} {'Similarity Score':<15}")
print("-" * 60)
for i, (job_title, score) in enumerate(ranked[:3], start=1):
    print(f"{i:<4} {job_title:<25} {score:.4f}")

print("=" * 60)

# Recommendation pipeline overview
print("\nRecommendation Pipeline:")
print("Input\n  |")
print("  v")
print("TF-IDF Vectorization\n  |")
print("  v")
print("Cosine Similarity\n  |")
print("  v")
print("Sort by similarity\n  |")
print("  v")
print("Top 3 Recommendations\n  |")
print("  v")
print("Output")