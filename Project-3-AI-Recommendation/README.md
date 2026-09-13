# Project 3: AI Recommendation Logic — Tech Stack Recommender

DecodeLabs AI Internship

---

## Project Title

AI Recommendation Logic — A Content-Based Tech Stack Recommender

---

## Objective

Build a simple content-based recommendation system that recommends the most relevant career/job roles based on the skills a user enters. The system takes at least 3 skills as input, matches them against a small dataset of job roles, and returns the Top 3 most similar career options.

---

## What Recommendation Systems Are Doing in This Project

Recommendation systems try to predict what a user might like based on available information. This project is a recommendation system where:

- **The "user"** is a person entering their technical skills
- **The "items"** are career/job roles (Data Scientist, DevOps Engineer, etc.)
- **The "match"** is made by comparing the user's skills against the skills each job role requires

The system does not have a "like" or rating history — it judges similarity purely from the text content of the skills.

---

## Content-Based Filtering

Content-based filtering recommends items by comparing the actual content (text) of items with the user's profile (their entered skills). Here, each job role is described by a list of skills (its content). The user's skills form their profile. Recommendations are made by finding job roles whose skill content overlaps most strongly with the user's skills.

No interaction data from other users is needed — only the relationship between what the user has and what each job role requires.

---

## Dataset Used

The dataset is stored in `raw_skills.csv` and contains 10 job roles. Each row has:

- **job_title** — the name of the career role
- **skills** — a space-separated list of relevant skills for that role

| job_title | skills |
|---|---|
| Data Scientist | python machine learning statistics data analysis pandas numpy visualization |
| DevOps Engineer | python cloud computing docker kubernetes automation ci cd linux shell |
| Backend Developer | python java javascript sql api database django flask node |
| Cloud Architect | cloud computing aws azure gcp architecture networking security design |
| Software Developer | python java javascript git testing agile problem solving collaboration |
| Machine Learning Engineer | python machine learning deep learning tensorflow data pipeline model deployment |
| Data Analyst | python sql excel tableau data analysis statistics visualization reporting |
| Web Developer | javascript html css react node frontend backend web |
| Cybersecurity Analyst | security networking linux monitoring threat detection risk management |
| Database Administrator | sql database mysql postgresql oracle backup performance tuning |

---

## TF-IDF

TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numbers so a computer can compare it.

- **Term Frequency (TF):** how often a word appears in a document
- **Inverse Document Frequency (IDF):** how rare a word is across all documents — rare words get higher importance than common filler words

`TfidfVectorizer` from scikit-learn converts:
- every job role's skill string, and
- the user's combined skills,

into numerical vectors. Each unique skill becomes a dimension, and the value reflects how important that skill is for that particular document.

---

## Cosine Similarity

Cosine similarity measures the angle between two vectors (0 = completely different, 1 = identical direction). It ignores how long the vectors are and focuses on direction:

```
similarity = dot product of vectors / (length of vector A * length of vector B)
```

The user's skill vector is compared against every job role's vector. The higher the score, the more aligned that job role is with the user's skills.

---

## How the Recommendation Pipeline Works

```
Input
  |
  v
TF-IDF Vectorization
  |
  v
Cosine Similarity
  |
  v
Sort by similarity
  |
  v
Top 3 Recommendations
  |
  v
Output
```

1. **Input** — the user types their skills, separated by commas (minimum 3)
2. **TF-IDF Vectorization** — `TfidfVectorizer` converts the user's skills and all job role skills into numerical vectors
3. **Cosine Similarity** — the user vector is compared with every job role vector
4. **Sort by similarity** — job roles are ordered from highest to lowest similarity score
5. **Top 3 Recommendations** — the 3 highest-scoring roles are selected
6. **Output** — recommendations and their similarity scores are displayed in the terminal

---

## Top-3 Recommendation Logic

After cosine similarity is calculated for all roles, the results are sorted in descending order:

```python
ranked = sorted(
    zip(df["job_title"], similarity_scores),
    key=lambda x: x[1],
    reverse=True,
)
```

The first 3 entries of the sorted list become the final recommendations, shown with their similarity scores.

---

## Input Validation

- If the user enters **fewer than 3 skills**, the program asks again until at least 3 are provided
- If a skill does not appear in the job-role descriptions, it does not contribute to the similarity score with those roles

---

## Technologies / Libraries

- Python 3.x
- pandas — reading the CSV dataset
- scikit-learn — TF-IDF vectorization and cosine similarity

No other libraries are required.

---

## Installation

```bash
pip install pandas scikit-learn
```

---

## How to Run

```bash
cd Project-3-AI-Recommendation
python recommender.py
```

---

## Example Input

```
Enter your skills (separated by commas): Python, Cloud Computing, Automation
```

---

## Actual Example Output

The following is the real output produced by running `python recommender.py` with the input `Python, Cloud Computing, Automation`:

```
Loading job role dataset...
Loaded 10 job roles: ['Data Scientist', 'DevOps Engineer', 'Backend Developer', 'Cloud Architect', 'Software Developer', 'Machine Learning Engineer', 'Data Analyst', 'Web Developer', 'Cybersecurity Analyst', 'Database Administrator']
------------------------------------------------------------

Enter your skills (separated by commas): 
Your skills: Python, Cloud Computing, Automation
------------------------------------------------------------

Applying TF-IDF vectorization...
TF-IDF matrix shape: (11, 64) (documents x unique words)

Calculating cosine similarity...

============================================================
            TOP 3 RECOMMENDED CAREER ROLES
============================================================
#    Job Role                  Similarity Score
------------------------------------------------------------
1    DevOps Engineer           0.5191
2    Cloud Architect           0.2821
3    Data Scientist            0.0685
============================================================

Recommendation Pipeline:
Input
  |
  v
TF-IDF Vectorization
  |
  v
Cosine Similarity
  |
  v
Sort by similarity
  |
  v
Top 3 Recommendations
  |
  v
Output
```

---

## Reflection

This project introduced the fundamentals of content-based recommendation systems using real, simple techniques:

- **Text becomes math:** TF-IDF converts words into numeric vectors, showing how even basic text analysis can power meaningful recommendations.
- **Similarity is a distance measure:** Cosine similarity compares the "direction" of skill profiles — an intuitive way to measure how compatible a person is with a role.
- **Pipeline thinking:** The flow of input → vectorize → compare → sort → output is the same backbone used by larger recommendation systems.
- **No magic required:** Classic ML techniques like TF-IDF + cosine similarity still produce sensible, explainable results — a good reminder that not everything needs neural networks.
- **Validation matters:** Enforcing a minimum of 3 skills keeps the system meaningful while still being tolerant of uncommon skills.

---

## Project Structure

```
Project-3-AI-Recommendation/
├── recommender.py       # Main recommendation script
├── raw_skills.csv       # Job role / skills dataset
└── README.md            # This file
```