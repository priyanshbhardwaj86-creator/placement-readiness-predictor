import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# CONCEPT: random_state / seed
# When you generate random numbers, the result changes every run.
# Setting a seed (any fixed number) makes it reproducible —
# you and your teammate get the exact same dataset every time.
# ---------------------------------------------------------------
np.random.seed(42)

NUM_STUDENTS = 500  # how many fake students to generate

# ---------------------------------------------------------------
# STEP 1: Generate each feature (column) randomly
# np.random.uniform(low, high, size) → float between low and high
# np.random.randint(low, high, size) → integer between low and high
# ---------------------------------------------------------------

cgpa = np.round(np.random.uniform(5.0, 10.0, NUM_STUDENTS), 2)
# CGPA between 5.0 and 10.0, rounded to 2 decimal places

num_projects = np.random.randint(0, 6, NUM_STUDENTS)
# Number of projects: 0 to 5

mock_interview_score = np.random.randint(0, 101, NUM_STUDENTS)
# Mock interview score: 0 to 100

internships = np.random.randint(0, 4, NUM_STUDENTS)
# Number of internships: 0 to 3

# Skills: each skill is 0 (no) or 1 (yes)
dsa_skill    = np.random.randint(0, 2, NUM_STUDENTS)
ml_skill     = np.random.randint(0, 2, NUM_STUDENTS)
webdev_skill = np.random.randint(0, 2, NUM_STUDENTS)

# ---------------------------------------------------------------
# STEP 2: Calculate a composite "readiness score" (0 to 100)
# This is the formula WE decide — it reflects what matters most.
# Higher weight = more important feature.
# ---------------------------------------------------------------

readiness_score = (
    (cgpa / 10)                * 25 +   # max 25 points
    (num_projects / 5)         * 20 +   # max 20 points
    (mock_interview_score/100) * 30 +   # max 30 points  ← most important
    (internships / 3)          * 15 +   # max 15 points
    ((dsa_skill + ml_skill + webdev_skill) / 3) * 10  # max 10 points
)
# Total possible: 100 points

# ---------------------------------------------------------------
# STEP 3: Add a small amount of noise (real data is never perfect)
# np.random.normal(mean, std_dev, size) → adds slight randomness
# We clip to keep scores within 0–100
# ---------------------------------------------------------------

noise = np.random.normal(0, 3, NUM_STUDENTS)
readiness_score = np.clip(readiness_score + noise, 0, 100)
readiness_score = np.round(readiness_score, 2)

# ---------------------------------------------------------------
# STEP 4: Convert score → label (classification target)
# pd.cut() splits a continuous value into labeled bins
#   0–40  → "Not Ready"
#  40–65  → "Maybe"
#  65–100 → "Ready"
# ---------------------------------------------------------------

labels = pd.cut(
    readiness_score,
    bins=[0, 40, 65, 100],
    labels=["Not Ready", "Maybe", "Ready"],
    include_lowest=True
)

# ---------------------------------------------------------------
# STEP 5: Assemble everything into a DataFrame
# A DataFrame is like an Excel table — rows = students, cols = features
# ---------------------------------------------------------------

df = pd.DataFrame({
    "cgpa":                 cgpa,
    "num_projects":         num_projects,
    "mock_interview_score": mock_interview_score,
    "internships":          internships,
    "dsa_skill":            dsa_skill,
    "ml_skill":             ml_skill,
    "webdev_skill":         webdev_skill,
    "readiness_score":      readiness_score,   # keep for reference
    "placement_status":     labels             # ← this is our TARGET column
})

# ---------------------------------------------------------------
# STEP 6: Save to CSV
# index=False means don't write row numbers (0,1,2...) into the file
# ---------------------------------------------------------------

output_path = "data_sample/placement_data.csv"
df.to_csv(output_path, index=False)

print(f"Dataset saved to: {output_path}")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())
print("\nLabel distribution (how many students in each class):")
print(df["placement_status"].value_counts())
