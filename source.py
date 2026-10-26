#!/usr/bin/env python
# coding: utf-8

# # Early Warning Signs for Students at Risk of Failing 📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# Schools often find out a student is failing only when final grades come out. At that point it is too late to help. If teachers and counselors could spot warning signs early in the year, they could step in with tutoring, parent meetings, or extra support.
# 
# This matters because failing a course is one of the strongest predictors of dropping out, and dropping out affects income, health, and job chances for life. Portugal, where this data comes from, had one of the highest school failure rates in Europe when the data was collected. Underage drinking is also a public health concern, so it is worth checking whether it is a real warning sign.
# 
# **Stakeholder:** a school counselor who wants to know which students to check on early in the year.

# In[1]:


get_ipython().run_line_magic('pip', 'install pandas matplotlib seaborn lxml ucimlrepo')


# ## Project Question
# 1. Which early signals best predict failing the year: first period grade (G1), absences, past failures, study time, or alcohol use?
# 2. How early can we tell? Does the G1 grade already separate students who pass from students who fail?
# 3. Do social habits like going out and weekend drinking still matter once we account for study time and past failures?

# ## What would an answer look like?
# * A ranked bar chart showing which signals are most related to failing (final grade G3 below 10 out of 20).
# * A line chart of fail rate by G1 grade range, showing where the risk jumps.
# * A simple watch list rule, for example: students with 1 or more past failures and G1 below 10 fail X percent of the time.
# 
# **Limitations:** the data covers only two schools in Portugal, habits are self reported, and the results show links, not proof of cause.

# ## Data Sources
# | # | Source | Type | Description |
# |---|---|---|---|
# | 1 | Kaggle: Student Alcohol Consumption (student-mat.csv) | File (CSV) | 395 students, math class |
# | 2 | UCI Machine Learning Repository: Student Performance (id 320), loaded with the ucimlrepo package | API | 649 students, Portuguese language class |
# | 3 | Wikipedia: Academic grading in Portugal | Scraped web page | What each range of the 0 to 20 grading scale means |
# 
# Sources 1 and 2 come from the same study (Cortez and Silva, 2008) of two high schools in Portugal. Source 1 is loaded from the Kaggle file and source 2 through the UCI API. They cover different classes and are linked by student attributes.
# 
# ### How the datasets relate
# * **Math and Portuguese:** many students took both classes. They are matched on school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, nursery, and internet.
# * **Grading scale:** joined on the final grade (G3) by grade range, which turns a number like 14 into a label like "Good" and confirms that below 10 means failing.

# In[2]:


import pandas as pd
import requests
from io import StringIO
from ucimlrepo import fetch_ucirepo

math_df = pd.read_csv("data/student-mat.csv", sep=",")
print(math_df.shape)
math_df.head()


# In[3]:


student_performance = fetch_ucirepo(id=320)
por_df = pd.concat(
    [student_performance.data.features, student_performance.data.targets],
    axis=1,
)
print(por_df.shape)
por_df.head()


# In[4]:


url = "https://en.wikipedia.org/wiki/Academic_grading_in_Portugal"
headers = {"User-Agent": "IT4063C student project"}
html = requests.get(url, headers=headers, timeout=30).text
tables = pd.read_html(StringIO(html))
print(f"Found {len(tables)} tables")
for i, table in enumerate(tables):
    print(i, table.shape)
    display(table.head())
grading_df = tables[1].copy()
grade_numbers = grading_df["Grade"].astype(str).str.findall(r"\d+(?:\.\d+)?")


# In[5]:


grading_df["max_grade"] = grade_numbers.str[0].astype(float)
grading_df["min_grade"] = grade_numbers.str[-1].astype(float)
grading_df


# In[6]:


merge_keys = ["school", "sex", "age", "address", "famsize", "Pstatus",
              "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]
both_df = math_df.merge(por_df, on=merge_keys, suffixes=("_mat", "_por"))
print(f"Students in both classes: {len(both_df)}")

math_labeled = pd.merge_asof(
    math_df.assign(G3=math_df["G3"].astype(float)).sort_values("G3"),
    grading_df[["min_grade", "Qualification"]].sort_values("min_grade"),
    left_on="G3",
    right_on="min_grade",
    direction="backward",
)
math_labeled["Qualification"].value_counts()


# ## Approach and Analysis
# 1. Load the three sources and merge the math and Portuguese data on student attributes.
# 2. Create a pass or fail label (G3 below 10 is fail) and add grade labels from the grading scale table.
# 3. Compare fail rates across each warning signal (G1 range, absences, past failures, study time, alcohol use) with bar and line charts.
# 4. Check which signals still matter when looked at together, using correlation and a simple logistic regression.
# 5. Turn the strongest signals into a simple watch list rule a counselor could use.

# # Checkpoint 2
# 
# ## Exploratory Data Analysis (EDA)
# To look at both classes together, I stack the math and Portuguese data into one table called `students` and add a `course` column. I also add a `failed` column, where 1 means the final grade (G3) is below 10 out of 20, which is failing according to the Portuguese grading scale.

# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

students = pd.concat(
    [math_df.assign(course="Math"), por_df.assign(course="Portuguese")],
    ignore_index=True,
)
students["failed"] = (students["G3"] < 10).astype(int)

print(students.shape)
students.groupby("course")["failed"].agg(["size", "mean"]).round(3)


# In[8]:


print("Data types:")
print(students.dtypes.value_counts())
print("\nMissing values:", students.isna().sum().sum())
print("Duplicate rows in math:", math_df.duplicated().sum())
print("Duplicate rows in Portuguese:", por_df.duplicated().sum())

students[["age", "absences", "failures", "G1", "G2", "G3"]].describe().round(2)


# **First look at data quality:** there are no missing values and no duplicate rows in either file. 17 columns are stored as text, and 8 of those are yes/no answers that should become 0/1 numbers. The summary also shows two things to look into: some students have a final grade of 0, and absences go up to 75 while the median is only 2.

# In[9]:


plt.figure(figsize=(10, 5))
sns.histplot(data=students, x="G3", hue="course", bins=21)
plt.axvline(9.5, color="red", linestyle="--")
plt.title("Final grade (G3) distribution")
plt.xlabel("Final grade (0 to 20)")
plt.show()


# **Visualization 1: Final grade distribution.** Most grades fall between 8 and 15, and the red line shows the pass line of 10. Many more math students fall left of it than Portuguese students. The most important finding is the separate bar at 0: 38 math students and 15 Portuguese students have a final grade of 0. This looks like a data issue, which I investigate in the cleaning section.

# In[10]:


students["G1_band"] = pd.cut(
    students["G1"],
    bins=[-1, 7, 9, 11, 13, 20],
    labels=["0-7", "8-9", "10-11", "12-13", "14-20"],
)

plt.figure(figsize=(10, 5))
sns.barplot(data=students, x="G1_band", y="failed", hue="course", errorbar=None)
plt.title("Fail rate by first period grade (G1)")
plt.xlabel("First period grade range (G1)")
plt.ylabel("Share of students who failed the year")
plt.show()

students.groupby(["course", "G1_band"], observed=True)["failed"].mean().unstack().round(2)


# **Visualization 2: Fail rate by first period grade.** This answers my second question: yes, the first period grade already separates most students. Students with a G1 of 0 to 7 failed 90% of the time in math and 84% in Portuguese. Students with a G1 of 12 or higher almost never failed. The risky middle group is 8 to 11, which is where a counselor would get the most value from checking in early.

# In[11]:


signals = ["failed", "G1", "failures", "absences", "studytime",
           "goout", "Dalc", "Walc", "Medu", "Fedu", "age"]

plt.figure(figsize=(11, 8))
sns.heatmap(students[signals].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation between early signals and failing")
plt.show()


# **Visualization 3: Correlation heatmap.** The strongest link to failing is the first period grade (G1, -0.61), followed by past failures (0.37). Study time, going out, and alcohol use (Dalc, Walc) have only weak correlations with failing, between 0.08 and 0.11 in absolute value. Weekday and weekend drinking are strongly related to each other (0.63), so a model probably only needs one of them. Parents' education (Medu and Fedu) is also closely related (0.64).

# In[12]:


failures_rate = students.groupby(["failures", "course"])["failed"].mean().unstack()

failures_rate.plot(kind="bar", figsize=(10, 5), rot=0)
plt.title("Fail rate by number of past class failures")
plt.xlabel("Past class failures")
plt.ylabel("Fail rate")
plt.show()

failures_rate.round(2)


# **Visualization 4: Fail rate by past failures (pandas and matplotlib).** Past failures are a strong warning sign. In math, 25% of students with no past failures failed the year, compared to 82% of students with 2 past failures. Portuguese shows the same pattern, from 9% to 50%. Together with the heatmap, this partly answers my third question: social habits like going out and drinking matter a little, but much less than grades and past failures.

# In[13]:


plt.figure(figsize=(10, 4))
sns.boxplot(data=students, x="absences", y="course")
plt.title("School absences by course")
plt.xlabel("Number of absences")
plt.ylabel("")
plt.show()

print("Students with more than 30 absences:", (students["absences"] > 30).sum())


# **Visualization 5: Absences box plot.** Absences are very skewed. Most students have under 10, but a few have 40, 56, and even 75. Only 6 students have more than 30 absences. These values are possible, so they are not errors, but they can pull averages and models in one direction, so I cap them in the cleaning section.

# In[14]:


plt.figure(figsize=(7, 7))
sns.scatterplot(data=both_df, x="G3_mat", y="G3_por", alpha=0.5)
plt.axvline(9.5, color="red", linestyle="--")
plt.axhline(9.5, color="red", linestyle="--")
plt.title("Math vs Portuguese final grade (students in both classes)")
plt.xlabel("Math final grade (G3)")
plt.ylabel("Portuguese final grade (G3)")
plt.show()

print("Correlation:", round(both_df["G3_mat"].corr(both_df["G3_por"]), 2))


# **Visualization 6: Math vs Portuguese grades.** For the 382 students in both classes, the correlation is 0.48. Students who do well in one class usually do well in the other, but not always. Many students pass Portuguese and fail math (the top left area), so a student's risk should be checked per class, not just per student. The column of points at math grade 0 shows the same 0 grade issue from Visualization 1.

# ## Data Cleaning and Transformation
# 
# ### Missing and duplicate values

# In[15]:


print("Missing values:", students.isna().sum().sum())
print("Full duplicate rows:", students.duplicated().sum())
print("Math rows sharing the 13 merge keys:", math_df.duplicated(subset=merge_keys).sum())
print("Portuguese rows sharing the 13 merge keys:", por_df.duplicated(subset=merge_keys).sum())


# There are no missing values and no full duplicate rows, so nothing needs to be filled or removed. However, 4 math rows and 12 Portuguese rows share the same 13 values I use to merge the two classes. These are different students who happen to look the same on those columns, not duplicates. It means the merge is not a perfect student ID, which is a limitation of this dataset that I note for the analysis.
# 
# ### Anomaly: final grade of 0

# In[16]:


zero_g3 = students[students["G3"] == 0]
print("Students with a final grade of 0:", len(zero_g3))
print("Highest absences in this group:", zero_g3["absences"].max())
display(zero_g3[["course", "G1", "G2", "G3", "absences"]].head())

students["no_final_exam"] = ((students["G3"] == 0) & (students["absences"] == 0)).astype(int)
students.groupby("course")["no_final_exam"].sum()


# All 53 students with a final grade of 0 also have 0 absences, and they had real grades earlier in the year (G1 up to 12). A student who never missed class but scored 0 does not make sense. The most likely meaning is that these students left the class or did not take the final exam, and the absences were not recorded after that.
# 
# I am keeping these students because leaving a class is exactly the outcome a counselor wants to prevent, so they still count as failed. I added a `no_final_exam` flag so I can test my results with and without them later.
# 
# ### Outliers: absences

# In[17]:


cap = students["absences"].quantile(0.99)
students["absences_capped"] = students["absences"].clip(upper=cap)

print(f"99th percentile of absences: {cap:.1f}")
print("Rows capped:", (students["absences"] > cap).sum())


# I capped absences at the 99th percentile (about 26). This changes only 11 rows. The original `absences` column is kept, and `absences_capped` will be used for modeling, so a few extreme values do not dominate the results.
# 
# ### Data type changes

# In[18]:


yes_no_cols = ["schoolsup", "famsup", "paid", "activities",
               "nursery", "higher", "internet", "romantic"]
for col in yes_no_cols:
    students[col] = (students[col] == "yes").astype(int)

nominal_cols = ["school", "sex", "address", "famsize", "Pstatus",
                "Mjob", "Fjob", "reason", "guardian", "course"]
students[nominal_cols] = students[nominal_cols].astype("category")

# Grade ranges come from the scraped grading table (grading_df)
students["grade_label"] = pd.cut(
    students["G3"],
    bins=[0, 3.5, 9.5, 13.5, 15.5, 17.5, 20.01],
    labels=["Poor", "Weak", "Sufficient", "Good", "Very good", "Excellent"],
    right=False,
)

print(students.shape, "missing:", students.isna().sum().sum())
students[["course", "G3", "grade_label", "failed", "no_final_exam", "absences_capped"]].sample(8, random_state=42)


# I made three type changes:
# * The 8 yes/no columns became 1 and 0 so they can be used in calculations and models.
# * The 10 text columns with no order (like school, job, and course) became the `category` type.
# * The final grade now has a label from the scraped Portuguese grading scale (Poor, Weak, Sufficient, Good, Very good, Excellent). The labels line up exactly with my `failed` column, since Poor and Weak are the failing labels, which confirms the pass line of 10.
# 
# The cleaned table has 1,044 rows, 39 columns, and no missing values.

# ## Machine Learning Plan
# 
# **Model type:** this is a classification problem. The target is `failed` (1 or 0). I plan to start with logistic regression because it is simple and easy to explain to a counselor, then compare it with a decision tree and a random forest.
# 
# **Features:** only information a school has early in the year: G1, past failures, absences, study time, going out, alcohol use, parents' education, and support columns. I will not use G2, because it comes too late in the year to give an early warning.
# 
# **Challenges I expect:**
# 1. **Imbalanced classes.** Only 15% of Portuguese students and 33% of math students failed. A model could look accurate by predicting "pass" for everyone.
# 2. **Small dataset.** There are only 395 math students, so a complex model could memorize the data instead of learning from it.
# 3. **The grade 0 group.** Students who likely left the class may look different from students who failed normally.
# 4. **Text columns.** Columns like Mjob and reason need to be converted into numbers.
# 
# **How I plan to address them:**
# 1. Use recall and F1 score instead of only accuracy, and try `class_weight="balanced"`, because missing a student who will fail is worse than checking in on a student who passes.
# 2. Use cross validation and keep the models simple.
# 3. Train with and without the `no_final_exam` students and compare the results.
# 4. Use one hot encoding (`pd.get_dummies`) for the category columns.

# ## Prior Feedback and Updates
# **Feedback received:** [write the feedback you got from your peer reviewer or the instructor here]
# 
# **Changes made:**
# * Added a combined `students` table so both classes can be compared in every chart.
# * Explained clearly that the math and Portuguese files come from the same study and how they are linked.
# * Added a data cleaning section and a plan for the machine learning model.

# ## Resources and References
# * Cortez, P. and Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance.
# * UCI Machine Learning Repository, Student Performance: https://archive.ics.uci.edu/dataset/320/student+performance
# * Kaggle, Student Alcohol Consumption: https://www.kaggle.com/datasets/uciml/student-alcohol-consumption
# * Wikipedia, Academic grading in Portugal: https://en.wikipedia.org/wiki/Academic_grading_in_Portugal

# In[19]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

