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

# In[3]:


import pandas as pd
import requests
from io import StringIO
from ucimlrepo import fetch_ucirepo

math_df = pd.read_csv("data/student-mat.csv", sep=",")
print(math_df.shape)
math_df.head()


# In[4]:


student_performance = fetch_ucirepo(id=320)
por_df = pd.concat(
    [student_performance.data.features, student_performance.data.targets],
    axis=1,
)
print(por_df.shape)
por_df.head()


# In[ ]:


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


# In[7]:


grading_df["max_grade"] = grade_numbers.str[0].astype(float)
grading_df["min_grade"] = grade_numbers.str[-1].astype(float)
grading_df


# In[8]:


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

# In[ ]:


# Start your code here


# ## Resources and References
# * Cortez, P. and Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance.
# * UCI Machine Learning Repository, Student Performance: https://archive.ics.uci.edu/dataset/320/student+performance
# * Kaggle, Student Alcohol Consumption: https://www.kaggle.com/datasets/uciml/student-alcohol-consumption
# * Wikipedia, Academic grading in Portugal: https://en.wikipedia.org/wiki/Academic_grading_in_Portugal

# In[10]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

