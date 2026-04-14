# 🎓 Student Performance Dashboard

A data-driven web app that turns raw student records into an **early warning system** — helping teachers, tutors, and product teams identify at-risk students before they fail.

🔗 **[Live Demo](https://student-performance-dashboard-yzfsb58fzuuuhocvbf6qhb.streamlit.app/)** 

---

## 💡 Problem It Solves

In a class of 40 students, 8 will fail Math. Most of the time, nobody sees it coming until the report card.

The data was always there — quiz scores, performance patterns, demographic factors. But it was scattered and never acted on. This dashboard connects the dots and surfaces the right insights at the right time.

---

## 🚀 Features

| Tab | What It Does |
|---|---|
| 📊 **Overview** | KPI cards, score distributions, grade breakdown, pass/fail per subject |
| 🔍 **Factor Analysis** | Impact of test prep, gender, lunch type, parental education on scores |
| ⚠️ **At-Risk Students** | Identifies failing students, drop-off points, filterable score table |
| 🤖 **Grade Predictor** | ML model that predicts a student's grade from their profile |

---

## 🛠️ Tech Stack

- **Python** — data processing and ML
- **Streamlit** — web app framework and deployment
- **Plotly** — interactive charts and visualizations
- **Pandas / NumPy** — data manipulation
- **Scikit-learn** — Random Forest grade prediction model

---

## 📊 Dataset

**Students Performance in Exams** — Kaggle  
1,000 student records with scores in Math, Reading, and Writing along with demographic factors like gender, parental education, lunch type, and test preparation status.

---

## 🤖 How AI Helped

I used **Claude AI** as a development partner throughout this project.

- My existing **Jupyter notebook analysis** was converted into a clean, deployable Streamlit app
- The **4-tab product structure** was designed to be actionable, not just analytical
- **Deployment errors** on Streamlit Cloud were debugged in real time

The product thinking and data analysis were mine — AI helped me ship it 3x faster.

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/student-performance-dashboard.git
cd student-performance-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
student-performance-dashboard/
├── app.py                    # Main Streamlit application
├── StudentsPerformance.csv   # Dataset
├── requirements.txt          # Python dependencies
└── README.md                 # You are here
```

---

## 📈 Key Insights from the Data

- **Math** is the weakest subject — highest failure rate across all student groups
- Students who completed **test preparation** score 8-10 points higher on average
- **Standard lunch** students consistently outperform free/reduced lunch students
- **Parental education level** has a measurable impact on student grades

---

## 🔮 What I'd Build Next

- Connect to a **live database** instead of a static CSV
- Add **time-series tracking** to show individual student progress over weeks
- Build a **teacher alert system** — auto-notify when a student crosses the at-risk threshold
- Integrate an **AI tutor recommendation engine** — suggest specific topics to revise based on weak areas

---

## 👤 Author

**[Ketan Tewari]**  
[(https://www.linkedin.com/in/ketantewari)] 
[https://github.com/Ketan2612]

---

*Built as part of the Cuemath AI Builder placement drive — April 2026*
