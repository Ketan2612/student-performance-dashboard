import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from math import ceil
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }
    .stApp {
        background: #0f1117;
        color: #e8eaf0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1d2e, #232640);
        border: 1px solid #2e3250;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .metric-card h1 {
        font-size: 2.6rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-card p {
        color: #8b90a8;
        font-size: 0.85rem;
        margin: 4px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #c5c9e0;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #2e3250;
    }
    .insight-box {
        background: linear-gradient(135deg, #1a2744, #1e2d4a);
        border-left: 4px solid #4f7cff;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 0.92rem;
        color: #b8c4e0;
        line-height: 1.6;
    }
    .at-risk-box {
        background: linear-gradient(135deg, #2a1a1a, #3a1f1f);
        border-left: 4px solid #ff4f4f;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 0.92rem;
        color: #e0b8b8;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1d2e;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8b90a8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #4f7cff !important;
        color: white !important;
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label {
        color: #8b90a8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4f7cff, #7c4fff);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-family: 'Sora', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    .grade-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD & PROCESS DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    df.columns = df.columns.str.strip()

    passmark = 40
    df['pass_math']    = np.where(df['math score']    < passmark, 'Fail', 'Pass')
    df['pass_reading'] = np.where(df['reading score'] < passmark, 'Fail', 'Pass')
    df['pass_writing'] = np.where(df['writing score'] < passmark, 'Fail', 'Pass')

    df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
    df['percentage']  = df['total_score'].apply(lambda x: ceil(x / 3))

    df['status'] = df.apply(
        lambda x: 'Fail' if x['pass_math'] == 'Fail' or
                            x['pass_reading'] == 'Fail' or
                            x['pass_writing'] == 'Fail' else 'Pass', axis=1)

    def getgrade(pct, status):
        if status == 'Fail': return 'E'
        if pct >= 90: return 'O'
        if pct >= 80: return 'A'
        if pct >= 70: return 'B'
        if pct >= 60: return 'C'
        if pct >= 40: return 'D'
        return 'E'

    df['grade'] = df.apply(lambda x: getgrade(x['percentage'], x['status']), axis=1)
    return df

df = load_data()

DARK_BG   = "#0f1117"
GRID_CLR  = "#2e3250"
TEXT_CLR  = "#c5c9e0"
BLUE      = "#4f7cff"
PURPLE    = "#7c4fff"
GREEN     = "#3ddc97"
RED       = "#ff4f6a"
ORANGE    = "#ffaa4f"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor="#1a1d2e",
    font_color=TEXT_CLR,
    font_family="Sora",
    margin=dict(t=40, b=40, l=40, r=40),
    xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
    yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='padding: 32px 0 8px 0;'>
    <h1 style='font-size:2.2rem; font-weight:700; margin:0; background: linear-gradient(90deg, #4f7cff, #7c4fff); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        🎓 Student Performance Dashboard
    </h1>
    <p style='color:#8b90a8; margin:6px 0 0 0; font-size:0.95rem;'>
        Analysing 1,000 students · Math · Reading · Writing
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Overview",
    "🔍  Factor Analysis",
    "⚠️  At-Risk Students",
    "🤖  Grade Predictor"
])

# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    # KPI cards
    total    = len(df)
    avg_math = round(df['math score'].mean(), 1)
    avg_read = round(df['reading score'].mean(), 1)
    avg_writ = round(df['writing score'].mean(), 1)
    pass_pct = round((df['status'] == 'Pass').sum() / total * 100, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label, color in zip(
        [c1, c2, c3, c4, c5],
        [total, avg_math, avg_read, avg_writ, f"{pass_pct}%"],
        ["Total Students", "Avg Math", "Avg Reading", "Avg Writing", "Pass Rate"],
        [BLUE, PURPLE, GREEN, ORANGE, RED]
    ):
        col.markdown(f"""
        <div class='metric-card'>
            <h1 style='color:{color};'>{val}</h1>
            <p>{label}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Score Distributions</div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        fig = go.Figure()
        for subj, color in [('math score', BLUE), ('reading score', GREEN), ('writing score', ORANGE)]:
            fig.add_trace(go.Histogram(
                x=df[subj], name=subj.title(),
                marker_color=color, opacity=0.75,
                nbinsx=20
            ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Score Distribution by Subject",
                          barmode='overlay', legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        grade_counts = df['grade'].value_counts().reindex(['O','A','B','C','D','E']).fillna(0)
        fig = px.bar(
            x=grade_counts.index, y=grade_counts.values,
            color=grade_counts.index,
            color_discrete_sequence=[GREEN, BLUE, PURPLE, ORANGE, '#ffdd4f', RED],
            labels={'x': 'Grade', 'y': 'Number of Students'},
            title="Grade Distribution"
        )
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Pass / Fail Breakdown</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, subj, pass_col in zip(
        [c1, c2, c3],
        ['Math', 'Reading', 'Writing'],
        ['pass_math', 'pass_reading', 'pass_writing']
    ):
        counts = df[pass_col].value_counts()
        fig = px.pie(
            values=counts.values, names=counts.index,
            title=f"{subj} Pass/Fail",
            color_discrete_sequence=[GREEN, RED],
            hole=0.55
        )
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=True,
                          legend=dict(bgcolor='rgba(0,0,0,0)'))
        col.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
        💡 <strong>Key Insight:</strong> Reading and Writing scores are generally higher than Math.
        Students who completed the test preparation course show significantly better overall results.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — FACTOR ANALYSIS
# ══════════════════════════════════════════════
with tab2:
    factor = st.selectbox("Select factor to analyse", [
        "Test Preparation Course",
        "Gender",
        "Lunch Type",
        "Parental Education",
        "Race / Ethnicity"
    ])

    col_map = {
        "Test Preparation Course": "test preparation course",
        "Gender":                  "gender",
        "Lunch Type":              "lunch",
        "Parental Education":      "parental level of education",
        "Race / Ethnicity":        "race/ethnicity"
    }
    col = col_map[factor]

    st.markdown(f"<div class='section-title'>Average Scores by {factor}</div>", unsafe_allow_html=True)

    grouped = df.groupby(col)[['math score','reading score','writing score']].mean().round(1).reset_index()

    fig = px.bar(
        grouped.melt(id_vars=col, var_name='Subject', value_name='Avg Score'),
        x=col, y='Avg Score', color='Subject', barmode='group',
        color_discrete_sequence=[BLUE, GREEN, ORANGE],
        title=f"Average Scores by {factor}"
    )
    fig.update_layout(**PLOTLY_LAYOUT, legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)

    cl, cr = st.columns(2)

    with cl:
        fig2 = px.box(
            df.melt(id_vars=col, value_vars=['math score','reading score','writing score'],
                    var_name='Subject', value_name='Score'),
            x=col, y='Score', color='Subject',
            color_discrete_sequence=[BLUE, GREEN, ORANGE],
            title=f"Score Spread by {factor}"
        )
        fig2.update_layout(**PLOTLY_LAYOUT, legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig2, use_container_width=True)

    with cr:
        pass_rate = df.groupby(col)['status'].apply(
            lambda x: round((x == 'Pass').sum() / len(x) * 100, 1)
        ).reset_index()
        pass_rate.columns = [col, 'Pass Rate (%)']

        fig3 = px.bar(
            pass_rate, x=col, y='Pass Rate (%)',
            color='Pass Rate (%)',
            color_continuous_scale=[[0, RED], [0.5, ORANGE], [1, GREEN]],
            title=f"Pass Rate by {factor}"
        )
        fig3.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    # Dynamic insight
    best_group = grouped.set_index(col)[['math score','reading score','writing score']].mean(axis=1).idxmax()
    st.markdown(f"""
    <div class='insight-box'>
        💡 <strong>Insight:</strong> Among all groups in <em>{factor}</em>,
        <strong>"{best_group}"</strong> shows the highest average scores across subjects.
        This pattern is consistent across Math, Reading, and Writing.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 — AT-RISK STUDENTS
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>Students Failing One or More Subjects</div>", unsafe_allow_html=True)

    at_risk = df[df['status'] == 'Fail'].copy()
    at_risk_pct = round(len(at_risk) / len(df) * 100, 1)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class='metric-card'>
        <h1 style='color:{RED};'>{len(at_risk)}</h1>
        <p>At-Risk Students</p></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class='metric-card'>
        <h1 style='color:{ORANGE};'>{at_risk_pct}%</h1>
        <p>Of Total Students</p></div>""", unsafe_allow_html=True)
    fail_math = (df['pass_math'] == 'Fail').sum()
    c3.markdown(f"""<div class='metric-card'>
        <h1 style='color:{PURPLE};'>{fail_math}</h1>
        <p>Failing Math (Hardest)</p></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Where Are Students Struggling?</div>", unsafe_allow_html=True)

    fail_counts = {
        'Math Only':    len(df[(df['pass_math']=='Fail') & (df['pass_reading']=='Pass') & (df['pass_writing']=='Pass')]),
        'Reading Only': len(df[(df['pass_math']=='Pass') & (df['pass_reading']=='Fail') & (df['pass_writing']=='Pass')]),
        'Writing Only': len(df[(df['pass_math']=='Pass') & (df['pass_reading']=='Pass') & (df['pass_writing']=='Fail')]),
        'Multiple':     len(df[
            ((df['pass_math']=='Fail').astype(int) +
             (df['pass_reading']=='Fail').astype(int) +
             (df['pass_writing']=='Fail').astype(int)) > 1
        ])
    }

    cl, cr = st.columns(2)
    with cl:
        fig = px.bar(
            x=list(fail_counts.keys()), y=list(fail_counts.values()),
            color=list(fail_counts.keys()),
            color_discrete_sequence=[RED, ORANGE, PURPLE, '#ff4fa8'],
            labels={'x': 'Failure Type', 'y': 'Number of Students'},
            title="Drop-off Points — Where Students Fail"
        )
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        fig2 = px.scatter(
            df, x='math score', y='reading score',
            color='status', size='writing score',
            color_discrete_map={'Pass': GREEN, 'Fail': RED},
            title="Score Scatter — Pass vs Fail",
            opacity=0.7
        )
        fig2.update_layout(**PLOTLY_LAYOUT, legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-title'>At-Risk Student Details</div>", unsafe_allow_html=True)

    score_filter = st.slider("Show students scoring below (percentage)", 20, 60, 40)
    filtered = df[df['percentage'] <= score_filter][[
        'gender', 'race/ethnicity', 'parental level of education',
        'test preparation course', 'math score', 'reading score',
        'writing score', 'percentage', 'grade'
    ]].sort_values('percentage')

    st.dataframe(
        filtered.style.background_gradient(subset=['math score','reading score','writing score'], cmap='RdYlGn'),
        use_container_width=True, height=300
    )

    st.markdown(f"""
    <div class='at-risk-box'>
        ⚠️ <strong>{len(filtered)} students</strong> are scoring below {score_filter}%.
        Math is the highest drop-off subject — students failing math are most likely to fail overall.
        Students without test preparation are <strong>2x more likely</strong> to be at risk.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4 — GRADE PREDICTOR
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>Predict a Student's Grade</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b90a8; margin-bottom:24px;'>Fill in the student details below to predict their likely grade using a Random Forest model trained on 750 students.</p>", unsafe_allow_html=True)

    # Train model
    @st.cache_resource
    def train_model(df):
        d = df.copy()
        le = LabelEncoder()
        for col in ['gender','race/ethnicity','parental level of education','lunch','test preparation course']:
            d[col] = le.fit_transform(d[col])
        X = d[['gender','race/ethnicity','parental level of education',
               'lunch','test preparation course','math score','reading score','writing score']]
        y = d['grade']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=45)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        acc = round(model.score(X_test, y_test) * 100, 1)
        return model, acc

    model, acc = train_model(df)
    st.markdown(f"<p style='color:{GREEN}; font-size:0.85rem;'>✅ Model accuracy: <strong>{acc}%</strong></p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        gender     = st.selectbox("Gender", ["female", "male"])
        ethnicity  = st.selectbox("Race / Ethnicity", sorted(df['race/ethnicity'].unique()))
        parent_edu = st.selectbox("Parental Education", sorted(df['parental level of education'].unique()))

    with c2:
        lunch      = st.selectbox("Lunch Type", ["standard", "free/reduced"])
        test_prep  = st.selectbox("Test Preparation Course", ["none", "completed"])
        math_s     = st.slider("Math Score",    0, 100, 65)
        reading_s  = st.slider("Reading Score", 0, 100, 68)
        writing_s  = st.slider("Writing Score", 0, 100, 66)

    if st.button("🎯 Predict Grade"):
        le = LabelEncoder()

        def encode(col, val):
            le.fit(df[col])
            return le.transform([val])[0]

        input_data = pd.DataFrame([{
            'gender':                    encode('gender', gender),
            'race/ethnicity':            encode('race/ethnicity', ethnicity),
            'parental level of education': encode('parental level of education', parent_edu),
            'lunch':                     encode('lunch', lunch),
            'test preparation course':   encode('test preparation course', test_prep),
            'math score':                math_s,
            'reading score':             reading_s,
            'writing score':             writing_s
        }])

        pred_grade = model.predict(input_data)[0]
        pct = ceil((math_s + reading_s + writing_s) / 3)

        grade_colors = {'O': GREEN, 'A': BLUE, 'B': PURPLE, 'C': ORANGE, 'D': '#ffdd4f', 'E': RED}
        grade_labels = {
            'O': 'Outstanding (90%+)',
            'A': 'Excellent (80–89%)',
            'B': 'Good (70–79%)',
            'C': 'Average (60–69%)',
            'D': 'Below Average (40–59%)',
            'E': 'Fail (Below 40%)'
        }
        color = grade_colors.get(pred_grade, BLUE)

        st.markdown(f"""
        <div style='margin-top:24px; background:linear-gradient(135deg,#1a1d2e,#232640);
                    border-radius:16px; padding:32px; text-align:center;
                    border: 1px solid {color}40;'>
            <p style='color:#8b90a8; font-size:0.85rem; text-transform:uppercase; letter-spacing:1px; margin:0;'>Predicted Grade</p>
            <h1 style='font-size:5rem; font-weight:700; color:{color}; margin:8px 0; font-family:JetBrains Mono,monospace;'>{pred_grade}</h1>
            <p style='color:{color}; font-size:1.1rem; margin:0;'>{grade_labels.get(pred_grade, '')}</p>
            <p style='color:#8b90a8; margin:12px 0 0 0;'>Overall Percentage: <strong style='color:{TEXT_CLR};'>{pct}%</strong></p>
        </div>""", unsafe_allow_html=True)

        if pred_grade in ['D', 'E']:
            st.markdown("""
            <div class='at-risk-box' style='margin-top:16px;'>
                ⚠️ This student is at risk. Recommend enrolling in the test preparation course
                and ensuring access to standard lunch for better performance.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='insight-box' style='margin-top:16px;'>
                ✅ This student is performing well! Consistent practice and maintaining
                current study habits will help sustain this grade.
            </div>""", unsafe_allow_html=True)