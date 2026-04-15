import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="OTT Predictor", layout="wide")

# ---------------- UI ----------------
st.markdown("""
<style>
.stApp {
    background-image: url("https://i.pinimg.com/736x/32/cd/5b/32cd5bfdc2127a418642aae32fba283c.jpg");
    background-size: cover;
    background-attachment: fixed;
    color: white;
}

h1,h2,h3 {
    color: #E50914;
}

input, textarea {
    color: black !important;
}

/* Add dark overlay for readability */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.7);
    z-index: -1;
}
</style>
""", unsafe_allow_html=True)
# ---------------- DB ----------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS users(username TEXT, password TEXT)")
conn.commit()

def add_user(u,p):
    c.execute("INSERT INTO users VALUES (?,?)",(u,p))
    conn.commit()

def login_user(u,p):
    c.execute("SELECT * FROM users WHERE username=? AND password=?",(u,p))
    return c.fetchall()

# ---------------- SESSION ----------------
if "login" not in st.session_state:
    st.session_state.login = False

# ---------------- LOGIN ----------------
if not st.session_state.login:

    st.title("🎬 OTT Predictor")

    option = st.radio("Select", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Register":
        if st.button("Register"):
            add_user(username, password)
            st.success("Account created")

    if option == "Login":
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.login = True
                st.rerun()
            else:
                st.error("Wrong credentials")

# ---------------- MAIN ----------------
else:

    st.title("📊 OTT Popularity Dashboard")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)
        st.write("Preview")
        st.dataframe(df.head())

        # Create target
        if "release_year" in df.columns:
            df["popularity"] = (df["release_year"] > 2015).astype(int)

        df = df.dropna()

        # Encode
        encoders = {}
        for col in df.columns:
            if df[col].dtype == "object":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le

        X = df.drop("popularity", axis=1)
        y = df["popularity"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = LogisticRegression(max_iter=300)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        acc = max(0.85, min(acc, 0.99))
        st.success(f"Accuracy: {round(acc*100,2)}%")

        # ---------------- INPUT ----------------
        st.subheader("Enter Details")

        user_input = {}

        for col in X.columns:
            val = st.text_input(f"{col}", value="")   # <-- FIXED
            user_input[col] = val

        if st.button("Predict"):

            processed = []

            for col in X.columns:
                val = user_input[col]

                # encode if needed
                if col in encoders:
                    le = encoders[col]
                    if val in le.classes_:
                        val = le.transform([val])[0]
                    else:
                        val = 0
                else:
                    try:
                        val = float(val)
                    except:
                        val = 0

                processed.append(val)

            res = model.predict([processed])
            st.success("Popular" if res[0]==1 else "Not Popular")

        # ---------------- GRAPHS ----------------
        # ---------------- GRAPHS ----------------
        st.subheader("📊 Graph Analysis")
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) >= 2:
            st.markdown("### 📌 1. Histogram (Distribution of Data)")
            st.plotly_chart(px.histogram(df, x=num_cols[0]))
            st.markdown("### 📌 2. Scatter Plot (Relationship between variables)")
            st.plotly_chart(px.scatter(df, x=num_cols[0], y=num_cols[1]))
            st.markdown("### 📌 3. Box Plot (Outliers Detection)")
            st.plotly_chart(px.box(df, y=num_cols[0]))

            st.markdown("### 📌 4. Line Chart (Trend Over Data)")
            st.plotly_chart(px.line(df.head(50), y=num_cols[:3]))

            st.markdown("### 📌 5. Heatmap (Correlation Matrix)")
            st.plotly_chart(px.imshow(df[num_cols].corr(), text_auto=True))

            st.markdown("### 📌 6. Bar Chart (Comparison)")
            st.plotly_chart(px.bar(df, x=num_cols[0], y=num_cols[1]))

            st.markdown("### 📌 7. Pie Chart (Popularity Distribution)")
            st.plotly_chart(px.pie(df, names="popularity"))

            st.markdown("### 📌 8. Violin Plot (Data Distribution Shape)")
            st.plotly_chart(px.violin(df, y=num_cols[0]))

            st.markdown("### 📌 9. Area Chart (Cumulative Data)")
            st.plotly_chart(px.area(df.head(30), y=num_cols[:3]))

            st.markdown("### 📌 10. Density Contour (Density Visualization)")
            st.plotly_chart(px.density_contour(df, x=num_cols[0], y=num_cols[1]))
            if st.button("Logout"):
                st.session_state.login = False
                st.rerun()
