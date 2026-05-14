from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import re
import nltk
import pickle
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)


df = pd.read_csv("clean_kialo.csv", on_bad_lines='skip')

# CLEAN DATA
df = df[df['content'].notna()]                 # remove empty
df = df[df['content'].str.len() > 20]          # remove short junk like "5"
df = df[df['content'].str.contains('[a-zA-Z]')] # keep real sentences


# LOAD DATASET

df = df.dropna()

# IMPORTANT: adjust column names if needed
# (we assume these for now)
TEXT_COLUMN = "argument"
STANCE_COLUMN = "stance"



# -----------------------------
# LOAD YOUR TRAINED MODEL
# -----------------------------
with open("stance_model.pkl", "rb") as f:
    stance_model = pickle.load(f)

def predict_stance(topic, content):
    text = f"{topic} {content}"
    return stance_model.predict([text])[0]

# -----------------------------
# SAFE NLTK DOWNLOAD
# -----------------------------
def safe_nltk_download(resource_name):
    try:
        nltk.data.find(resource_name)
    except LookupError:
        nltk.download(resource_name.split('/')[-1])

safe_nltk_download('tokenizers/punkt')
safe_nltk_download('corpora/stopwords')

stop_words = set(stopwords.words('english'))

# -----------------------------
# DATABASE CONNECTION
# -----------------------------
def get_db():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn

# -----------------------------
# DATABASE SETUP
# -----------------------------
def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT UNIQUE
        )
    ''')

    c.execute('''
    CREATE TABLE IF NOT EXISTS replies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        argument_id INTEGER,
        content TEXT,
        side TEXT,
        created_at TEXT
    )
''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS arguments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id INTEGER,
            side TEXT,
            predicted_side TEXT,
            content TEXT,
            username TEXT, 
            relevance_score REAL,
            evidence_score INTEGER,
            upvotes INTEGER DEFAULT 0,
            downvotes INTEGER DEFAULT 0,
            strength_score REAL DEFAULT 0,
            toxicity_label TEXT DEFAULT 'Low',
            fallacy_label TEXT DEFAULT 'None',
            fallacy_explanation TEXT DEFAULT 'No logical fallacy detected.',
            quality_label TEXT DEFAULT 'Medium',
            created_at TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            argument_id INTEGER,
            content TEXT,
            created_at TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            argument_id INTEGER,
            voter_ip TEXT,
            vote_type TEXT,
            UNIQUE(argument_id, voter_ip)
        )
    ''')

    c.execute("CREATE INDEX IF NOT EXISTS idx_topic ON arguments(topic_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_strength ON arguments(strength_score)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_comment_arg ON comments(argument_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_votes_arg ON votes(argument_id)")

    conn.commit()

    # add missing column for old databases
    try:
        c.execute("ALTER TABLE arguments ADD COLUMN fallacy_explanation TEXT DEFAULT 'No logical fallacy detected.'")
        conn.commit()
    except:
        pass

    conn.close()

init_db()

# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

# -----------------------------
# RELEVANCE CALCULATION
# -----------------------------
def calculate_relevance(topic, argument):
    topic = topic.lower()
    argument = argument.lower()

    common_words = set(topic.split()) & set(argument.split())

    if len(common_words) == 0:
        return 0.10

    return round(len(common_words) / len(topic.split()), 2)

# -----------------------------
# EVIDENCE SCORE
# -----------------------------
def calculate_evidence_score(text):
    score = 0

    if re.search(r'\d+', text):
        score += 1

    keywords = [
        "according", "study", "research", "data",
        "report", "statistics", "survey", "analysis", "published",
        "evidence", "source", "journal", "because", "since"
    ]

    text_clean = preprocess_text(text)
    tokens = text_clean.split()

    for word in keywords:
        if word in tokens:
            score += 1

    if "http" in text or "www." in text:
        score += 1

    return score

# -----------------------------
# AI CONTRIBUTION
# -----------------------------
def detect_reasoning(text):
    keywords = ["because", "therefore", "thus", "since", "so", "hence"]
    for word in keywords:
        if word in text.lower():
            return 1
    return 0

def custom_ai_score(relevance, evidence, reasoning, toxicity_penalty, fallacy_penalty, upvotes=0, downvotes=0):
    net_votes = upvotes - downvotes
    score = (
        (relevance * 50)
        + (evidence * 10)
        + (reasoning * 20)
        + (net_votes * 3)
        - toxicity_penalty
        - fallacy_penalty
    )
    return round(score, 2)

def get_quality_label(score):
    if score >= 45:
        return "Strong"
    elif score >= 20:
        return "Medium"
    return "Weak"

# -----------------------------
# TOXICITY DETECTOR
# -----------------------------
def detect_toxicity(text):
    toxic_words = [
        "idiot", "stupid", "dumb", "hate", "nonsense", "trash",
        "shut up", "moron", "ridiculous", "useless"
    ]

    lowered = text.lower()
    toxic_score = 0

    for word in toxic_words:
        if word in lowered:
            toxic_score += 2

    if text.count("!") >= 3:
        toxic_score += 1

    uppercase_words = [w for w in text.split() if len(w) > 2 and w.isupper()]
    if len(uppercase_words) >= 2:
        toxic_score += 1

    if toxic_score >= 4:
        return "High", 15
    elif toxic_score >= 2:
        return "Medium", 8
    return "Low", 0

# -----------------------------
# ADVANCED FALLACY DETECTOR
# -----------------------------
def detect_fallacy(text):
    lowered = text.lower()

    if any(x in lowered for x in ["idiot", "stupid", "dumb", "moron", "lazy people"]):
        return "Ad Hominem", "This argument attacks a person or group instead of addressing the actual issue.", 12

    if any(x in lowered for x in ["everyone", "nobody", "always", "never", "all people"]):
        return "Hasty Generalization", "This argument makes a broad claim without enough evidence.", 10

    if any(x in lowered for x in ["either you", "only two choices", "if not this then that"]):
        return "False Dilemma", "This argument presents only two options when more possibilities may exist.", 10

    if any(x in lowered for x in ["this will lead to", "soon everyone will", "eventually society will collapse", "will lead to disaster"]):
        return "Slippery Slope", "This argument assumes extreme consequences without clearly proving them.", 10

    if any(x in lowered for x in ["think of the children", "heartless", "cruel", "shame on"]):
        return "Appeal to Emotion", "This argument tries to persuade using emotion instead of strong logic.", 8

    if "because of this" in lowered and "therefore" in lowered:
        return "False Cause", "This argument assumes a cause-and-effect relationship without enough proof.", 8

    return "None", "No logical fallacy detected.", 0



# -----------------------------
# BIAS DETECTOR (NEW)
# -----------------------------

def detect_bias(text):
    lowered = text.lower()

    bias_words = [
        "obviously", "clearly", "everyone", "nobody", "always", "never",
        "all", "only", "must", "completely", "definitely", "undoubtedly"
    ]

    bias_phrases = [
        "no doubt", "of course", "it is certain", "without question"
    ]

    # check single words
    for word in bias_words:
        if f" {word} " in f" {lowered} ":
            return "Biased", f"The argument uses strong subjective word '{word}', indicating bias."

    # check phrases
    for phrase in bias_phrases:
        if phrase in lowered:
            return "Biased", f"The argument uses strong subjective phrase '{phrase}', indicating bias."

    return "Neutral", "No strong bias detected."



# -----------------------------
# AUTO ARGUMENT GENERATOR (NEW)
# -----------------------------


def generate_argument(topic, side):
    topic = str(topic).strip().lower()
    side = str(side).strip().lower()

    # clean csv data
    df_ai["topic"] = df_ai["topic"].astype(str).str.strip().str.lower()
    df_ai["side"] = df_ai["side"].astype(str).str.strip().str.lower()
    df_ai["content"] = df_ai["content"].astype(str).str.strip()

    # find same topic + same side
    topic_rows = df_ai[
        (df_ai["topic"].str.contains(topic, na=False)) &
        (df_ai["side"] == side)
    ]

    # if exact/similar topic found
    if not topic_rows.empty:
        valid = topic_rows[topic_rows["content"].str.len() > 20]["content"].dropna().tolist()
        if valid:
            return random.choice(valid)

    # fallback: any argument from same side
    side_rows = df_ai[df_ai["side"] == side]
    valid = side_rows[side_rows["content"].str.len() > 20]["content"].dropna().tolist()

    if valid:
        return random.choice(valid)

    return "No generated argument found from dataset."



# AI REPLY


import pandas as pd
import random

df_ai = pd.read_csv("clean_kialo.csv")

def get_ai_reply(topic, user_side):

    topic = topic.lower().strip()

    # flexible topic match
    df_topic = df_ai[df_ai["topic"].str.lower().str.contains(topic, na=False)]

    if df_topic.empty:
        return "No matching topic found in dataset."

    # choose opposite side
    opposite = "con" if user_side.lower() == "pro" else "pro"

    df_opposite = df_topic[df_topic["side"] == opposite]

    if not df_opposite.empty:
        return random.choice(df_opposite["content"].dropna().tolist())

    # fallback
    return random.choice(df_topic["content"].dropna().tolist())

# -----------------------------
# HOME PAGE
# -----------------------------


@app.route('/')
def index():
    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT * FROM topics ORDER BY id DESC")
    topics = c.fetchall()

    c.execute("""
        SELECT arguments.*, topics.title
        FROM arguments
        JOIN topics ON arguments.topic_id = topics.id
        ORDER BY arguments.strength_score DESC, arguments.id DESC
    """)
    arguments = c.fetchall()

    updated_arguments = []

    for arg in arguments:
        arg = dict(arg)
        arg["ai_reply"] = get_ai_reply(arg["title"], arg["side"])
        updated_arguments.append(arg)

    arguments = updated_arguments

    c.execute("SELECT * FROM comments ORDER BY id DESC")
    comments = c.fetchall()

    c.execute("SELECT * FROM replies ORDER BY id DESC")
    replies = c.fetchall()

    conn.close()

    return render_template(
        'index.html',
        topics=topics,
        arguments=arguments,
        comments=comments,
        replies=replies
    )



# -----------------------------
# LEADERBOARD PAGE
# -----------------------------
@app.route('/leaderboard')
def leaderboard():
    conn = get_db()
    c = conn.cursor()

    c.execute("""
        SELECT arguments.*, topics.title
        FROM arguments
        JOIN topics ON arguments.topic_id = topics.id
        ORDER BY arguments.strength_score DESC
        LIMIT 10
    """)
    top_arguments = c.fetchall()

    conn.close()
    return render_template("leaderboard.html", top_arguments=top_arguments)

# -----------------------------
# ANALYTICS PAGE
# -----------------------------
@app.route('/analytics')
def analytics():
    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) AS total_topics FROM topics")
    total_topics = c.fetchone()["total_topics"]

    c.execute("SELECT COUNT(*) AS total_arguments FROM arguments")
    total_arguments = c.fetchone()["total_arguments"]

    c.execute("SELECT COUNT(*) AS total_comments FROM comments")
    total_comments = c.fetchone()["total_comments"]

    c.execute("SELECT COUNT(*) AS pro_count FROM arguments WHERE predicted_side='pro'")
    pro_count = c.fetchone()["pro_count"]

    c.execute("SELECT COUNT(*) AS con_count FROM arguments WHERE predicted_side='con'")
    con_count = c.fetchone()["con_count"]

    c.execute("SELECT COUNT(*) AS toxic_high FROM arguments WHERE toxicity_label='High'")
    toxic_high = c.fetchone()["toxic_high"]

    c.execute("SELECT COUNT(*) AS fallacy_count FROM arguments WHERE fallacy_label!='None'")
    fallacy_count = c.fetchone()["fallacy_count"]

    conn.close()

    return render_template(
        "analytics.html",
        total_topics=total_topics,
        total_arguments=total_arguments,
        total_comments=total_comments,
        pro_count=pro_count,
        con_count=con_count,
        toxic_high=toxic_high,
        fallacy_count=fallacy_count
    )


# -----------------------------
# DEBATE SUMMARY PAGE
# -----------------------------
@app.route('/summary')
def summary():
    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) AS total_topics FROM topics")
    total_topics = c.fetchone()["total_topics"]

    c.execute("SELECT COUNT(*) AS total_arguments FROM arguments")
    total_arguments = c.fetchone()["total_arguments"]

    c.execute("SELECT COUNT(*) AS pro_count FROM arguments WHERE predicted_side='pro'")
    pro_count = c.fetchone()["pro_count"]

    c.execute("SELECT COUNT(*) AS con_count FROM arguments WHERE predicted_side='con'")
    con_count = c.fetchone()["con_count"]

    c.execute("SELECT COUNT(*) AS strong_count FROM arguments WHERE quality_label='Strong'")
    strong_count = c.fetchone()["strong_count"]

    c.execute("SELECT COUNT(*) AS fallacy_count FROM arguments WHERE fallacy_label!='None'")
    fallacy_count = c.fetchone()["fallacy_count"]

    c.execute("SELECT COUNT(*) AS biased_count FROM arguments WHERE bias_label='Biased'")
    biased_count = c.fetchone()["biased_count"]

    conn.close()

    if pro_count > con_count:
        dominant_side = "Pro"
    elif con_count > pro_count:
        dominant_side = "Con"
    else:
        dominant_side = "Balanced"

    summary_text = (
        f"The platform currently contains {total_topics} topics and {total_arguments} arguments. "
        f"The dominant predicted stance is {dominant_side}. "
        f"There are {strong_count} strong arguments, {fallacy_count} arguments containing logical fallacies, "
        f"and {biased_count} arguments showing bias."
    )

    return render_template("summary.html", summary_text=summary_text)




# -----------------------------
# ADD TOPIC
# -----------------------------
@app.route('/add_topic', methods=['POST'])
def add_topic():
    title = request.form['title']

    conn = get_db()
    c = conn.cursor()

    try:
        c.execute("INSERT INTO topics (title) VALUES (?)", (title,))
    except:
        pass

    conn.commit()
    conn.close()

    return redirect('/')

# -----------------------------
# ADD ARGUMENT
# -----------------------------
@app.route('/add_argument', methods=['POST'])
def add_argument():
    topic_id = request.form['topic_id']
    side = request.form['side']
    content = request.form['content']
    username = request.form['username']
    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT title FROM topics WHERE id=?", (topic_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return redirect('/')

    topic = row['title']

    predicted_side = predict_stance(topic, content)

    # AI opponent reply
    ai_reply = get_ai_reply(topic, side)

    relevance = calculate_relevance(topic, content)
    evidence = calculate_evidence_score(content)
    reasoning = detect_reasoning(content)

    toxicity_label, toxicity_penalty = detect_toxicity(content)
    fallacy_label, fallacy_explanation, fallacy_penalty = detect_fallacy(content)


    bias_label, bias_explanation = detect_bias(content)

    strength = custom_ai_score(
        relevance,
        evidence,
        reasoning,
        toxicity_penalty,
        fallacy_penalty,
        0,
        0
    )

    quality_label = get_quality_label(strength)
    username = request.form.get("username", "Anonymous")

    c.execute("""
    INSERT INTO arguments
    (topic_id, side, predicted_side, content, username,
     relevance_score, evidence_score, upvotes, downvotes,
     strength_score, toxicity_label, fallacy_label,
     fallacy_explanation, quality_label, bias_label,
     bias_explanation, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    topic_id,
    side,
    predicted_side,
    content,
    username,              # ✅ correct position
    relevance,
    evidence,
    strength,
    toxicity_label,
    fallacy_label,
    fallacy_explanation,
    quality_label,
    bias_label,
    bias_explanation,
    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
))
    conn.commit()
    conn.close()

    return redirect('/')


# -----------------------------
# ADD OPPONENT REPLY
# -----------------------------
@app.route('/add_reply', methods=['POST'])
def add_reply():
    argument_id = request.form['argument_id']
    content = request.form['content']
    side = request.form['side']

    conn = get_db()
    c = conn.cursor()

    c.execute(
        "INSERT INTO replies (argument_id, content, side, created_at) VALUES (?, ?, ?, ?)",
        (argument_id, content, side, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )

    conn.commit()
    conn.close()

    return redirect('/')

# -----------------------------
# AUTO GENERATE ARGUMENT PAGE
# -----------------------------


@app.route('/generate_argument', methods=['POST'])
def generate_argument_route():
    topic = request.form['topic']
    side = request.form['side']

    generated_text = generate_argument(topic, side)

    print("DEBUG:", generated_text)   # 👈 important for testing

    return render_template(
        "generated_argument.html",
        topic=topic,
        side=side,
        text=generated_text   # ✅ MUST be 'text'
    )

# -----------------------------
# VOTING SYSTEM
# -----------------------------
@app.route('/vote/<int:arg_id>/<string:vote_type>')
def vote(arg_id, vote_type):
    voter_ip = request.remote_addr or "unknown"

    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT * FROM votes WHERE argument_id=? AND voter_ip=?", (arg_id, voter_ip))
    existing_vote = c.fetchone()

    if existing_vote:
        conn.close()
        return redirect('/')

    try:
        c.execute(
            "INSERT INTO votes (argument_id, voter_ip, vote_type) VALUES (?, ?, ?)",
            (arg_id, voter_ip, vote_type)
        )
    except:
        conn.close()
        return redirect('/')

    if vote_type == "up":
        c.execute("UPDATE arguments SET upvotes = upvotes + 1 WHERE id=?", (arg_id,))
    elif vote_type == "down":
        c.execute("UPDATE arguments SET downvotes = downvotes + 1 WHERE id=?", (arg_id,))

    c.execute("""
        SELECT relevance_score, evidence_score, upvotes, downvotes, content
        FROM arguments
        WHERE id=?
    """, (arg_id,))
    data = c.fetchone()

    if data:
        reasoning = detect_reasoning(data['content'])
        toxicity_label, toxicity_penalty = detect_toxicity(data['content'])
        fallacy_label, fallacy_explanation, fallacy_penalty = detect_fallacy(data['content'])

        new_strength = custom_ai_score(
            data['relevance_score'],
            data['evidence_score'],
            reasoning,
            toxicity_penalty,
            fallacy_penalty,
            data['upvotes'],
            data['downvotes']
        )

        new_quality = get_quality_label(new_strength)

        c.execute("""
            UPDATE arguments
            SET strength_score=?, quality_label=?, toxicity_label=?, fallacy_label=?, fallacy_explanation=?
            WHERE id=?
        """, (new_strength, new_quality, toxicity_label, fallacy_label, fallacy_explanation, arg_id))

    conn.commit()
    conn.close()

    return redirect('/')

# -----------------------------
# ADD COMMENT
# -----------------------------
@app.route('/add_comment', methods=['POST'])
def add_comment():
    argument_id = request.form['argument_id']
    content = request.form['content']

    conn = get_db()
    c = conn.cursor()

    c.execute(
        "INSERT INTO comments (argument_id, content, created_at) VALUES (?, ?, ?)",
        (argument_id, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )

    conn.commit()
    conn.close()

    return redirect('/')

# -----------------------------
# WINNER PAGE
# -----------------------------
@app.route('/winner')
def winner():
    conn = get_db()
    c = conn.cursor()

    c.execute("""
        SELECT side, SUM(strength_score) AS total_score
        FROM arguments
        GROUP BY side
    """)
    rows = c.fetchall()
    conn.close()

    pro_score = 0
    con_score = 0

    for row in rows:
        if row['side'].lower() == 'pro':
            pro_score = row['total_score'] or 0
        elif row['side'].lower() == 'con':
            con_score = row['total_score'] or 0

    if pro_score > con_score:
        winner_side = "Pro"
    elif con_score > pro_score:
        winner_side = "Con"
    else:
        winner_side = "Tie"

    return render_template(
        "winner.html",
        pro_score=round(pro_score, 2),
        con_score=round(con_score, 2),
        winner=winner_side
    )
# debate tree
@app.route('/debate_tree')
def debate_tree():
    conn = get_db()
    c = conn.cursor()

    c.execute("""
        SELECT arguments.*, topics.title
        FROM arguments
        JOIN topics ON arguments.topic_id = topics.id
        ORDER BY arguments.id DESC
    """)
    arguments = c.fetchall()

    updated_arguments = []
    for arg in arguments:
        arg = dict(arg)
        arg["ai_reply"] = get_ai_reply(arg["title"], arg["side"])
        updated_arguments.append(arg)

    arguments = updated_arguments

    c.execute("SELECT * FROM replies ORDER BY id ASC")
    replies = c.fetchall()

    conn.close()

    return render_template(
        "debate_tree.html",
        arguments=arguments,
        replies=replies
    )

# -----------------------------
# DEMO PAGE
# -----------------------------
@app.route('/demo', methods=['GET', 'POST'])
def demo():
    result = None

    if request.method == 'POST':
        topic = request.form['topic']
        argument = request.form['argument']

        topic_clean = preprocess_text(topic)
        argument_clean = preprocess_text(argument)

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([topic_clean, argument_clean])

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        result = {
            "topic_original": topic,
            "topic_clean": topic_clean,
            "argument_original": argument,
            "argument_clean": argument_clean,
            "tfidf_features": vectorizer.get_feature_names_out().tolist(),
            "tfidf_vectors": vectors.toarray().tolist(),
            "cosine_similarity": round(float(similarity), 4)
        }

    return render_template("demo.html", result=result)



# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
    









