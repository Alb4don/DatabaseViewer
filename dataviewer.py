import os
import re
import sqlite3
import tempfile
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, inspect
from pymongo import MongoClient

st.set_page_config(page_title="Data Insights & Context Graph", layout="wide")

def secure_filename(filename: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\.-]', '_', filename)

def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r'<[^>]*>', '', text)

def clean_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(lambda x: str(x) if isinstance(x, (dict, list, tuple, set)) else x)
            df_clean[col] = df_clean[col].astype(str)
    return df_clean

def build_context_graph(df: pd.DataFrame, text_column: str, threshold: float = 0.2):
    texts = df[text_column].dropna().astype(str).apply(sanitize_text).tolist()
    texts = [t for t in texts if t.strip()]
    
    if not texts or len(texts) < 2:
        return nx.Graph()
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return nx.Graph()
        
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    G = nx.Graph()
    for i in range(len(texts)):
        node_label = texts[i][:30] + '...' if len(texts[i]) > 30 else texts[i]
        G.add_node(i, text=node_label, full_text=texts[i])
        
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if similarity_matrix[i][j] >= threshold:
                G.add_edge(i, j, weight=similarity_matrix[i][j])
    return G

def plot_graph(G: nx.Graph):
    if len(G.nodes) == 0:
        st.warning("Not enough textual data or connections to generate a graph.")
        return
        
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    hover_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        hover_text.append(G.nodes[node]['text'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=hover_text,
        textposition="bottom center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title=dict(text='Node Connections', side='right'),
                xanchor='left'
            ),
            line_width=2))

    node_adjacencies = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
    node_trace.marker.color = node_adjacencies

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("Database Viewer")
    
    db_type = st.sidebar.selectbox(
        "Select Database Type", 
        ["SQLite (File Upload)", "MySQL / MariaDB", "PostgreSQL", "MongoDB", "Oracle", "MS SQL Server"]
    )
    
    if "current_db_type" not in st.session_state or st.session_state["current_db_type"] != db_type:
        st.session_state.clear()
        st.session_state["current_db_type"] = db_type

    tables = []
    df = pd.DataFrame()
    connection_successful = False
    
    if db_type == "SQLite (File Upload)":
        uploaded_file = st.sidebar.file_uploader("Upload Database File", type=['db', 'sqlite', 'sqlite3'])
        if uploaded_file is not None:
            file_ext = uploaded_file.name.split('.')[-1]
            secure_filename(uploaded_file.name)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                
            try:
                conn = sqlite3.connect(tmp_file_path, uri=True)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                if tables:
                    selected_table = st.sidebar.selectbox("Select Table", tables)
                    if selected_table in tables:
                        df = pd.read_sql_query(f'SELECT * FROM "{selected_table}"', conn)
                        connection_successful = True
            except Exception as e:
                st.error(f"Error reading SQLite file: {str(e)}")
            finally:
                if 'conn' in locals() and conn:
                    conn.close()
                if os.path.exists(tmp_file_path):
                    try:
                        os.remove(tmp_file_path)
                    except Exception:
                        pass
    else:
        with st.sidebar.form("db_credentials"):
            host = st.text_input("Host", "localhost")
            port = st.text_input("Port")
            db_name = st.text_input("Database Name")
            user = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Connect")
            
        if submitted and host and db_name:
            try:
                if db_type == "MongoDB":
                    port = int(port) if port else 27017
                    client = MongoClient(host, port, username=user, password=password, serverSelectionTimeoutMS=5000)
                    db = client[db_name]
                    tables = db.list_collection_names()
                    if tables:
                        st.session_state['mongo_client'] = client
                        st.session_state['db_name'] = db_name
                        st.session_state['tables'] = tables
                else:
                    if db_type in ["MySQL", "MariaDB"]:
                        port = port if port else "3306"
                        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"
                    elif db_type == "PostgreSQL":
                        port = port if port else "5432"
                        url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
                    elif db_type == "Oracle":
                        port = port if port else "1521"
                        url = f"oracle+oracledb://{user}:{password}@{host}:{port}/?service_name={db_name}"
                    elif db_type == "MS SQL Server":
                        port = port if port else "1433"
                        url = f"mssql+pymssql://{user}:{password}@{host}:{port}/{db_name}"
                        
                    engine = create_engine(url)
                    inspector = inspect(engine)
                    tables = inspector.get_table_names()
                    if tables:
                        st.session_state['sql_engine'] = engine
                        st.session_state['tables'] = tables
            except Exception as e:
                st.sidebar.error(f"Connection Error: {str(e)}")

    if 'tables' in st.session_state and db_type != "SQLite (File Upload)":
        tables = st.session_state['tables']
        if tables:
            selected_table = st.sidebar.selectbox("Select Table", tables)
            try:
                if db_type == "MongoDB" and 'mongo_client' in st.session_state:
                    client = st.session_state['mongo_client']
                    db = client[st.session_state['db_name']]
                    data = list(db[selected_table].find().limit(5000))
                    df = pd.DataFrame(data)
                    connection_successful = True
                elif 'sql_engine' in st.session_state:
                    engine = st.session_state['sql_engine']
                    df = pd.read_sql_table(selected_table, engine)
                    connection_successful = True
            except Exception as e:
                st.error(f"Error reading table: {str(e)}")

    if connection_successful and not df.empty:
        df_display = clean_dataframe_for_display(df)
        st.dataframe(df_display, use_container_width=True)
        
        text_cols = df_display.select_dtypes(include=['object', 'string']).columns.tolist()
        
        if text_cols:
            st.subheader("AI Context Graph")
            col1, col2 = st.columns(2)
            with col1:
                selected_text_col = st.selectbox("Select column representing notes/text", text_cols)
            with col2:
                similarity_threshold = st.slider("Context Similarity Threshold", 0.0, 1.0, 0.2, 0.05)
                
            if selected_text_col:
                with st.spinner("Generating AI Context Graph..."):
                    G = build_context_graph(df_display, selected_text_col, similarity_threshold)
                    plot_graph(G)
        else:
            st.info("No text columns found for context graph generation.")
    elif connection_successful:
        st.info("Table is empty.")

if __name__ == "__main__":
    main()
