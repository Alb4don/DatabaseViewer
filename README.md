# Features

- Displays full dataframes with cleaned object types for safe rendering.
- Builds graphs using TF-IDF vectorization on sanitized text inputs.
- Handles edge cases like empty tables or insufficient text data.
  
![dbfront2](https://github.com/user-attachments/assets/709c5278-6080-4c78-845f-1778fcf860d8)
![dbfront](https://github.com/user-attachments/assets/7d14cad7-6bc7-441d-ad4c-2bbaee0d59aa)

  
# Requirements

 Install via pip:

     pip install streamlit pandas networkx plotly scikit-learn sqlalchemy pymongo pymysql psycopg2 pymssql oracledb

- Streamlit launches with ***streamlit run dataviewer.py.*** No additional configuration needed.

# Limitations

- Graphs require at least two non-empty text entries, large datasets may slow vectorization.
- MongoDB limits to 5000 documents.
- No query customization beyond full table loads. Assumes text columns are object/string dtype.

