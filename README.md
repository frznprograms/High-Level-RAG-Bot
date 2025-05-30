Hi, this is
# Hammond

Named affectionately after Top Gear host Richard Hammond, this bot was designed to be a a small (like Richard), high-level RAG application built primarily on `langchain` and deployed using `streamlit`. 

This application utilizes some basic principles of RAG, including but not limited to: 
1. (Brief) Data Cleaning to prepare text for embedding 
2. Embedding of context documents to vectors 
3. Storage of vector embeddings in a database (for simplicity and data security, only a small number of PDF documents were used and hosted on an in-memory database)
4. Retrieval of relevant documents upon a query being made
5. Generating a relevant response
6. Sending this response to a website in a presentable and simple format

As a high-level implementation, this application focuses on simplicity and convenience. `streamlit` was used to handle both frontend and backend access. Tools like `fastapi` ought to be used for speed and better control. 