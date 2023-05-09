# LIRA

### Description
Legal Information Retrieval, Analysis and Summarization - LIRA(S)
- Allows user to input the query or the reference document for searching
- Searches for documents relevant to query. Returns top 5 matches which can be viewed.
- Summarizes the document chosen for viewing.
- Provides documents similar to the one being viewed. 

### Docker Commands
- Building docker image:<br>
```docker build -t lira-docker . --platform linux/amd64```

- Running docker image:<br>
```docker run -p 8888:80 lira-docker```

### Running locally
- Run Docker commands
- Site hosted on ```localhost:8888```
