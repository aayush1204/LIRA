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

### Pushing image to Google Cloud Run
- Build image using:<br>
```gcloud builds submit --tag us.gcr.io/lira-379304/lira-docker```
- Deploy on Google Cloud Run <br>
``` 
gcloud run deploy lira-service \                            
 --image us.gcr.io/lira-379304/lira-docker \
 --project lira-379304 \
 --region "us-central1" \
 --allow-unauthenticated
```

### Google Cloud Run hosted website
Link: ```https://lira-service-hj65fnd7ba-uc.a.run.app/```