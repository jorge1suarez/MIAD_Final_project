
We need:
1) An Artifact repository: miad-repo
2) Vertex AI training Job: 
3) Google cloud storage bucket: miad-bucket


* Install the Google Cloud CLI.
* build/push image 

```bash
gcloud builds submit --config cloudbuild.yaml
```