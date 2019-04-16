GCP_PROJECT_NAME="[GCP Project name. e.g. chameleon-research]"
GCP_REGION=us-central1
GCP_ZONE=us-central1-b
GCS_BUCKET_DATAPROC="[GCS Bucket name for the Dataproc cluster. e.g. chameleon_dataproc_cluster_bucket]"
DATAPROC_CLUSTER_NAME="[Dataproc cluster name. e.g. chameleon-dataproc-cluster]"


gcloud dataproc clusters delete chameleon-dataproc-cluster \
--project ${GCP_PROJECT_NAME} \
--region ${GCP_REGION}

gsutil -m rm -r gs://${GCS_BUCKET_DATAPROC}/
