GCP_PROJECT_NAME="[GCP Project name. e.g. chameleon-research]"
GCP_REGION=us-central1
GCP_ZONE=us-central1-b
GCS_BUCKET_DATAPROC="[GCS Bucket name for the Dataproc cluster. e.g. chameleon_dataproc_cluster_bucket]"
DATAPROC_CLUSTER_NAME="[Dataproc cluster name. e.g. chameleon-dataproc-cluster]"

gsutil mb -c regional -l us-central1 -p ${GCP_PROJECT_NAME}  gs://${GCS_BUCKET_DATAPROC} 
gcloud dataproc clusters create chameleon-dataproc-cluster \
    --project ${GCP_PROJECT_NAME} \
    --bucket ${GCS_BUCKET_DATAPROC} \
    --image-version 1.3 \
    --region us-central1 \
    --zone ${GCP_ZONE} \
    --num-workers 4 \
    --scopes cloud-platform \
    --initialization-actions gs://dataproc-initialization-actions/jupyter/jupyter.sh \
    --initialization-action-timeout 20m \
    --master-machine-type "n1-highmem-4" \
    --worker-machine-type "n1-standard-4" \
    --worker-machine-type "n1-standard-4" \
    --worker-boot-disk-size=500GB \
    --master-boot-disk-size=500GB

    
#--properties spark:spark.driver.maxResultSize=20328m,spark:spark.driver.memory=20656m,spark:spark.executor.heartbeatInterval=30s,spark:spark.yarn.executor.memoryOverhead=5058,spark:spark.executor.memory=9310m,spark:spark.yarn.am.memory=8586m