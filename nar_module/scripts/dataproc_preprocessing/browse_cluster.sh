GCP_PROJECT_NAME="[GCP Project name. e.g. chameleon-research]"
GCP_REGION=us-central1
GCP_ZONE=us-central1-b
GCS_BUCKET_DATAPROC="[GCS Bucket name for the Dataproc cluster. e.g. chameleon_dataproc_cluster_bucket]"
DATAPROC_CLUSTER_NAME="[Dataproc cluster name. e.g. chameleon-dataproc-cluster]"

PORT=$(( ( RANDOM % 1100 )  + 1000 ))

#Open tunnel
gcloud compute ssh --project ${GCP_PROJECT_NAME} --zone=${GCP_ZONE} \
  --ssh-flag="-D" --ssh-flag="$PORT" --ssh-flag="-N" --ssh-flag="-n" "${DATAPROC_CLUSTER_NAME}-m" &

sleep 7

#Open browser
google-chrome \
    "http://${DATAPROC_CLUSTER_NAME}-m:8123" \
    --proxy-server="socks5://localhost:$PORT" \
    --host-resolver-rules="MAP * 0.0.0.0 , EXCLUDE localhost" \
    --user-data-dir=${HOME}/.google-chrome/session${DISPLAY}
