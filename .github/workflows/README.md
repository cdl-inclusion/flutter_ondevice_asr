# Performance test workflows

These workflows build an APK and run on-device inference performance tests on
Firebase Test Lab. Each one targets a specific model.

| Workflow | Model | Bucket-path secret |
| --- | --- | --- |
| `performance_test_whisper_tiny_int8.yml` | Whisper tiny (int8) | `WHISPER_TINY_MODEL_ZIP_BUCKET_PATH` |
| `performance_test_whisper_small_int8.yml` | Whisper small (int8) | `WHISPER_SMALL_MODEL_ZIP_BUCKET_PATH` |
| `performance_test_fastconformer_hybrid_int8.yml` | FastConformer hybrid (int8) | `FASTCONFORMER_HYBRID_INT8_ZIP_BUCKET_PATH` |
| `performance_test_fastconformer_hybrid_fp32.yml` | FastConformer hybrid (fp32) | `FASTCONFORMER_HYBRID_FP32_ZIP_BUCKET_PATH` |


## Where do the models come from?

At run time the workflow downloads a zipped model from a Google Cloud Storage bucket 
and unzips it into the Flutter assets directory. The `gs://…` path to that zip 
is stored in a GitHub secret (see the table above).

## Uploading a model

The zip must contain the model files at its **root** (no wrapping folder), so
that `unzip -d <assets dir>` places them directly into the model directory.

```bash
# 1. Zip the int8 model files at the zip root.
cd assets/transcribers/whisper/models/whisper_small/int8
zip -r ~/whisper_small_model.zip .

# 2. Upload to the GCS bucket (use the same bucket as the existing models).
gcloud storage cp ~/whisper_small_model.zip \
  gs://<your-bucket>/<path>/whisper_small_model.zip
```

## Configuring the secret

Secret values are write-only in GitHub — you can overwrite but not read them.
Add / update the bucket-path secret so the workflow can find the uploaded zip:

  ```bash
  # repository secret
  gh secret set WHISPER_SMALL_MODEL_ZIP_BUCKET_PATH \
    --body "gs://<your-bucket>/<path>/whisper_small_model.zip"

  # or environment-scoped secret
  gh secret set WHISPER_SMALL_MODEL_ZIP_BUCKET_PATH \
    --env <environment> \
    --body "gs://<your-bucket>/<path>/whisper_small_model.zip"
  ```

## GCS permissions

The workflow authenticates to Google Cloud using the service account in the
`GOOGLE_SA_KEY` secret. That service account must have **read** access to the
model object:

```bash
gcloud storage buckets add-iam-policy-binding gs://<your-bucket> \
  --member="serviceAccount:<sa-email>" \
  --role="roles/storage.objectViewer"
```

The service account email is the `client_email` field inside the `GOOGLE_SA_KEY`
JSON key.
