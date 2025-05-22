import argparse
import os
import pandas as pd
import joblib # For saving models
from google.cloud import bigquery, storage # Import storage client
# Import GPU-accelerated UMAP and HDBSCAN from cuML
from cuml.manifold import UMAP as cuML_UMAP
from cuml.cluster import HDBSCAN as cuML_HDBSCAN
from sklearn.pipeline import Pipeline
import hypertune # Vertex AI Hyperparameter Tuning library
import numpy as np # For handling numerical arrays
import tempfile # For creating temporary local files
from datetime import datetime

import argparse
 
 
import umap # UMAP library
import hdbscan # HDBSCAN library
# Removed TfidfVectorizer as embeddings are pre-computed
#from sklearn.metrics import silhouette_score # Example clustering metric
from sklearn.pipeline import Pipeline
import hypertune # Vertex AI Hyperparameter Tuning library
import numpy as np # For handling numerical arrays

import tempfile

def get_args():
  """Parses command-line arguments for hyperparameters and BigQuery details."""
  parser = argparse.ArgumentParser()

  # BigQuery Arguments
  parser.add_argument('--bq_project_id', type=str, required=True,
                      help='Google Cloud Project ID for BigQuery.')

  parser.add_argument('--specific_objective_id', type=str, default=0,
                      help='Google Cloud Project ID for BigQuery.')

  # UMAP Hyperparameters
  parser.add_argument('--umap_n_neighbors', type=int, default=15,
                      help='UMAP: The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.')
  parser.add_argument('--umap_min_dist', type=float, default=0.1,
                      help='UMAP: The effective minimum distance between embedded points.')
  parser.add_argument('--umap_n_components', type=int, default=20,
                      help='UMAP: The dimension of the space to embed into.')
  parser.add_argument('--umap_metric', type=str, default='cosine',
                      help='UMAP: The metric to use to compute distances in high dimensional space.')


  # HDBSCAN Hyperparameters
  parser.add_argument('--hdbscan_min_cluster_size', type=int, default=10,
                      help='HDBSCAN: The minimum size of clusters.')
  parser.add_argument('--hdbscan_min_samples', type=int, default=None,
                      help='HDBSCAN: The number of samples in a neighborhood for a point to be considered a core point. Defaults to min_cluster_size.')
  parser.add_argument('--hdbscan_cluster_selection_epsilon', type=float, default=0.0,
                      help='HDBSCAN: A distance threshold. Clusters below this value will be merged.')
  parser.add_argument('--hdbscan_metric', type=str, default='euclidean',
                      help='HDBSCAN: The metric to use when computing distances between data points.')

  # Model Output Arguments
  parser.add_argument('--model_dir', type=str, default=os.getenv('AIP_MODEL_DIR'),
                      help='AIP_MODEL_DIR is provided by Vertex AI for model saving.')


  args = parser.parse_args()

  # Set default for hdbscan_min_samples if not provided
  if args.hdbscan_min_samples is None:
      args.hdbscan_min_samples = args.hdbscan_min_cluster_size

  return args

def load_data_from_bigquery(project_id, specific_objective_id):
    """Loads embedding vectors from a specified BigQuery table."""
    client = bigquery.Client(project=project_id)
    
    query = f"""
     select  p.project_id, p.content , p,text_embedding
    from `eu_projects.project_summary_embeddings` p
    inner join `eu_projects.project_specific_objective` so on p.project_id  = so.project_id
    where so.specific_objective_id = {specific_objective_id} or {specific_objective_id} = 0
    """

    print(f"Executing BigQuery query: {query}")
    df = client.query(query).to_dataframe()

    # Ensure the embedding column contains lists/arrays and convert to a NumPy array
    # Drop rows where the embedding might be null or empty
    df = df.dropna(subset=["text_embedding"])
    # Convert list of lists to a NumPy array for UMAP
    embeddings = np.array(df["text_embedding"].tolist())
    print(f"Successfully loaded {len(embeddings)} embedding vectors from BigQuery.")
    print(f"Shape of loaded embeddings: {embeddings.shape}")
    return embeddings

def main():
    start = datetime.now()
    args = get_args()

    # 1. Load Data (pre-computed embeddings)
    data_load_start = datetime.now()
    try:
        # embeddings will now be a NumPy array of vectors
        embeddings = load_data_from_bigquery(args.bq_project_id, args.specific_objective_id)
    except Exception as e:
        print(f"Error loading embeddings from BigQuery: {e}")
        # For hyperparameter tuning, if data loading fails, report a very low score
        # so this trial is penalized.
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            metric_value=-1.0, # Or any value indicating failure/poor performance
            metric_name='dbcv_score'
        )
        return

    if embeddings.size == 0:
            print("No valid embedding vectors found after loading from BigQuery. Exiting.")
            hpt = hypertune.HyperTune()
            hpt.report_hyperparameter_tuning_metric(
                metric_value=-1.0,
                metric_name='dbcv_score'
            )
            return

  # Check if embeddings have at least 2 dimensions for UMAP (if n_components > 1)
    if embeddings.ndim < 2 or embeddings.shape[1] < args.umap_n_components:
        print(f"Input embeddings have {embeddings.shape[1]} dimensions, which is less than UMAP's target n_components={args.umap_n_components}. Adjusting n_components to input dimension.")
        # Adjust n_components to be at most the input dimension
        args.umap_n_components = min(args.umap_n_components, embeddings.shape[1])
        if args.umap_n_components < 2:
            print("UMAP n_components cannot be less than 2 for silhouette score calculation. Exiting.")
            hpt = hypertune.HyperTune()
            hpt.report_hyperparameter_tuning_metric(
                metric_value=-1.0,
                metric_name='dbcv_score'
            )
            return
    # 2. UMAP Dimensionality Reduction
    umap_start = datetime.now()
    print(f"Embeddings loading duration {(umap_start - data_load_start).total_seconds()}")

    print(f"Applying UMAP with n_neighbors={args.umap_n_neighbors}, min_dist={args.umap_min_dist}, n_components={args.umap_n_components}, metric={args.umap_metric}...")
    try:
        umap_model = cuML_UMAP(
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            n_components=args.umap_n_components,
            metric=args.umap_metric,
            random_state=42 # For reproducibility
        )
        reduced_vectors = umap_model.fit_transform(embeddings) # Use the loaded embeddings directly
        print(f"Reduced vectors shape: {reduced_vectors.shape}")
    except Exception as e:
        print(f"Error during UMAP transformation: {e}")
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            metric_value=-1.0,
            metric_name='dbcv_score'
        )
        return

    hdbscan_start = datetime.now()
    print(f"UMAP duration {(hdbscan_start - umap_start).total_seconds()}")

    # 3. HDBSCAN Clustering
    print(f"Applying HDBSCAN with min_cluster_size={args.hdbscan_min_cluster_size}, min_samples={args.hdbscan_min_samples}, cluster_selection_epsilon={args.hdbscan_cluster_selection_epsilon}, metric={args.hdbscan_metric}...")
    try:
        clusterer = cuML_HDBSCAN(
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=args.hdbscan_min_samples,
            cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
            metric=args.hdbscan_metric,
            prediction_data=True # Required for future prediction if saving model
        )
        clusters = clusterer.fit_predict(reduced_vectors)
        print(f"HDBSCAN found {len(set(clusters)) - (1 if -1 in clusters else 0)} clusters (excluding noise).")
        print(f"Number of noise points (-1): {list(clusters).count(-1)}")
    except Exception as e:
        print(f"Error during HDBSCAN clustering: {e}")
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            metric_value=-1.0,
            metric_name='dbcv_score'
        )
        return
   
    print(f"HDBSCAN duration: {(datetime.now() - hdbscan_start).total_seconds()}")

    # 4. Evaluate Clustering
    # Silhouette Score ranges from -1 (bad) to +1 (good).
    # It requires at least 2 clusters (excluding noise).
    dbcv_score = -1.0 # Default to a very low score if evaluation is not possible
    unique_clusters = set(clusters)
    if len(unique_clusters) > 1 and (len(unique_clusters) > 2 or -1 not in unique_clusters):
        # Calculate silhouette score only on non-noise points if there are at least 2 clusters
        # and if there are noise points, ensure there are at least 2 non-noise clusters.
        # If all points are noise or only one cluster is found, silhouette score is undefined.
        non_noise_indices = [i for i, label in enumerate(clusters) if label != -1]
        if len(non_noise_indices) >= 2:
            try:
                # Ensure reduced_vectors is a NumPy array for advanced indexing
                if not isinstance(reduced_vectors, np.ndarray):
                    reduced_vectors = np.array(reduced_vectors)
                #score = silhouette_score(reduced_vectors[non_noise_indices], clusters[non_noise_indices])
                
                metric_for_dbcv = args.hdbscan_metric if args.hdbscan_metric in ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'] else 'euclidean'
                reduced_vectors = reduced_vectors.astype(np.float64)
                dbcv_score = hdbscan.validity.validity_index(reduced_vectors, clusters, metric=metric_for_dbcv)
                
                print(f"dbcv_score: {dbcv_score}")
            except Exception as e:
                print(f"Error calculating dbcv_score: {e}")
                dbcv_score = -1.0
        else:
            print("Not enough non-noise points or clusters to calculate dbcv_score  .")
    else:
        print("Not enough clusters (or only noise) to calculate dbcv_score  .")

    # 5. Report Metric to Vertex AI Hyperparameter Tuning

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        metric_value=dbcv_score,
        hyperparameter_metric_tag='dbcv_score' # This name must match the metric_spec in your tuning job
    )

    # 6. Save Models (Optional but Recommended)
    # Save the UMAP model and HDBSCAN clusterer
    if args.model_dir:
        print(f"Saving models to {args.model_dir}...")
        try:
            # Create a temporary local directory to save the model
            with tempfile.TemporaryDirectory() as tmpdir:
                local_model_path = os.path.join(tmpdir, 'model.joblib')
                
                # Attempt to save the pipeline with cuML models to local path
                # Note: cuML models can sometimes have serialization issues with joblib
                # if not handled carefully (e.g., device context).
                # If this joblib.dump fails, the outer try-except will catch it.
                model_pipeline = Pipeline([
                    ('umap', umap_model),
                    ('hdbscan', clusterer)
                ])
                joblib.dump(model_pipeline, local_model_path)
                print(f"Models saved temporarily to {local_model_path}")

                # Upload the local model file to GCS
                gcs_bucket_name = args.model_dir.split('/')[2]
                gcs_blob_path = '/'.join(args.model_dir.split('/')[3:]) + '/model.joblib'
                
                storage_client = storage.Client()
                bucket = storage_client.bucket(gcs_bucket_name)
                blob = bucket.blob(gcs_blob_path)
                blob.upload_from_filename(local_model_path)
                print(f"Model uploaded to gs://{gcs_bucket_name}/{gcs_blob_path}")

        except Exception as e:
            print(f"Error saving or uploading model: {e}")
            print(f"Saving a dummy file to indicate trial completion in {args.model_dir}.")
            # Fallback: Save a placeholder if actual model saving/upload fails
            # This ensures the trial directory in GCS isn't empty, which can sometimes
            # cause issues with Vertex AI expecting artifacts.
            gcs_bucket_name = args.model_dir.split('/')[2]
            gcs_blob_path = '/'.join(args.model_dir.split('/')[3:]) + '/cuML_model_placeholder.txt'
            storage_client = storage.Client()
            bucket = storage_client.bucket(gcs_bucket_name)
            blob = bucket.blob(gcs_blob_path)
            blob.upload_from_string("cuML models completed. Best parameters are reported in Vertex AI.")
            print(f"Placeholder uploaded to gs://{gcs_bucket_name}/{gcs_blob_path}")
    else:
        print("AIP_MODEL_DIR not set, skipping model saving.")



    print("Domne")
    duration = datetime.now() - start
    print(f'Execution time:{duration.total_seconds()}')
    print(duration.total_seconds())



if __name__ == '__main__':
    main()