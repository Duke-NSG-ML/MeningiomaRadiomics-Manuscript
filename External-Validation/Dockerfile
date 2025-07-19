FROM python:3.9.21-slim

ENV OMP_NUM_THREADS=1 \
    INPUT_DIR=/data/BraTS-MEN-Test \
    OUTCOME_CSV=/data/outcomes.csv \
    FEATURES_OUT=features.csv \
    TOPO_PKL=topo.pkl \
    TOPO_CSV=topo.csv \
    AGE_CSV=/data/AgeSex.csv \
    MERGED_X=X.csv \
    MERGED_Y=y.csv \
    PIPELINE_FILE=calibrated_pipeline.pkl \
    FEATURES_FILE=model_feature_names.pkl \
    PRED_OUT=/data/predictions.csv \
    METRICS_OUT=/data/metrics.txt
WORKDIR /app
# make sure pip & numpy (<2) are in place for pyradiomics
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "numpy<2"

# pyradiomics & other deps
RUN pip install --no-cache-dir pyradiomics

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all scripts (extract_features.py, topology_extraction.py, join.py)
COPY . .

# final ENTRYPOINT runs all three scripts in sequence
ENTRYPOINT ["bash","-lc", "\
    python extract_features.py \
      --input  $INPUT_DIR \
      --age $AGE_CSV \
      --outcome $OUTCOME_CSV \
      --output  $FEATURES_OUT && \
    python topology_extraction.py \
      --data-dir    $INPUT_DIR \
      --outcome-csv $OUTCOME_CSV \
      --output-pkl  $TOPO_PKL \
      --output-csv  $TOPO_CSV && \
    python join.py \
      --feature_csv   $FEATURES_OUT \
      --topo_csv      $TOPO_CSV \
      --age_csv       $AGE_CSV \
      --outcome_csv   $OUTCOME_CSV \
      --x_out         $MERGED_X \
      --y_out         $MERGED_Y && \
    python ML.py \
      --pipeline $PIPELINE_FILE \
      --features $FEATURES_FILE \
      --X $MERGED_X \
      --y $MERGED_Y \
      --probs_csv $PRED_OUT \
      --metrics $METRICS_OUT \
"]  
