CURRENT_MODEL_PATH=''
MERGED_MODEL_DIR=''
EVAL_DATASETS_DIR=''
OUTPUT_DIR=''
STEP=''
TEMPLATE=''

  # 模型评估
python evaluation.py \
  --old-model-path ${CURRENT_MODEL_PATH} \
  --new-model-path ${MERGED_MODEL_DIR} \
  --input-file ${EVAL_DATASETS_DIR}/${CURRENT_EVAL_DATASET}.jsonl \
  --output-dir ${OUTPUT_DIR} \
  --step ${STEP} \
  --template ${TEMPLATE} \
  --max-batch-size 64 \
