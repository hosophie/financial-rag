#!/bin/bash
clear
# 设置变量
CSV_FILE="test_case-backup.csv"
OUTPUT_DIR="./output"
BASE_NAME="experiment_0_baseline"

# 运行所有实验
echo "=== 运行批量测试 ==="
python batch_test.py --file "$CSV_FILE" --desc "" --exp_name "$BASE_NAME"
python batch_test.py --file "$CSV_FILE" --desc "" --exp_name "experiment_1_query_expansion"
python batch_test.py --file "$CSV_FILE" --desc "" --exp_name "experiment_2_hyde"
python batch_test.py --file "$CSV_FILE" --desc "" --exp_name "experiment_3_doc_filter"
python batch_test.py --file "$CSV_FILE" --desc "" --exp_name "experiment_4_expansion_plus_filter"
python batch_test.py --file "$CSV_FILE" --desc "" --exp_name "experiment_5_refine_generation"
python batch_test.py --file "$CSV_FILE" --desc "" --exp_name "experiment_6_full_pipeline"

echo "=== 运行回归测试 ==="
# 回归测试
python regression_test.py --baseline "$OUTPUT_DIR/test_${BASE_NAME}.json" --current "$OUTPUT_DIR/test__experiment_1_query_expansion.json" --output "$OUTPUT_DIR/regression_experiment_0_1_query_expansion.json"
python regression_test.py --baseline "$OUTPUT_DIR/test_${BASE_NAME}.json" --current "$OUTPUT_DIR/test__experiment_2_hyde.json" --output "$OUTPUT_DIR/regression_experiment_0_2_hyde.json"
python regression_test.py --baseline "$OUTPUT_DIR/test_${BASE_NAME}.json" --current "$OUTPUT_DIR/test__experiment_3_doc_filter.json" --output "$OUTPUT_DIR/regression_experiment_0_3_doc_filter.json"
python regression_test.py --baseline "$OUTPUT_DIR/test_${BASE_NAME}.json" --current "$OUTPUT_DIR/test__experiment_4_expansion_plus_filter.json" --output "$OUTPUT_DIR/regression_experiment_4_expansion_plus_filter.json"
python regression_test.py --baseline "$OUTPUT_DIR/test_${BASE_NAME}.json" --current "$OUTPUT_DIR/test__experiment_5_refine_generation.json" --output "$OUTPUT_DIR/regression_experiment_0_5_refine_generation.json"
python regression_test.py --baseline "$OUTPUT_DIR/test_${BASE_NAME}.json" --current "$OUTPUT_DIR/test__experiment_6_full_pipeline.json" --output "$OUTPUT_DIR/regression_experiment_0_6_full_pipeline.json"

echo "=== 所有测试完成 ==="