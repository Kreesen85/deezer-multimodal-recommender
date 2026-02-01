#!/bin/bash
# Quick start script for XGBoost baseline

echo "=========================================="
echo "XGBoost Baseline - Deezer Skip Prediction"
echo "=========================================="
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if preprocessed data exists
DATA_FILE="../../data/processed/preprocessing/train_preprocessed_sample.csv"

if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Error: Preprocessed data not found at $DATA_FILE"
    echo ""
    echo "Please generate preprocessed data first:"
    echo "  cd ../../02_preprocessing"
    echo "  python demo_preprocessing_with_users.py"
    exit 1
fi

echo "✓ Data file found: $DATA_FILE"
echo ""

# Run with Anaconda Python
if command -v /opt/anaconda3/bin/python &> /dev/null; then
    echo "Using Anaconda Python..."
    /opt/anaconda3/bin/python xgboost_baseline.py
elif command -v python &> /dev/null; then
    echo "Using system Python..."
    python xgboost_baseline.py
else
    echo "❌ Error: Python not found"
    exit 1
fi
