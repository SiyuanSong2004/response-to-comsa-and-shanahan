#!/bin/bash

echo "Starting Experiment 2: Cross-Model Prediction"
mkdir -p temppred_data

if [ ! -d "selfref_data" ]; then
    echo "Error: Run Experiment 1 first"
    exit 1
fi

all_models=("gpt-4o-2024-08-06" "gpt-4.1-2025-04-14" "gemini-2.0-flash" "gemini-2.5-flash")

get_model_family() {
    case "$1" in
        gpt-4o-2024-08-06|gpt-4.1-2025-04-14)
            echo "gpt"
            ;;
        gemini-2.0-flash|gemini-2.5-flash)
            echo "gemini"
            ;;
    esac
}

gpt_delay=1
gemini_delay=6

for predictor_model in "${all_models[@]}"; do
    predictor_family=$(get_model_family "$predictor_model")
    
    if [ "$predictor_family" = "gpt" ]; then
        delay=$gpt_delay
    else
        delay=$gemini_delay
    fi
    
    for target_model in "${all_models[@]}"; do
        echo "$predictor_model → $target_model"
        
        python tempprediction.py \
            --input-file "selfref_data/${target_model}_responses.csv" \
            --model-family "$predictor_family" \
            --model-name "$predictor_model" \
            --output-dir temppred_data \
            --delay $delay
        
        if [ $? -eq 0 ]; then
            echo "✓ completed"
        else
            echo "✗ failed"
        fi
    done
done

echo "Experiment 2 completed" 