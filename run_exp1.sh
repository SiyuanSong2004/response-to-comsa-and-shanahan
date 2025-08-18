#!/bin/bash

echo "Starting Experiment 1: Self-Reflection"
mkdir -p selfref_data

declare -A models=(
    ["gpt-4o-2024-08-06"]="gpt"
    ["gpt-4.1-2025-04-14"]="gpt"
    ["gemini-2.0-flash"]="gemini"
    ["gemini-2.5-flash"]="gemini"
)

gpt_delay=1
gemini_delay=6

for model_name in "${!models[@]}"; do
    model_family=${models[$model_name]}
    
    if [ "$model_family" = "gpt" ]; then
        delay=$gpt_delay
    else
        delay=$gemini_delay
    fi
    
    echo "Running $model_name..."
    
    python selfreflection.py \
        --model_family "$model_family" \
        --model-name "$model_name" \
        --temp-start 0.0 \
        --temp-end 2.0 \
        --temp-step 0.1 \
        --prompt-types factual normal creative \
        --subjects elephants unicorns murlocs \
        --num-runs 3 \
        --delay $delay
    
    if [ $? -eq 0 ]; then
        echo "✓ $model_name completed"
    else
        echo "✗ $model_name failed"
    fi
done

echo "Experiment 1 completed" 