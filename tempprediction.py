#!/usr/bin/env python3
"""
script to get the data for within- and cross-model temperature prediction experiments
MAIN PROMPT: This is an LLM generated sentence: {sent}. The model is prompted to generate a {type} sentence about {subject}. In a short paragraph, analyze whether the temperature of the model is high or low, given the produced sentence. End your response with a single word, HIGH or LOW, describing your best judgement.
"""

import argparse
import os
import pandas as pd
import time
from typing import Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class TempPrediction:
    def __init__(self, model_family: str, model_name: Optional[str] = None):
        self.model_family = model_family.lower()
        self.model_name = model_name
        self.client = None
        
        # Main prompt template
        self.prompt_template = "This is an LLM generated sentence: {sent}. The model is prompted to generate a {type} sentence about {subject}. In a short paragraph, analyze whether the temperature of the model is high or low, given the produced sentence. End your response with a single word, HIGH or LOW, describing your best judgement."
        
        self._initialize_model()
    
    def _initialize_model(self):
        if self.model_family == 'gpt':
            if not OPENAI_AVAILABLE:
                raise ImportError("Install OpenAI library: pip install openai")
            self.client = OpenAI()
            self.model_name = self.model_name or "gpt-4o"
            
        elif self.model_family == 'gemini':
            if not GEMINI_AVAILABLE:
                raise ImportError("Install Google GenAI library: pip install google-genai")
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            self.client = genai.Client(api_key=api_key)
            self.model_name = self.model_name or "gemini-2.0-flash"
            
        else:
            raise ValueError(f"Unsupported model family: {self.model_family}. Use 'gpt' or 'gemini'")
    
    def generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 300) -> str:
        if self.model_family == 'gpt':
            return self._generate_gpt_response(prompt, temperature, max_tokens)
        elif self.model_family == 'gemini':
            return self._generate_gemini_response(prompt, temperature, max_tokens)
    
    def _generate_gpt_response(self, prompt: str, temperature: float, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    
    def _generate_gemini_response(self, prompt: str, temperature: float, max_tokens: int) -> str:
        config_params = {
            "max_output_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Disable thinking for Gemini 2.5 to prevent token waste
        if "2.5" in self.model_name:
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**config_params)
        )
        
        if response.text is None:
            raise RuntimeError("Gemini returned empty response")
        
        return response.text.strip()
    
    def predict_temperatures(self, input_file: str, output_dir: str, predictor_temperatures: list = [0.0], delay: float = 0) -> list:
        """
        Predict temperatures for sentences in the input CSV file.
        
        Args:
            input_file: Path to CSV file with sentences to analyze
            output_dir: Directory to save results
            predictor_temperatures: List of temperatures to use for prediction
            delay: Delay between requests (seconds)
            
        Returns:
            List of output CSV file paths
        """
        # Read input data
        try:
            df = pd.read_csv(input_file)
            print(f"Loaded {len(df)} sentences from {input_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to read input file: {e}")
        
        # Validate required columns
        required_columns = ['sentence', 'subject', 'prompt_type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Filter valid rows only
        print("Filtering for valid responses...")
        original_count = len(df)
        
        # Keep only rows with valid sentences and valid predicted temperatures
        df = df[
            (df['sentence'].notna()) &  # Not null
            (df['sentence'] != '') &    # Not empty
            (~df['sentence'].str.contains(r'^ERROR:', case=False, na=False)) &  # Not error message
            (df['response'].str.contains(r'\b(HIGH|LOW)\b', case=False, na=False))  # Response contains HIGH or LOW
        ].copy()
        
        filtered_count = len(df)
        print(f"Filtered {original_count} → {filtered_count} valid rows ({original_count - filtered_count} invalid rows removed)")
        
        # Determine predicted model from filename or add column if missing
        if 'model_name' not in df.columns:
            # Try to extract from filename
            filename = os.path.basename(input_file)
            if '_responses.csv' in filename or '_test.csv' in filename:
                predicted_model = filename.replace('_responses.csv', '').replace('_test.csv', '')
                df['model_name'] = predicted_model
                print(f"Inferred predicted model: {predicted_model}")
            else:
                df['model_name'] = 'unknown'
                print("Warning: Could not determine predicted model, using 'unknown'")
        
        # Get unique predicted model for filename
        predicted_models = df['model_name'].unique()
        if len(predicted_models) == 1:
            predicted_model = predicted_models[0]
        else:
            predicted_model = 'mixed'
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = []
        
        # Run prediction for each predictor temperature
        for pred_temp in predictor_temperatures:
            # Generate output filename with predictor temperature
            temp_str = f"temp{pred_temp}".replace('.', '_')
            output_file = os.path.join(output_dir, f"{self.model_name}_{predicted_model}_{temp_str}.csv")
            
            # Initialize results
            results = []
            total_sentences = len(df)
            
            print(f"\nStarting temperature prediction with {self.model_family}: {self.model_name}")
            print(f"Predictor temperature: {pred_temp}")
            print(f"Analyzing {total_sentences} sentences")
            print(f"Output will be saved to: {output_file}")
        
            for idx, row in df.iterrows():
                print(f"Progress: {idx + 1}/{total_sentences} - {row['subject']}, {row['prompt_type']}")
                
                try:
                    # Create prediction prompt
                    prompt = self.prompt_template.format(
                        sent=row['sentence'],
                        type=row['prompt_type'],
                        subject=row['subject']
                    )
                    
                    # Generate prediction with specified predictor temperature
                    response = self.generate_response(prompt, temperature=pred_temp)
                    
                    if not response.strip():
                        raise RuntimeError("Empty response")
                    
                    # Extract prediction (last word)
                    predicted_temp = response.split()[-1].strip().upper()
                    
                    # Store result
                    result = {
                        'original_sentence': row['sentence'],
                        'subject': row['subject'],
                        'prompt_type': row['prompt_type'],
                        'predicted_model': row['model_name'],
                        'predictor_model': self.model_name,
                        'predictor_family': self.model_family,
                        'predictor_temperature': pred_temp,
                        'prediction_response': response,
                        'predicted_temperature': predicted_temp
                    }
                    
                    # Add original columns if they exist
                    if 'temperature' in row:
                        result['temperature'] = row['temperature']
                    if 'num_of_runs' in row:
                        result['run_id'] = row['num_of_runs']
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
                    # Add error entry
                    error_result = {
                        'original_sentence': row['sentence'],
                        'subject': row['subject'],
                        'prompt_type': row['prompt_type'],
                        'predicted_model': row['model_name'],
                        'predictor_model': self.model_name,
                        'predictor_family': self.model_family,
                        'predictor_temperature': pred_temp,
                        'prediction_response': f"ERROR: {str(e)}",
                        'predicted_temperature': 'ERROR'
                    }
                    results.append(error_result)
                    continue
                
                # Add delay if specified
                if delay > 0:
                    time.sleep(delay)
            
            # Save results
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_file, index=False)
            
            print(f"Temperature prediction completed for predictor temp {pred_temp}!")
            print(f"Successfully processed: {len([r for r in results if r['predicted_temperature'] != 'ERROR'])} sentences")
            print(f"Errors: {len([r for r in results if r['predicted_temperature'] == 'ERROR'])} sentences")
            print(f"Results saved to: {output_file}")
            
            output_files.append(output_file)
        
        return output_files


def main():
    parser = argparse.ArgumentParser(description="Predict LLM temperatures from generated sentences")
    
    parser.add_argument('--input-file', required=True, 
                       help='Input CSV file with sentences to analyze')
    parser.add_argument('--model-family', required=True, choices=['gpt', 'gemini'],
                       help='Model family for prediction (gpt: OpenAI, gemini: Google)')
    parser.add_argument('--model-name', help='Specific model name for prediction')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for results')
    parser.add_argument('--predictor-temperatures', nargs='+', type=float, default=[0.0],
                       help='Temperatures to use for predictor model (default: [0.0])')
    parser.add_argument('--delay', type=float, default=0,
                       help='Delay between requests (seconds)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return 1
    
    try:
        # Initialize predictor
        predictor = TempPrediction(args.model_family, args.model_name)
        
        # Run temperature prediction
        output_files = predictor.predict_temperatures(
            args.input_file,
            args.output_dir,
            args.predictor_temperatures,
            args.delay
        )
        
        print(f"\nPrediction Summary:")
        print(f"Predictor Model: {predictor.model_family} {predictor.model_name}")
        print(f"Predictor Temperatures: {args.predictor_temperatures}")
        print(f"Input File: {args.input_file}")
        print(f"Output Files: {output_files}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
