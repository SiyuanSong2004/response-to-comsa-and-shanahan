#!/usr/bin/env python3
"""
Self-reflection experiments for LLM temperature prediction.
Supports OpenAI GPT and Google Gemini models.
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


class ModelSelfReflection:
    def __init__(self, model_family: str, model_name: Optional[str] = None):
        self.model_family = model_family.lower()
        self.model_name = model_name
        self.client = None
        
        self.subjects = ['elephants', 'unicorns', 'murlocs']
        self.prompt_templates = {
            'factual': "Write a factual sentence about {subject}. Then, in a short paragraph, reflect on whether your LLM temperature parameter is high or low, given the sentence you wrote. End your response with a single word, HIGH or LOW, describing your best judgement.",
            'normal': "Write a sentence about {subject}. Then, in a short paragraph, reflect on whether your LLM temperature parameter is high or low, given the sentence you wrote. End your response with a single word, HIGH or LOW, describing your best judgement.",
            'creative': "Write a crazy sentence about {subject}. Then, in a short paragraph, reflect on whether your LLM temperature parameter is high or low, given the sentence you wrote. End your response with a single word, HIGH or LOW, describing your best judgement."
        }
        
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
    
    def generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
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
    
    def run_experiment(self, temperature_range=(0, 2.0, 0.1), prompt_types=None, subjects=None, 
                      num_runs=3, output_file=None, delay=0):
        if prompt_types is None:
            prompt_types = ['factual', 'normal', 'creative']
        if subjects is None:
            subjects = self.subjects
        if output_file is None:
            output_file = f"selfref_data/{self.model_name}_responses.csv"
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        result_df = pd.DataFrame(columns=[
            "num_of_runs", "temperature", "subject", "prompt_type", "response", 
            "sentence", "predicted_temperature", "model_family", "model_name"
        ])
        
        start, end, step = temperature_range
        temperatures = [round(start + i * step, 2) for i in range(int((end - start) / step) + 1)]
        
        total_experiments = len(temperatures) * len(subjects) * len(prompt_types) * num_runs
        current_experiment = 0
        
        print(f"Total runs: {total_experiments}")
        
        for temp in temperatures:
            for subject in subjects:
                for prompt_type in prompt_types:
                    for run_id in range(1, num_runs + 1):
                        current_experiment += 1
                        print(f"Progress: {current_experiment}/{total_experiments} - "
                              f"T={temp}, {subject}, {prompt_type}, run {run_id}")
                        
                        try:
                            prompt = self.prompt_templates[prompt_type].format(subject=subject)
                            response_text = self.generate_response(prompt, temperature=temp)
                            
                            if not response_text.strip():
                                raise RuntimeError("Empty response")
                            
                            predicted_temperature = response_text.split()[-1].strip().upper()
                            sentence = response_text.split('\n')[0] if '\n' in response_text else response_text
                            
                            new_row = {
                                "num_of_runs": run_id,
                                "temperature": temp,
                                "subject": subject,
                                "prompt_type": prompt_type,
                                "response": response_text,
                                "sentence": sentence,
                                "predicted_temperature": predicted_temperature,
                                "model_family": self.model_family,
                                "model_name": self.model_name
                            }
                            result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
                            
                        except Exception as e:
                            print(f"Error: {e}")
                            continue
                        
                        if delay > 0:
                            time.sleep(delay)
        
        result_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        return result_df


def run_test_experiment(model_reflection, temperatures, prompt_types=None, subjects=None, 
                       num_runs=1, output_file=None, delay=0):
    if prompt_types is None:
        prompt_types = ['factual', 'normal', 'creative']
    if subjects is None:
        subjects = model_reflection.subjects
    if output_file is None:
        output_file = f"selfref_data/{model_reflection.model_name}_test.csv"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    result_df = pd.DataFrame(columns=[
        "num_of_runs", "temperature", "subject", "prompt_type", "response", 
        "sentence", "predicted_temperature", "model_family", "model_name"
    ])
    
    total_experiments = len(temperatures) * len(subjects) * len(prompt_types) * num_runs
    current_experiment = 0
    
    print(f"TEST mode - {model_reflection.model_family}: {model_reflection.model_name}")
    print(f"Temperatures: {temperatures}")
    
    for temp in temperatures:
        for subject in subjects:
            for prompt_type in prompt_types:
                for run_id in range(1, num_runs + 1):
                    current_experiment += 1
                    print(f"{current_experiment}/{total_experiments} - T={temp}, {subject}, {prompt_type}")
                    
                    try:
                        prompt = model_reflection.prompt_templates[prompt_type].format(subject=subject)
                        response_text = model_reflection.generate_response(prompt, temperature=temp)
                        
                        if not response_text.strip():
                            raise RuntimeError("Empty response")
                        
                        predicted_temperature = response_text.split()[-1].strip().upper()
                        sentence = response_text.split('\n')[0] if '\n' in response_text else response_text
                        
                        new_row = {
                            "num_of_runs": run_id,
                            "temperature": temp,
                            "subject": subject,
                            "prompt_type": prompt_type,
                            "response": response_text,
                            "sentence": sentence,
                            "predicted_temperature": predicted_temperature,
                            "model_family": model_reflection.model_family,
                            "model_name": model_reflection.model_name
                        }
                        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
                        
                    except Exception as e:
                        print(f"Error: {e}")
                        continue
                    
                    if delay > 0:
                        time.sleep(delay)
    
    result_df.to_csv(output_file, index=False)
    print(f"Test results saved to {output_file}")
    return result_df


def main():
    parser = argparse.ArgumentParser(description="LLM self-reflection temperature experiments")
    
    parser.add_argument('--model_family', required=True, choices=['gpt', 'gemini'],
                       help='Model family (gpt: OpenAI, gemini: Google)')
    parser.add_argument('--model-name', help='Specific model id')
    parser.add_argument('--temp-start', type=float, default=0.0, help='Start temperature')
    parser.add_argument('--temp-end', type=float, default=2.0, help='End temperature')
    parser.add_argument('--temp-step', type=float, default=0.1, help='Temperature step')
    parser.add_argument('--prompt-types', nargs='+', default=['factual', 'normal', 'creative'],
                       choices=['factual', 'normal', 'creative'], help='Prompt types')
    parser.add_argument('--subjects', nargs='+', default=['elephants', 'unicorns', 'murlocs'],
                       choices=['elephants', 'unicorns', 'murlocs'], help='Subjects')
    parser.add_argument('--num-runs', type=int, default=3, help='Runs per configuration')
    parser.add_argument('--output', help='Output CSV file path')
    parser.add_argument('--delay', type=float, default=0, help='Delay between requests (seconds)')
    parser.add_argument('--test', action='store_true', 
                       help='Test mode: elephant, 1 run, temps [0.5, 1.5, 0.5]')
    
    args = parser.parse_args()
    
    if args.test:
        print("TEST MODE: elephant subject, 1 run, temperatures [0.5, 1.0, 1.5]")
        args.subjects = ['elephants']
        args.num_runs = 1
    
    try:
        model_reflection = ModelSelfReflection(args.model_family, args.model_name)
        
        if args.test:
            results = run_test_experiment(
                model_reflection, 
                temperatures=[0.5, 1.0, 1.5],
                prompt_types=args.prompt_types,
                subjects=args.subjects,
                num_runs=args.num_runs,
                output_file=args.output,
                delay=args.delay
            )
        else:
            results = model_reflection.run_experiment(
                temperature_range=(args.temp_start, args.temp_end, args.temp_step),
                prompt_types=args.prompt_types,
                subjects=args.subjects,
                num_runs=args.num_runs,
                output_file=args.output,
                delay=args.delay
            )
        
        print(f"\nSummary: {model_reflection.model_family} {model_reflection.model_name}")
        print(f"Generated {len(results)} responses")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())