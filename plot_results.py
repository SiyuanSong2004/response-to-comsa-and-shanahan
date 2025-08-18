import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Dict, List, Tuple


class ExperimentPlotter:
    
    def __init__(self, selfref_dir: str = "selfref_data", temppred_dir: str = "temppred_data", 
                 output_dir: str = "figures"):
        self.selfref_dir = selfref_dir
        self.temppred_dir = temppred_dir
        self.output_dir = output_dir
        
        self.model_name_mapping = {
            'gpt-4.1-2025-04-14': 'gpt-4.1',
            'gpt-4o-2024-08-06': 'gpt-4o',
            'gemini-2.5-flash': 'gemini-2.5-flash',
            'gemini_2.0_flash': 'gemini-2.0-flash'
        }
        
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_plotting_style()
    
    def _setup_plotting_style(self):
        plt.rcParams.update({
            'font.size': 16,        
            'axes.titlesize': 26,  
            'axes.labelsize': 16,   
            'xtick.labelsize': 14,  
            'ytick.labelsize': 28,  
            'legend.fontsize': 24  
        })
    
    @staticmethod
    def classify_temp(value):
        if pd.isna(value):
            return 'error'
            
        value_str = str(value).upper().strip()
        
        has_high = 'HIGH' in value_str
        has_low = 'LOW' in value_str
        
        if (has_high and has_low) or (not has_high and not has_low):
            return 'error'
        elif has_low:
            return 0
        else:  
            return 1
    
    def load_experiment1_data(self):
        print("Loading Experiment 1 data...")
        exp1_data = pd.DataFrame()
        
        if not os.path.exists(self.selfref_dir):
            raise FileNotFoundError(f"Directory not found: {self.selfref_dir}")
        
        for file in os.listdir(self.selfref_dir):
            model_name = file.split('_')[0]
            file_path = os.path.join(self.selfref_dir, file)
            
            try:
                df = pd.read_csv(file_path)
                df['model'] = model_name
                
                df = df[
                    (df['sentence'].notna()) & 
                    (df['sentence'] != '') &    
                    (df['sentence'] != 'Sentence:') &
                    (~df['sentence'].str.contains(r'^ERROR:', case=False, na=False)) &
                    (df['response'].str.contains(r'\b(HIGH|LOW)\b', case=False, na=False))
                ].copy()
                
                exp1_data = pd.concat([exp1_data, df], ignore_index=True)
                print(f"  Loaded {len(df)} responses from {file}")
                
            except Exception as e:
                print(f"  Error loading {file}: {e}")
                continue
        
        exp1_data['model'] = exp1_data['model'].apply(lambda x: self.model_name_mapping.get(x, x))
        exp1_data['last_sentence'] = exp1_data['response'].apply(lambda x: ' '.join(x.rstrip()                                    .rsplit('\n', 1)[-1]                           .split()[-2:]))
        exp1_data['predicted_temp'] = exp1_data['last_sentence'].apply(self.classify_temp)
        exp1_data = exp1_data[exp1_data['predicted_temp'] != 'error']
        
        print(f"Total: {len(exp1_data)} responses")
        return exp1_data
    
    def load_experiment2_data(self):
        print("Loading Experiment 2 data...")
        
        if not os.path.exists(self.temppred_dir):
            raise FileNotFoundError(f"Directory not found: {self.temppred_dir}")
        
        result_files = [f for f in os.listdir(self.temppred_dir) if f.endswith('.csv')]
        if not result_files:
            raise FileNotFoundError(f"No files found in {self.temppred_dir}")
        
        exp2_data = pd.DataFrame()
        
        for file in result_files:
            file_path = os.path.join(self.temppred_dir, file)
            
            try:
                df = pd.read_csv(file_path)
                
                if 'predicted_model' not in df.columns:
                    continue
                
                predicted_model = df['predicted_model'].unique()[0]
                original_file = os.path.join(self.selfref_dir, f"{predicted_model}_responses.csv")
                if not os.path.exists(original_file):
                    continue
                
                original_data = pd.read_csv(original_file)
                original_data = original_data[
                    (original_data['sentence'].notna()) &
                    (original_data['sentence'] != '') &
                    (original_data['sentence'] != 'Sentence:') &
                    (~original_data['sentence'].str.contains(r'^ERROR:', case=False, na=False)) &
                    (original_data['response'].str.contains(r'\b(HIGH|LOW)\b', case=False, na=False))
                ].copy()
                
                original_data.drop(columns=['predicted_temperature', 'temperature'], inplace=True, errors='ignore')
                original_data['last_sentence'] = original_data['response'].apply(lambda x: ' '.join(x.rstrip()                                    .rsplit('\n', 1)[-1]                           .split()[-2:]))
                original_data['self_ref_temp'] = original_data['last_sentence'].apply(self.classify_temp)
                
                df = df[df['original_sentence'] != 'Sentence:']
                df = df.reset_index(drop=True)
                original_data = original_data.reset_index(drop=True)
                
                overlapping_cols = df.columns.intersection(original_data.columns)
                original_data_clean = original_data.drop(columns=overlapping_cols)
                df_merged = pd.concat([df, original_data_clean], axis=1)
                df_merged = df_merged[df_merged['self_ref_temp'] != 'error']
                
                exp2_data = pd.concat([exp2_data, df_merged], ignore_index=True)
                print(f"  Loaded {len(df_merged)} predictions from {file}")
                
            except Exception as e:
                print(f"  Error loading {file}: {e}")
                continue
        
        print(f"Total: {len(exp2_data)} predictions")
        
        exp2_data['predicted_model'] = exp2_data['predicted_model'].apply(lambda x: self.model_name_mapping.get(x, x))
        exp2_data['predictor_model'] = exp2_data['predictor_model'].apply(lambda x: self.model_name_mapping.get(x, x))
        return exp2_data
    
    def plot_experiment1(self, data, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'exp1_result.pdf')
        
        print("Generating Experiment 1 plot...")
        
        label_categorical_order = [
            'factual - elephants', 'factual - unicorns', 'factual - murlocs',
            'normal - elephants', 'normal - unicorns', 'normal - murlocs',
            'crazy - elephants', 'crazy - unicorns', 'crazy - murlocs'
        ]
        
        data = data.copy()
        data.sort_values(by=['model'], inplace=True)
        models = sorted(data['model'].unique())
        data['prompt_type'] = data['prompt_type'].str.replace('creative', 'crazy', case=False, regex=False)
        
        grouped_data = pd.DataFrame()
        
        for model in models:
            model_data = data[data['model'] == model].copy()
            print(f'  {model}: {len(model_data)} data points')
            
            model_data['predicted_temp'] = pd.to_numeric(model_data['predicted_temp'], errors='coerce')
            grouped_model_data = model_data.groupby(
                ['model', 'prompt_type', 'subject', 'temperature']
            )[['predicted_temp']].mean().reset_index()
            grouped_data = pd.concat([grouped_data, grouped_model_data], ignore_index=True)
        fig, axs = plt.subplots(2, 2, figsize=(15, 12), sharey='row')
        axs = axs.flatten()
        
        pivot_tables = []
        
        for model in models:
            model_df = grouped_data[grouped_data['model'] == model].copy()
            model_df['label'] = model_df['prompt_type'] + ' - ' + model_df['subject']
            model_df['label'] = pd.Categorical(model_df['label'], categories=label_categorical_order, ordered=True)
            model_df = model_df.sort_values(by='label')
            
            pivot_table = model_df.pivot(index='label', columns='temperature', values='predicted_temp')
            pivot_tables.append(pivot_table)
        
        vmin, vmax = 0, 1
        
        for ax, model, pivot_table in zip(axs, models, pivot_tables):
            labels = pivot_table.index.tolist()
            temps = pivot_table.columns.tolist()
            
            im = ax.imshow(pivot_table.values, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
            
            ax.set_xticks(np.arange(len(temps)))
            ax.set_xticklabels(temps)
            ax.tick_params(axis='x', labelbottom=True)
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels)
            
            ax.set_title(f'{model}')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        fig.subplots_adjust(bottom=0.1)
        plt.tight_layout()
        cbar = fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.05, pad=0.1)
        cbar.set_label('Predicted Temp', fontsize=26)
        
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"  Saved Experiment 1 plot to: {save_path}")

    def plot_experiment2(self, exp1_data, exp2_data, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'exp2_result.pdf')
        
        print("Generating Experiment 2 plot...")
        
        exp2_data['predicted_temperature'] = exp2_data['prediction_response'].apply(lambda x: ' '.join(x.rstrip()                                    .rsplit('\n', 1)[-1]                           .split()[-2:]))
        exp2_data['predicted_temperature_clean'] = exp2_data['predicted_temperature'].apply(self.classify_temp)
        exp2_data_clean = exp2_data[exp2_data['predicted_temperature_clean'] != 'error']
        exp2_data_clean = exp2_data_clean[['subject', 'prompt_type', 'predicted_model', 'predictor_model', 'temperature', 'predicted_temperature_clean']]
        exp2_data_clean['prediction_type'] = exp2_data_clean['predicted_model'] == exp2_data_clean['predictor_model']
        exp2_data_clean['prediction_type'] = exp2_data_clean['prediction_type'].replace({True: 'within-model predict', False: 'across-model predict'})
        
        exp1_data_clean = exp1_data.copy()
        exp1_data_clean['predicted_model'] = exp1_data_clean['model']
        exp1_data_clean['predictor_model'] = exp1_data_clean['model']
        exp1_data_clean['predicted_temperature_clean'] = exp1_data_clean['predicted_temp']
        exp1_data_clean = exp1_data_clean[['subject', 'prompt_type', 'predicted_model', 'predictor_model', 'temperature', 'predicted_temperature_clean']]
        exp1_data_clean['prediction_type'] = 'self-reflect'
        
        all_clean = pd.concat([exp1_data_clean, exp2_data_clean], ignore_index=True)
        
        def process_temperature(temp):
            if temp <= 0.5:
                return 0
            elif temp >= 1.5:
                return 1
            else:
                return 'medium'
        
        all_clean['temperature_type'] = all_clean['temperature'].astype(float).apply(process_temperature)
        all_clean = all_clean[all_clean['temperature_type'] != 'medium']
        
        result_df = pd.DataFrame()
        
        unique_conditions = all_clean[['predictor_model', 'predicted_model', 'prediction_type']].drop_duplicates()
        
        for (predictor_model, predicted_model, prediction_type) in unique_conditions.itertuples(index=False):
            subset = all_clean[
                (all_clean['predictor_model'] == predictor_model) & 
                (all_clean['predicted_model'] == predicted_model) &
                (all_clean['prediction_type'] == prediction_type)
            ]
            
            if len(subset) > 0:
                subset_numeric = subset.copy()
                subset_numeric['predicted_temperature_clean'] = pd.to_numeric(
                    subset_numeric['predicted_temperature_clean'], errors='coerce'
                )
                
                consistent_count = (subset_numeric['predicted_temperature_clean'] == subset_numeric['temperature_type']).sum()
                total_count = len(subset_numeric)
                acc = consistent_count / total_count if total_count > 0 else 0
            else:
                acc = 0
                total_count = 0
            
            new_row = pd.DataFrame([{
                'predictor_model': predictor_model,
                'predicted_model': predicted_model,
                'prediction_type': prediction_type,
                'accuracy': acc,
                'total_count': total_count
            }])
            result_df = pd.concat([result_df, new_row], ignore_index=True)
        
        fig, ax = plt.subplots(figsize=(15, 12))
        
        plt.rcParams.update({
            'font.size': 16,        
            'axes.titlesize': 28,  
            'axes.labelsize': 16,   
            'xtick.labelsize': 28,  
            'ytick.labelsize': 14,  
            'legend.fontsize': 24  
        })
        
        prediction_types = ['self-reflect', 'within-model predict', 'across-model predict']
        color_map = {
            'self-reflect': '#d76f56',  
            'within-model predict': '#be76c3',  
            'across-model predict': '#757575' 
        }
        
        predicted_models = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gpt-4.1', 'gpt-4o']
        predicted_models = [m for m in predicted_models if m in result_df['predicted_model'].unique()]
        bar_width = 0.2
        group_spacing = 0.5
        
        x_pos = 0
        group_centers = []
        
        for predicted_model in predicted_models:
            group_positions = []
            
            subset_group = result_df[result_df['predicted_model'] == predicted_model].copy()
            subset_group['type_order'] = subset_group['prediction_type'].apply(
                lambda x: prediction_types.index(x) if x in prediction_types else 999
            )
            subset_group = subset_group.sort_values(by='type_order').head(5)
            
            for _, row in subset_group.iterrows():
                acc = row['accuracy']
                pt = row['prediction_type']
                predictor = row['predictor_model']
                predictor_label = predictor
                
                ax.bar(x_pos, acc, bar_width, color=color_map[pt], alpha=0.8)
                
                ax.text(x_pos, acc + 0.01, f'{acc:.2f}', ha='center', va='bottom', 
                       fontsize=18, rotation=90)
                
                ax.text(x_pos, acc / 2, predictor_label, ha='center', va='center', 
                       fontsize=20, rotation=90, color='white', weight='bold')
                
                group_positions.append(x_pos)
                x_pos += bar_width + 0.02
            
            group_center = np.mean(group_positions)
            group_centers.append(group_center)
            x_pos += group_spacing
        
        ax.set_xticks([]) 
        ax.set_xlabel('Predicted Model', labelpad=40, fontsize=28)
        ax.set_ylabel('Accuracy', fontsize=28)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, center in enumerate(group_centers):
            ax.text(center, -0.01, predicted_models[i],
                   ha='center', va='top', transform=ax.transData, fontsize=24)
        
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=color_map[pt], alpha=0.8, label=pt) 
            for pt in prediction_types
        ]
        ax.legend(handles=legend_elements, title=None, loc='upper left', fontsize=34)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5)
        
        plt.tight_layout()
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"  Saved Experiment 2 plot to: {save_path}")

    def generate_all_plots(self, save_exp1=None, save_exp2=None):
        print("=== Generating All Plots ===")
        
        exp1_data = self.load_experiment1_data()
        exp2_data = self.load_experiment2_data()
        
        print(f"\nSummary: {len(exp1_data)} self-reflection, {len(exp2_data)} cross-model predictions")
        
        print(f"\n=== Experiment 1 ===")
        self.plot_experiment1(exp1_data, save_exp1)
        
        print(f"\n=== Experiment 2 ===")
        self.plot_experiment2(exp1_data, exp2_data, save_exp2)
        
        print(f"\n=== Done! ===")


def main():
    parser = argparse.ArgumentParser(description="Generate plots for LLM temperature prediction experiments")
    
    parser.add_argument('--selfref-dir', default='selfref_data', help='Self-reflection data directory')
    parser.add_argument('--temppred-dir', default='temppred_data', help='Temperature prediction data directory')
    parser.add_argument('--output-dir', default='figures', help='Output directory')
    parser.add_argument('--exp1-only', action='store_true', help='Generate only Experiment 1 plot')
    parser.add_argument('--exp2-only', action='store_true', help='Generate only Experiment 2 plot')
    parser.add_argument('--exp1-output', help='Custom output path for Experiment 1')
    parser.add_argument('--exp2-output', help='Custom output path for Experiment 2')
    
    args = parser.parse_args()
    
    plotter = ExperimentPlotter(args.selfref_dir, args.temppred_dir, args.output_dir)
    
    try:
        if args.exp1_only:
            print("=== Experiment 1 only ===")
            exp1_data = plotter.load_experiment1_data()
            plotter.plot_experiment1(exp1_data, args.exp1_output)
            
        elif args.exp2_only:
            print("=== Experiment 2 only ===")
            exp1_data = plotter.load_experiment1_data()
            exp2_data = plotter.load_experiment2_data()
            plotter.plot_experiment2(exp1_data, exp2_data, args.exp2_output)
            
        else:
            plotter.generate_all_plots(args.exp1_output, args.exp2_output)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 