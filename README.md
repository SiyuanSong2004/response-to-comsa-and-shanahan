# response-to-comsa-and-shanahan
code and data for paper *Privileged Self-Access Matters for Introspection in AI*.

Please cite our paper if you find our materials useful:

```
bibtex
```


### Dependencies
To replicate our analysis, you need basic data processing and plotting modules, including `pandas`, `numpy`, `matplotlib` and `seaborn`.

To replicate our experiments, you need the packages for API access: `openai` (for GPT models) and `google-genai` (for Gemini models). In addition, please add `OPENAI_API_KEY` and `GEMINI_API_KEY` to your environment variables.

### Models
We used the following models in our experiments:

| **Model Name** | **Model ID** |
|----------------|--------------|
| GPT-4o | gpt-4o-2024-08-06 |
| GPT-4.1 | gpt-4.1-2025-04-14 |
| Gemini-2.0-flash | gemini-2.0-flash |
| Gemini-2.5-flash | gemini-2.5-flash |

\*Please note that all our experiments were conducted in late June, 2025. Some of the models we used might be made unavailable due to GPT-5 updates. As of the time of this paper's completion, the models we used (or their updated versions) are still available through OpenAI and Google's API services.

### Replication 

To replicate our complete experiments, use the scripts `run_exp1.sh` and `run_exp2.sh`.

For custom experimental settings, use the individual python scripts directly:

```bash
# Self-reflection experiment with custom parameters
python selfreflection.py --model_family gpt --model-name gpt-4o-2024-08-06 --temp-start 0.0 --temp-end 1.0 --temp-step 0.1

# Cross-model prediction experiment
python tempprediction.py --input-file selfref_data/gpt-4o-2024-08-06_responses.csv --model-family gemini --model-name gemini-2.0-flash --output-dir temppred_data
```

To replicate our analysis and visualization, run the following command:

`python plot_results.py`

### References

This paper is a response to Comşa & Shanahan's work on LLM introspection. Please consider citing their original paper:

```
@article{comsa2025does,
  title={Does It Make Sense to Speak of Introspection in Large Language Models?},
  author={Comsa, Iulia M and Shanahan, Murray},
  journal={arXiv preprint arXiv:2506.05068},
  year={2025}
}
```

Please cite our papers if you find our materials useful:

- This paper:
```
[citation to be added]
```

- Our earlier work on introspection about knowledge of language:**
```
@article{song2025language,
  title={Language models fail to introspect about their knowledge of language},
  author={Song, Siyuan and Hu, Jennifer and Mahowald, Kyle},
  journal={arXiv preprint arXiv:2503.07513},
  year={2025}
}
```
