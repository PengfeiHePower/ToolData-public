<img alt="TRAJECT-Bench — Trajectory-Aware Tool-Use Evaluation for Agents" src="assets/benchmark_logo9.png" width="700">

[![Name](https://img.shields.io/badge/Benchmark-TRAJECT--Bench-7c3aed)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
[![HF Dataset](https://img.shields.io/badge/HuggingFace-dataset-blue)](https://huggingface.co/datasets/bigboss24/tool_usage)

> A comprehensive benchmark for evaluating tool-using language models with parallel and sequential tool-calling trajectories across practical domains.

## 🚀 Quickstart (TL;DR) (to finish soon)

```bash
git clone <repository-url>
cd ToolData-public
pip install -r requirements.txt

# Simple query evaluation
python evaluation/tool_evaluation_model.py

# Agentic evaluation (ReAct)
python evaluation/tool_evaluation_react.py

# Retrieval-based evaluation
python evaluation/tool_evaluation_retrieve.py
```

✨ A comprehensive benchmark for evaluating tool-using language models across multiple practical domains, designed to test models' ability to effectively utilize external tools for real-world tasks.

## 🎯 Overview

**🌟 Hightlights:**
- 🔧 **Multi-Tool Selection**: Combining multiple executable, production-style tools to solve complex queries  
- 🎯 **Practical Task Solving**: Incorporating practical tasks across diverse domains
- 🛤️ **Trajectory Structure Support**: First to evaluate both parallel and sequential tool-calling trajectories
- 📊 **Advanced Metrics**: Trajectory-aware evaluation metrics for comprehensive assessment
- 📈 **Query Difficulty Control**: Structured difficulty levels for progressive evaluation
- 🎲 **Multiple Tool-Pool Regimes**: Support for various tool pool setups, including whole toolset (mixture of tools), domain-specific tools, tool retrieval, small-scale fixed tool pool
- 🤖 **Agentic Method Support**: Evaluation frameworks for ReAct and other agentic approaches

<!-- Comparison with other tool benchmarks: -->

| Benchmark                         | Practical tools | Large&diverse tool| Trajectory structure<sup>1</sup> | Trajectory scaling<sup>2</sup> | Trajectory-aware metrics<sup>3</sup> | Query difficulty control | Tool-pool regimes<sup>4</sup> | Agentic methods |
|-----------------------------------|-----------------|-----------------|----------------------|-----------------------------|--------------------------|---------------------------|----------------------------|----------------|
| MetaTool                          | ✅              | ❌              | ❌                   | ❌                          | ❌                       | ❌                        | ❌                         | ❌             |
| API-Bank                          | ✅              | ❌              | ❌                   | ❌                          | ❌                       | ❌                        | ❌                         | ❌             |
| ToolBench                         | ✅              | ✅              | ❌                   | ❌                          | ❌                       | ❌                        | ❌                         | ✅             |
| Gorilla                           | ✅              | ✅              | ❌                   | ❌                          | ❌                       | ❌                        | ✅                         | ❌             |
| Berkeley Function-Calling (BFCL)  | ✅              | ✅              | ❌                   | ❌                          | ❌                       | ❌                        | ❌                         | ❌             |
| ToolQA                            | ❌              | ❌              | ❌                   | ❌                          | ❌                       | ✅                        | ❌                         | ✅             |
| **TRAJECT-Bench (ours)**          | **✅**          | **✅**          | **✅**               | **✅**                       | **✅**                   | **✅**                    | **✅**                      | **✅**          |

<sup>1</sup> **Trajectory structure**: Evaluates support for different tool-calling patterns, including parallel (independent tools) and sequential (dependent tool chains) execution strategies  
<sup>2</sup> **Trajectory scaling**: Tests model performance across varying task complexity levels, from simple 3-tool scenarios to complex 10+ tool orchestration  
<sup>3</sup> **Trajectory-aware metrics**: Provides comprehensive evaluation beyond final results, measuring the quality of the entire tool-calling trajectory  
<sup>4</sup> **Tool-pool regimes**: Supports diverse evaluation strategies including whole toolset, domain-specific pools, retrieval-based selection, and fixed small-scale tool pools



### 🔗 Quick Links

- 📊 Dataset on Hugging Face: [`bigboss24/TRAJECT-Bench`](https://huggingface.co/datasets/bigboss24/TRAJECT-Bench)
- 📁 Public data folder: `public_data/v2/`
- 🧪 Evaluation scripts: `evaluation/tool_evaluation_model.py`, `evaluation/tool_evaluation_react.py`, `evaluation/tool_evaluation_retrieve.py` (to finish soon)

## 🏗️ Benchmark Structure

### 🌍 Domains Covered

The benchmark covers 10 carefully selected domains that require external tools rather than internal model capabilities:

- ✈️ **Travel**: Hotel booking, flight information, trip planning, transportation
- 🌤️ **Weather**: Forecasts, meteorological data, climate information
- 💰 **Finance**: Market data, trading platforms, cryptocurrency, banking
- 🗺️ **Mapping**: Location services, routing, geographic data
- 🎵 **Music**: Streaming, lyrics, artist information, music metadata
- 📰 **News & Media**: News articles, multimedia content, current events
- 📚 **Education**: Learning resources, academic data, research tools
- 📧 **Email**: Communication, automation, folder management
- 🎮 **Gaming**: Game data, statistics, account management
- 🛒 **eCommerce**: Online shopping, inventory, product information

### 📁 Data Organization

```
public_data/
├── v2/
│   ├── tools/           # Tool definitions and APIs for each domain
│   └── parallel/        # Parallel trajectory sets
│       ├── Education/   # Education domain
│           ├──hard_ver.json   # Hard queries
│           └──simple_ver.json # Simple queries
│       ├── Finance/     # Domain-specific test cases
│       ├── Travel/      # Multi-tool scenarios
│       └── ...          # Other domains
```

### 📊 Dataset at a Glance (for parallel, sequential on the way)

- 🌍 **10 domains**: Education, Email, Finance, Gaming, Mapping, Music, News_Media, Travel, Weather, eCommerce
- 🎯 **50 practical task types**: 5 representative user task types for each domain
- 🛤️ **2,000 total parallel trajectories** with different depth: 5 trajectory per number of tools (from 3-10) for each task
- 📝 **2,000 queries per difficulty**: one simple and one hard for each trajectory
- 🔢 **4,000 total parallel trajectories**: simple (2,000) + hard (2,000)
- 🛠️ Tool metadata: `public_data/v2/tools/*.json`

### 🔄 Query/Trajectories Types

We take a trajectory->query strategy

1. 🔗 **Parallel tool-calling trajectory**: Independent tools collaborate for one task. Test sub-task planning and tool-usage capability.
2. ⏭️ **Sequential tool-calling trajectory**: Trajectories with strong dependency among tools, i.e. latter tools require former tools' outputs.
3. 📝 **Simple Queries**: Straightforward and clear instructions requiring multiple tools
4. 🧠 **Hard Queries**: Indirect and indicating queries that are challenging, but avoid vagueness and much openness.

### 📋 Query JSON Structure

Each entry in `public_data/v2/parallel/<Domain>/*.json` follows this structure:

- **query**: Natural language instruction
- **tool_list**: Tool-calling trajectory for solving the query
  - **tool name**: `<parent tool name>: <API name>`
  - **tool description**: Brief description of the API endpoint
  - **required parameters**: List of objects `{ "name": string, "value": string }`
  - **optional parameters**: List of objects `{ "name": string, "value": string }` (can be empty)
  - **parent tool name**: Name of the tool/provider
  - **API name**: Endpoint name
  - **domain name**: Domain category
- **trajectory_type**: `parallel` or `sequential`
- **task_name**: Short title of the general task type
- **task_description**: Extended description of the task type

Example:

```json
{
  "query": "First, retrieve the post with ID P36288 from FluentMe. Then, get information about the university with ID 394596. Finally, show me a list of video game adaptations from the Transmedia Storytelling database.",
  "tool_list": [
    {
      "tool name": "thefluentme: Get post by id",
      "tool description": "Returns a specific post using its post_id identifier...",
      "required parameters": [{ "name": "post_id", "value": "P36288" }],
      "optional parameters": [],
      "parent tool name": "thefluentme",
      "API name": "Get post by id",
      "domain name": "Education"
    },
    ...
  ],
  "trajectory_type": "parallel",
  "task_name": "Language and Culture Homework Helper",
  "task_description": "This task assists students with language and culture assignments. It involves finding definitions, translations, character information, and contextual examples for various languages."
}
```

### 📝 Minimal sequential example (illustrative)

```json
{
  "query": "Find a flight from NYC to SFO next Friday and then get the 3-day weather forecast for the arrival dates in SFO.",
  "tool_list": [
    {
      "tool name": "skyscanner_flights: search_flights",
      "required parameters": [{ "name": "origin", "value": "NYC" }, { "name": "destination", "value": "SFO" }, { "name": "date", "value": "<next_friday>" }],
      "optional parameters": [],
      "parent tool name": "skyscanner_flights",
      "API name": "search_flights",
      "domain name": "Travel"
    },
    {
      "tool name": "forecast_lookup: 3day",
      "required parameters": [{ "name": "location", "value": "SFO" }, { "name": "date", "value": "<arrival_date>" }],
      "optional parameters": [],
      "parent tool name": "forecast_lookup",
      "API name": "3day",
      "domain name": "Weather"
    }
  ],
  "trajectory_type": "sequential",
  "task_name": "Travel planning with weather guardrails",
  "task_description": "Plan travel and check weather constraints based on retrieved itinerary."
}
```

## 📊 Evaluation Metrics

The benchmark evaluates models on:

- 🎯 **Exact Match**: Exact same selected tools with/without order consideration
- 🔍 **Tool includion**: Fraction of ground-truth tools that appear in the predicted trajectory
- ⚙️ **Tool parameterization**: Proper usage of tools, including input values and formats
- 🏆 **Trajectory win rate**: Given the predicted vs. reference trajectories, an LLM judge picks which better satisfies the task
- ✅ **Trajectory satisfication**: Judge rates whether the predicted trajectory sufficiently satisfies the query (LLM judge)
- 🎯 **Solution Accuracy**: Compare predicted solution with the ground truth solution (executation involved)

📋 Detailed metrics in `utils/metrics.py`

## 🚀 Getting Started

### 📋 Prerequisites

- 🐍 Python 3.10+
- 📦 Required packages (see requirements.txt)

### ⚙️ Installation

```bash
git clone <repository-url>
cd ToolData-public
pip install -r requirements.txt
```

### 🧪 Running Evaluations (to finish soon)

```bash
# Simple query evaluation with direct prompting
python evaluation/tool_evaluation_model.py

# Agentic evaluation (ReAct)
python evaluation/tool_evaluation_react.py

# Retrieval-based evaluation
python evaluation/tool_evaluation_retrieve.py
```

### 📁 Data Access

The benchmark data is publicly available in the `public_data/` directory:

- 🛠️ **Tool Definitions**: JSON files containing API specifications and parameters
- 📝 **Test Queries**: Curated queries with ground truth tool execution paths


### 📊 Data Source

Primary dataset hosting:

- 🤗 Hugging Face dataset: [`bigboss24/tool_usage`](https://huggingface.co/datasets/bigboss24/tool_usage)

<!-- ## Benchmark Design Principles

1. **Practical Focus**: Queries are designed around real-world use cases
2. **Tool Dependency**: All queries require external tools for complete solutions
3. **Multi-Tool Scenarios**: Complex problems requiring tool coordination
4. **Domain Diversity**: Coverage across various practical domains
5. **Scalable Evaluation**: Consistent metrics across different model types -->

<!-- ## Leaderboard (placeholder)

| Model | Tool Selection | Planning | Overall |
|------|-----------------|----------|---------|
| Your Model | — | — | — |

Submit results via PR or issue. We will maintain a simple leaderboard here. -->
## 🔮 Future update

- ⭐ (Highest priority) Complete evaluation pipeline, add detailed instructions
- ⭐ (Highest priority) Add sequential trajectory and query data
- 🌍 Add more domains and tasks


## 🤝 Contributing

We welcome contributions to improve the benchmark:

- 🌍 Additional domains and tools
- 📊 Enhanced evaluation metrics
- 🔄 New query types and scenarios
- ⚡ Performance optimizations

<!-- ## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{toolbenchmark2024,
  title={Tool-Using Language Model Benchmark},
  author={ToolBench Team},
  year={2024},
  url={https://github.com/your-repo}
}
``` -->

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or feedback about the benchmark, please open an issue on GitHub or contact the maintainers.

---

*This benchmark is designed to advance the field of tool-using language models by providing comprehensive evaluation scenarios that reflect real-world tool usage patterns.*
