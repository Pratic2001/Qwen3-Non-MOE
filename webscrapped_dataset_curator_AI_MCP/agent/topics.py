"""
topics.py

Seed topics per category. These do two jobs:
  1. Bootstrap the very first round of queries for a fresh run.
  2. Act as a fallback if the Ollama query-planner call fails for any reason,
     so the agent degrades gracefully instead of stalling.

Feel free to extend these lists -- the agent will keep expanding beyond them
on its own (the planner is explicitly told to avoid repeating recent
queries), but a broader seed list gives it a better starting spread.
"""

TOPIC_SEEDS = {
    "chat": ["general conversation between two people.", "chat conversation.", "Role play chat", "Human-AI conversation"],
    "web": [
        "explainer how the stock market works", "history of the printing press",
        "climate change ocean currents explained", "how vaccines are developed",
        "urban planning traffic congestion solutions", "renewable energy grid storage",
        "psychology of decision making", "history of the internet protocol",
    ],
    "knowledge": [
        "history of ancient Rome government", "biology of coral reef ecosystems",
        "quantum mechanics double slit experiment", "world war two eastern front",
        "geology of plate tectonics", "history of the printing revolution",
        "astronomy black hole formation", "linguistics origin of language families",
    ],
    "reasoning": [
        "logic puzzle step by step solution", "case study root cause analysis",
        "debate pros and cons universal basic income", "ethical dilemma trolley problem analysis",
        "step by step troubleshooting guide software bug", "strategic decision making business case study",
    ],
    "code": [
        "python design patterns tutorial", "rust ownership borrowing explained",
        "how to implement a binary search tree", "REST API best practices tutorial",
        "concurrency patterns in go", "sql query optimization techniques",
        "javascript async await tutorial", "algorithm dynamic programming examples",
    ],
    "math": [
        "calculus integration by parts examples", "linear algebra eigenvalues explained",
        "probability bayes theorem worked examples", "number theory modular arithmetic",
        "combinatorics counting problems solutions", "differential equations worked examples",
    ],
    "science": [
        "physics thermodynamics second law explained", "chemistry reaction kinetics tutorial",
        "biology cell signaling pathways", "astronomy exoplanet detection methods",
    ],
}

# Short keyword queries for auto-discovering datasets on Hugging Face
# Hub / Kaggle (agent/public_sources.py's discover_hf_datasets /
# discover_kaggle_datasets). Deliberately NOT the same as TOPIC_SEEDS
# above: those are full natural-language sentences meant to seed the
# web-search query planner, which is exactly the wrong shape for hub
# search -- both HF's `list_datasets(search=...)` and Kaggle's dataset
# search do simple keyword/substring matching against dataset names and
# tags, so a 10-word sentence like "calculus integration by parts
# examples linear algebra eigenvalues explained" matches nothing and
# silently returns zero datasets every time. These are short, 1-3 word
# terms chosen to actually hit real dataset names/tags on each hub.
# `run_public_sources_for_category` tries each in order and merges/dedups
# results (up to --public-discover-limit) rather than using just one, so
# a single overly-specific or unlucky term doesn't zero out discovery for
# a whole category.
HUB_SEARCH_KEYWORDS = {
    "web": ["web text", "common crawl", "openwebtext"],
    "knowledge": ["wikipedia", "trivia qa", "encyclopedia"],
    "reasoning": ["reasoning", "chain of thought", "logic"],
    "code": ["code", "github code", "programming"],
    "math": ["math", "mathematics", "gsm8k"],
    "science": ["science qa", "science", "physics"],
}
