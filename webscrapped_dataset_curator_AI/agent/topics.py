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
