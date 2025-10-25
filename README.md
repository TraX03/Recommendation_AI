# Recommendation_AI
A Python-based hybrid recommendation and meal planning system designed for personalised recipe and content suggestions. Combines content-based filtering (TF-IDF & embeddings), collaborative filtering, and Q-learning agents for adaptive recommendation strategies.<br>

Features include:
- Personalised content recommendations: Supports multiple content types (recipes, community posts, tips, discussions).
  
- Hybrid scoring:
  Combines item similarity, user profiles, and collaborative filtering into a weighted hybrid score.
- Meal planning:
  Generates meal plans based on user preferences, dietary restrictions, regional cuisine preferences, and inventory availability.
- Inventory-aware suggestions:
  Prioritises recipes using available ingredients or near-expiry items.
- Cold-start handling: Uses onboarding suggestions and fallback strategies when insufficient interaction data is available.
- Adaptive fallback strategies: Q-learning agent selects optimal fallback methods (e.g., relaxing region filters, using inferred tags, full dataset exploration).
- Tag and TF-IDF-based content selection: Supports semantic and textual similarity for recipe selection.
