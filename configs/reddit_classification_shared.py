"""
Defines shared constants and mappings for the Reddit classification experiments.

This file centralizes subreddit lists, label-to-ID mappings, and other static
configuration details to ensure consistency across the training, evaluation,
and configuration scripts.
"""
LIFESTYLE_SUBREDDITS = [
    'movies', 'books', 'fitness', 'cooking', 'travel', 'music', 'art', 'history'
]

SCIENCE_TECH_SUBREDDITS = [
    'science', 'technology', 'askscience', 'gadgets', 'space', 'explainlikeimfive'
]


ALL_SUBREDDITS = sorted(list(set(
    LIFESTYLE_SUBREDDITS + 
    SCIENCE_TECH_SUBREDDITS
)))

SUBREDDIT_TO_ID = {name: i for i, name in enumerate(ALL_SUBREDDITS)}
ID_TO_SUBREDDIT = {i: name for i, name in enumerate(ALL_SUBREDDITS)}
NUM_LABELS = len(ALL_SUBREDDITS)
