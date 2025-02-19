import json
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime

#      SG: Total       -0.952
#               Birdies       -0.605
#                Bogeys        0.559
#           SG: Putting       -0.475
#  Greens in Regulation       -0.447
#         Putts per GIR        0.439
# SG: Approach to Green       -0.427
#    Feet of Putts Made       -0.419
#       SG: Off The Tee       -0.400
#            Scrambling       -0.363
#         Double Bogeys        0.294
#      Driving Distance       -0.224
#      Driving Accuracy       -0.216
#            Sand Saves       -0.213
#  SG: Around The Green       -0.203
#              Eagles -       -0.146
#         Longest Drive       -0.026
#                  Pars        0.022
# Weights for different statistical categories
WEIGHTS = {
    "SG: Total": 0.25,              # Strongest correlation (-0.952)
    "Birdie or Better Percentage": 0.15,  # 1st - Most important
    "Bogey Avoidance": 0.12,
    "SG: Putting": 0.18,
    "Greens in Regulation Percentage": 0.1,
    "SG: Approach the Green": 0.12,
    "SG: Off-the-Tee": 0.08,
}

# WEIGHTS = {
#     "SG: Total": 0.25,              # Strongest correlation by far
#     "Scoring Metrics": {
#         "Birdies": 0.15,            # Second strongest correlation
#         "Bogey Avoidance": 0.12,    # Third strongest correlation
#     },
#     "Approach Play": {
#         "SG: Approach": 0.12,       # Strong correlation + key differentiator
#         "Greens in Regulation": 0.10,# Consistent importance
#     },
#     "Putting": {
#         "SG: Putting": 0.10,        # Significant correlation
#         "Putts per GIR": 0.08,      # Important but slightly less
#     },
#     "Tee Game": {
#         "SG: Off The Tee": 0.08,    # Moderate correlation
#     }
# }

def load_player_stats() -> Dict:
    """Load player statistics from JSON file."""
    try:
        with open("pga_stats.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: pga_stats.json not found")
        return {}
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in pga_stats.json")
        return {}


def clean_stat_value(value: str) -> float:
    """Convert stat string values to float, handling percentages and special formats."""
    if isinstance(value, (int, float)):
        return float(value)

    if not value or value == "N/A":
        return 0.0

    # Remove percentage signs and convert to decimal
    if "%" in value:
        return float(value.strip("%")) / 100

    # Handle feet'inches" format (like 15'6" or 35' 1")
    if "'" in value:
        # Remove quotes and split on feet mark
        value = value.replace('"', "")
        feet, inches = value.split("'")
        feet = float(feet.strip())
        inches = float(inches.strip())
        return feet + inches / 12

    return float(value)


def get_field_stat_ranges(players_data: Dict) -> Dict:
    """Calculate the statistical ranges from the field of players."""
    stat_values = {key: [] for key in WEIGHTS.keys()}

    # Collect all values for each stat
    for player_data in players_data.values():
        stats = player_data.get("stats", {})
        if not stats:
            continue

        # Only include players that have all required stats
        has_all_stats = True
        for stat_name in stat_values.keys():
            try:
                value = stats.get(stat_name, {}).get("value", "0")
                if not value or value == "N/A":
                    has_all_stats = False
                    break
                clean_stat_value(value)  # Test if we can convert it
            except (ValueError, TypeError):
                has_all_stats = False
                break

        if has_all_stats:
            for stat_name in stat_values.keys():
                value = stats.get(stat_name, {}).get("value", "0")
                stat_values[stat_name].append(clean_stat_value(value))

    return {
        name: {"min": min(vals), "max": max(vals)}
        for name, vals in stat_values.items()
        if vals
    }


def normalize_stat(value: float, stat_name: str, stat_ranges: Dict) -> float:
    """Normalize a stat value based on the field's statistics."""
    stat_range = stat_ranges.get(stat_name)
    if not stat_range:
        return 0.0

    stat_min = stat_range["min"]
    stat_max = stat_range["max"]

    if stat_max == stat_min:
        return 0.0

    # For stats where lower is better, invert the normalization
    lower_is_better = any(
        phrase in stat_name
        for phrase in ["Approaches from", "Bogey"]  # All approach distances
    )

    normalized = (value - stat_min) / (stat_max - stat_min)
    if lower_is_better:
        normalized = 1 - normalized

    # Clip to [-1, 1] range
    return max(-1.0, min(1.0, normalized * 2 - 1))


def calculate_player_score(player_stats: Dict, name: str, stat_ranges: Dict) -> float:
    """Calculate weighted score for a player based on their stats."""
    stats = player_stats.get("stats", {})
    score = 0
    try:
        for stat_name in WEIGHTS.keys():
            stat_value = clean_stat_value(stats.get(stat_name, {}).get("value", "0"))
            normalized_value = normalize_stat(stat_value, stat_name, stat_ranges)
            score += normalized_value * WEIGHTS[stat_name]
    except (ValueError, TypeError) as e:
        print(f"Warning: Invalid stats for player {name}: {e}")
        return 0

    return score


def rank_players() -> List[Tuple[str, float]]:
    """Rank players based on their weighted scores."""
    players_data = load_player_stats()
    stat_ranges = get_field_stat_ranges(players_data)

    # Calculate scores for each player
    player_scores = []
    for player_name, stats in players_data.items():
        # Skip players with no stats
        if not stats.get("stats"):
            continue

        # Skip players missing any required stats
        player_stats = stats.get("stats", {})
        has_all_stats = all(
            player_stats.get(stat_name, {}).get("value") for stat_name in WEIGHTS.keys()
        )

        if has_all_stats:
            score = calculate_player_score(stats, player_name, stat_ranges)
            player_scores.append((player_name, score))
        else:
            missing_stats = [
                stat_name
                for stat_name in WEIGHTS.keys()
                if not player_stats.get(stat_name, {}).get("value")
            ]
            print(f"Player {player_name} has missing stats: {missing_stats}")

    # Sort players by score in descending order
    ranked_players = sorted(player_scores, key=lambda x: x[1], reverse=True)
    return ranked_players


def calculate_win_probabilities(rankings: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Convert scores to win probabilities using softmax transformation."""
    scores = np.array([score for _, score in rankings])
    
    # Scale the scores to prevent numerical instability
    scaled_scores = scores * 10
    
    # Apply softmax transformation
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
    probabilities = exp_scores / exp_scores.sum()
    
    return [(player, prob * 100) for (player, _), prob in zip(rankings, probabilities)]


def calculate_finish_probabilities(rankings: List[Tuple[str, float]], positions: int = 5) -> List[Tuple[str, float]]:
    """Convert scores to finish probabilities for top N positions using softmax transformation."""
    scores = np.array([score for _, score in rankings])
    
    # Scale the scores to prevent numerical instability
    scaled_scores = scores * 10
    
    # Calculate probabilities for each position
    total_probs = np.zeros(len(rankings))
    
    for pos in range(positions):
        # For each position, exclude players already "taken" by higher positions
        # This is a simplification - we're treating each position independently
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        position_probs = exp_scores / exp_scores.sum()
        
        # Add position probabilities to total
        total_probs += position_probs
    
    # Normalize to ensure no probability exceeds 100%
    total_probs = np.minimum(total_probs, 1.0)
    
    return [(player, prob * 100) for (player, _), prob in zip(rankings, total_probs)]


def display_rankings(rankings: List[Tuple[str, float]]) -> None:
    """Display the rankings in a formatted way with win, top 5, and top 10 probabilities."""
    print("\nPlayer Rankings:")
    print("-" * 175)  # Increased width to accommodate new column
    print(f"{'Rank':<6}{'Player':<30}{'Score':<8}{'Win Probability':<15}{'Top 5 Probability':<15}{'Top 10 Probability':<15}")
    print("-" * 175)  # Increased width to match

    win_probabilities = calculate_win_probabilities(rankings)
    top_5_probabilities = calculate_finish_probabilities(rankings, positions=5)
    top_10_probabilities = calculate_finish_probabilities(rankings, positions=10)

    for i, ((player, score), (_, win_prob), (_, top_5_prob), (_, top_10_prob)) in enumerate(
        zip(rankings, win_probabilities, top_5_probabilities, top_10_probabilities), 1
    ):
        score_100 = (score + 1) * 50  # Convert [-1,1] to [0,100]
        print(f"{i:<6}{player:<30}{score_100:<8.1f}{win_prob:>6.1f}%{top_5_prob:>14.1f}%{top_10_prob:>14.1f}%")


def generate_predictions_data(rankings: List[Tuple[str, float]]) -> Dict:
    """Generate a structured dictionary of predictions data"""
    
    win_probabilities = calculate_win_probabilities(rankings)
    top_5_probabilities = calculate_finish_probabilities(rankings, positions=5)
    top_10_probabilities = calculate_finish_probabilities(rankings, positions=10)
    
    # Convert scores from [-1,1] to [0,100] scale
    predictions = []
    for i, ((player, score), (_, win_prob), (_, top_5_prob), (_, top_10_prob)) in enumerate(
        zip(rankings, win_probabilities, top_5_probabilities, top_10_probabilities), 1
    ):
        score_100 = (score + 1) * 50  # Convert [-1,1] to [0,100]
        predictions.append({
            'rank': i,
            'player': player,
            'score': round(score_100, 1),
            'probabilities': {
                'win': round(win_prob, 1),
                'top_5': round(top_5_prob, 1),
                'top_10': round(top_10_prob, 1)
            }
        })
    
    predictions_data = {
        'predictions': predictions,
        'metadata': {
            'total_players': len(rankings),
            'generation_date': datetime.now().strftime('%Y-%m-%d'),
            'weights_used': WEIGHTS
        }
    }
    
    return predictions_data


def main():
    rankings = rank_players()
    
    # Generate predictions data
    predictions_data = generate_predictions_data(rankings)
    
    # Save to JSON file
    with open('tournament_predictions.json', 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    print("Predictions saved to tournament_predictions.json")
    
    # Still display rankings in console
    display_rankings(rankings)


if __name__ == "__main__":
    main()
