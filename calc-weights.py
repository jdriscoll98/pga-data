import json
import pandas as pd
import numpy as np
from typing import Dict, List
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_tournament_data(filename: str) -> List[Dict]:
    """Load tournament data from JSON file and flatten it into a list of player stats"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    def convert_finish_position(pos: str) -> int | None:
        """Convert finish position to integer, handling special cases"""
        # Skip special cases
        if pos in ['WD', 'DQ', 'CUT']:
            return None
        
        # Remove 'T' from tied positions
        pos = pos.replace('T', '')
        return int(pos)
    
    # Flatten the data structure
    players = []
    for year, year_data in data.items():
        for player in year_data['leaderboard']:
            finish_pos = convert_finish_position(player['finish'])
            # Only include players who completed the tournament
            if finish_pos is not None:
                player_data = {
                    'year': year,
                    'name': player['name'],
                    'finish': finish_pos,
                    **player['stats']  # Unpack all stats
                }
                players.append(player_data)
    
    return players

def clean_stat_value(value):
    """Clean stat values by converting percentage strings and other formats to float"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Handle percentage with fractions like "69.23% (36/52)"
        if '(' in value:
            value = value.split('(')[0].strip()
        # Handle distance measurements
        if 'yds' in value:
            value = value.replace('yds', '').strip()
        # Convert percentage to decimal
        if '%' in value:
            value = value.replace('%', '')
        try:
            return float(value)
        except ValueError:
            return None
    return None

def analyze_stats_correlation(players: List[Dict]) -> pd.DataFrame:
    """Calculate correlation between each stat and finishing position"""
    # Convert to DataFrame
    df = pd.DataFrame(players)
    # Clean numeric columns
    numeric_cols = []
    for col in df.columns:
        if col not in ['year', 'name', 'STAT']:
            df[col] = df[col].apply(clean_stat_value)
            if df[col].dtype in ['float64', 'int64']:
                numeric_cols.append(col)
    
    # Calculate correlation with finish position
    correlations = []
    for col in numeric_cols:
        if col != 'finish':
            correlation = df[col].corr(df['finish'])
            correlations.append({
                'stat': col,
                'correlation': correlation,
                'abs_correlation': abs(correlation)
            })
    
    # Convert to DataFrame and sort by absolute correlation
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)
    
    return corr_df

def analyze_stat_distributions(df: pd.DataFrame) -> None:
    """Analyze the distribution of each stat for top performers vs others"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Define top performers (e.g., top 10)
    df['is_top_10'] = df['finish'] <= 10
    
    for col in numeric_cols:
        if col != 'finish':
            # Calculate basic stats for top 10 vs others
            top_stats = df[df['is_top_10']][col].describe()
            others_stats = df[~df['is_top_10']][col].describe()
            
            print(f"\n{col} Statistics:")
            print("Top 10 players:")
            print(top_stats)
            print("\nOther players:")
            print(others_stats)

def analyze_percentiles(df: pd.DataFrame) -> None:
    """Show which percentile top finishers were in for each stat"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Look at top 5 finishers
    top_players = df[df['finish'] <= 5]
    
    for _, player in top_players.iterrows():
        print(f"\nPlayer: {player['name']} (Finish: {player['finish']})")
        for col in numeric_cols:
            if col not in ['finish', 'year']:
                percentile = stats.percentileofscore(df[col].dropna(), player[col])
                print(f"{col}: {percentile:.1f}th percentile")

def cluster_players(df: pd.DataFrame) -> None:
    """Group players into clusters based on their stats"""
    # Select numeric columns for clustering
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['finish', 'year']]
    
    # Create a copy of the dataframe with just the columns we need
    cluster_df = df[numeric_cols].copy()
    
    # Instead of dropping all rows with any missing values,
    # let's be more selective:
    
    # 1. First, check how many non-null values we have for each column
    non_null_counts = cluster_df.count()
    print("\nNumber of non-null values per column:")
    print(non_null_counts)
    
    # 2. Keep only columns that have at least 70% of values
    min_non_null = len(cluster_df) * 0.7
    columns_to_keep = non_null_counts[non_null_counts >= min_non_null].index
    cluster_df = cluster_df[columns_to_keep]
    
    # 3. Now fill remaining missing values with the mean
    cluster_df = cluster_df.fillna(cluster_df.mean())
    
    # 4. Determine number of clusters based on data size
    n_clusters = min(4, len(cluster_df) // 3)  # Ensure we have enough samples per cluster
    if n_clusters < 2:
        print("\nNot enough data for meaningful clustering")
        return
    
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(cluster_df)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # Add cluster labels back to original dataframe
    cluster_df['cluster'] = cluster_labels
    cluster_df['name'] = df.loc[cluster_df.index, 'name']
    cluster_df['finish'] = df.loc[cluster_df.index, 'finish']
    
    # Analyze clusters
    print(f"\nFound {n_clusters} distinct player groups:")
    
    for cluster in range(n_clusters):
        cluster_players = cluster_df[cluster_df['cluster'] == cluster]
        print(f"\nCluster {cluster}:")
        print(f"Average finish position: {cluster_players['finish'].mean():.1f}")
        print(f"Number of players: {len(cluster_players)}")
        print("\nKey characteristics:")
        # Only show the most distinctive stats (top 5 different from overall mean)
        stat_differences = {}
        for col in columns_to_keep:
            cluster_mean = cluster_players[col].mean()
            overall_mean = cluster_df[col].mean()
            stat_differences[col] = abs(cluster_mean - overall_mean)
        
        top_stats = sorted(stat_differences.items(), key=lambda x: x[1], reverse=True)[:5]
        for stat, diff in top_stats:
            cluster_mean = cluster_players[stat].mean()
            print(f"{stat}: {cluster_mean:.2f}")
        
        print("\nExample players in this cluster:")
        print(cluster_players.sort_values('finish')[['name', 'finish']].head().to_string())

def analyze_yearly_trends(df: pd.DataFrame) -> None:
    """Analyze how stats have changed over the years"""
    yearly_stats = df.groupby('year').agg({
        'finish': 'count',  # number of players
        **{col: 'mean' for col in df.select_dtypes(include=[np.number]).columns}
    }).round(2)
    
    print("\nYearly Trends:")
    print(yearly_stats)

def analyze_stat_combinations(df: pd.DataFrame) -> None:
    """Analyze how combinations of stats relate to performance"""
    # First, let's see what columns we actually have
    print("\nAvailable columns:")
    print(df.columns.tolist())
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['finish', 'year']]
    
    # Create composite stats only if the required columns exist
    composite_stats = []
    
    # Try accuracy combination
    if 'Driving Accuracy' in df.columns and 'Greens in Regulation' in df.columns:
        df['total_accuracy'] = df['Driving Accuracy'] + df['Greens in Regulation']
        composite_stats.append('total_accuracy')
    
    # Try short game combination
    if 'Scrambling' in df.columns and 'Putting Average' in df.columns:
        df['short_game_index'] = df['Scrambling'] + df['Putting Average']
        composite_stats.append('short_game_index')
    
    # Try other potentially useful combinations
    if 'Driving Distance' in df.columns and 'Driving Accuracy' in df.columns:
        df['driving_effectiveness'] = df['Driving Distance'] * df['Driving Accuracy'] / 100
        composite_stats.append('driving_effectiveness')
    
    # Analyze correlations of composite stats
    print("\nComposite Stat Correlations with Finish:")
    for stat in composite_stats:
        correlation = df[stat].corr(df['finish'])
        print(f"{stat}: {correlation:.3f}")
    
    if not composite_stats:
        print("No composite stats could be calculated with available columns")

def check_data_quality(df: pd.DataFrame) -> None:
    """Print data quality information"""
    print("\n=== DATA QUALITY REPORT ===")
    
    # Check missing values
    missing = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing[missing > 0])
    
    # Check number of complete rows
    complete_rows = df.dropna().shape[0]
    total_rows = df.shape[0]
    print(f"\nComplete rows: {complete_rows} out of {total_rows} ({complete_rows/total_rows*100:.1f}%)")

def generate_weights_data(df: pd.DataFrame, correlations: pd.DataFrame) -> Dict:
    """Generate a structured dictionary of analysis results"""
    
    # Get top 10 most correlated stats
    top_correlations = correlations.head(10).to_dict('records')
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate average stats for top 10 performers (numeric columns only)
    top_10_stats = df[df['finish'] <= 10][numeric_cols].mean().to_dict()
    # Remove non-relevant fields
    top_10_stats.pop('finish', None)
    top_10_stats.pop('year', None)
    
    # Calculate overall averages (numeric columns only)
    overall_stats = df[numeric_cols].mean().to_dict()
    overall_stats.pop('finish', None)
    overall_stats.pop('year', None)
    
    # Replace NaN with None (which becomes null in JSON)
    top_10_stats = {k: None if pd.isna(v) else v for k, v in top_10_stats.items()}
    overall_stats = {k: None if pd.isna(v) else v for k, v in overall_stats.items()}
    
    # Structure the data
    weights_data = {
        'correlations': [{k: None if pd.isna(v) else v for k, v in stat.items()} for stat in top_correlations],
        'top_10_averages': {k: round(v, 3) if v is not None else None for k, v in top_10_stats.items()},
        'overall_averages': {k: round(v, 3) if v is not None else None for k, v in overall_stats.items()},
        'metadata': {
            'total_players': len(df),
            'years_analyzed': sorted(df['year'].unique().tolist()),
            'generation_date': pd.Timestamp.now().strftime('%Y-%m-%d')
        }
    }
    
    return weights_data

def main():
    # Load and analyze data
    players = load_tournament_data('tournament_stats.json')
    df = pd.DataFrame(players)
    
    # Clean numeric columns
    for col in df.columns:
        if col not in ['year', 'name']:
            df[col] = df[col].apply(clean_stat_value)
    
    # Get correlations
    correlations = analyze_stats_correlation(players)
    
    # Generate weights data
    weights_data = generate_weights_data(df, correlations)
    
    # Save to JSON file
    with open('tournament_weights.json', 'w') as f:
        json.dump(weights_data, f, indent=2)
    
    print("Analysis complete. Results saved to tournament_weights.json")
    
    # Original analysis code can remain if you still want console output
    check_data_quality(df)
    print("\n=== CORRELATION ANALYSIS ===")
    print(correlations[['stat', 'correlation']].to_string(index=False))
    
    print("\n=== DISTRIBUTION ANALYSIS ===")
    analyze_stat_distributions(df)
    
    print("\n=== TOP PERFORMER PERCENTILES ===")
    analyze_percentiles(df)
    
    print("\n=== PLAYER CLUSTERS ===")
    cluster_players(df)
    
    print("\n=== YEARLY TRENDS ===")
    analyze_yearly_trends(df)
    
    print("\n=== STAT COMBINATIONS ===")
    analyze_stat_combinations(df)

if __name__ == "__main__":
    main()
