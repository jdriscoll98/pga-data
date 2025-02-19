def parse_odds(odds_str):
    # Convert odds string (e.g. "+2000" or "-110") to decimal multiplier
    odds = int(odds_str)
    if odds > 0:
        return odds/100
    else:
        return 100/abs(odds)

def calculate_ev(bet_amount, odds, probability, num_bets):
    # Calculate profit if won (odds * bet_amount), minus the lost bets on other players
    profit_if_won = (bet_amount * parse_odds(odds)) - (bet_amount * (num_bets - 1))
    
    # Loss amount is just the bet amount (other cases covered in other players' EVs)
    loss_amount = -bet_amount
    
    # Calculate EV using formula
    ev = (probability * profit_if_won) + ((1 - probability) * loss_amount)
    return ev

def calculate_top_5_ev(bet_amount, odds, probability):
    # For top 5, we don't subtract other bets since multiple players can place
    profit_if_won = bet_amount * parse_odds(odds)
    loss_amount = -bet_amount
    
    # Calculate EV using formula
    ev = (probability * profit_if_won) + ((1 - probability) * loss_amount)
    return ev

def calculate_top_10_ev(bet_amount, odds, probability):
    # For top 10, we don't subtract other bets since multiple players can place
    profit_if_won = bet_amount * parse_odds(odds)
    loss_amount = -bet_amount
    
    # Calculate EV using formula
    ev = (probability * profit_if_won) + ((1 - probability) * loss_amount)
    return ev

def main():
    # Process winner bets
    total_winner_ev = 0
    total_top_5_ev = 0
    total_top_10_ev = 0
    bet_amount = 10  # Fixed $10 bet amount
    
    # First count number of bets we'll make
    with open('payouts-winner.txt', 'r') as f:
        num_winner_bets = sum(1 for line in f.readlines() if line.strip() and not line.startswith('Winner'))
    
    print(f"\nCalculating EV for ${bet_amount} WINNER bets on {num_winner_bets} players:")
    print("-" * 80)
    print(f"{'Player':<25} {'Odds':<8} {'Prob':<8} {'Profit if Won':>12} {'EV':>8}")
    print("-" * 80)
    
    with open('payouts-winner.txt', 'r') as f:
        # Skip header
        next(f)
        
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
                
            # Extract player name and combine parts until we hit the odds
            name_parts = []
            i = 0
            while i < len(parts) and not parts[i].startswith('+') and not parts[i].startswith('-'):
                name_parts.append(parts[i])
                i += 1
            
            name = " ".join(name_parts)
            odds = parts[i]  # +XXXX odds
            prob = float(parts[-1].strip('%'))/100  # Convert percentage to decimal
            
            profit_if_won = (bet_amount * parse_odds(odds)) - (bet_amount * (num_winner_bets - 1))
            ev = calculate_ev(bet_amount, odds, prob, num_winner_bets)
            total_winner_ev += ev
            
            print(f"{name:<25} {odds:<8} {prob:>7.1%} ${profit_if_won:>11.2f} ${ev:>7.2f}")
    
    # Process top 5 bets
    print(f"\nCalculating EV for ${bet_amount} TOP 5 bets:")
    print("-" * 80)
    print(f"{'Player':<25} {'Odds':<8} {'Prob':<8} {'Profit if Won':>12} {'EV':>8}")
    print("-" * 80)
    
    with open('payouts-top-5.txt', 'r') as f:
        # Skip header
        next(f)
        
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
                
            # Extract player name and combine parts until we hit the odds
            name_parts = []
            i = 0
            while i < len(parts) and not parts[i].startswith('+') and not parts[i].startswith('-'):
                name_parts.append(parts[i])
                i += 1
            
            name = " ".join(name_parts)
            odds = parts[i]  # +XXXX odds
            prob = float(parts[-1].strip('%'))/100  # Convert percentage to decimal
            
            profit_if_won = bet_amount * parse_odds(odds)
            ev = calculate_top_5_ev(bet_amount, odds, prob)
            total_top_5_ev += ev
            
            print(f"{name:<25} {odds:<8} {prob:>7.1%} ${profit_if_won:>11.2f} ${ev:>7.2f}")
    
    # Process top 10 bets
    print(f"\nCalculating EV for ${bet_amount} TOP 10 bets:")
    print("-" * 80)
    print(f"{'Player':<25} {'Odds':<8} {'Prob':<8} {'Profit if Won':>12} {'EV':>8}")
    print("-" * 80)
    
    with open('payouts-top-10.txt', 'r') as f:
        # Skip header
        next(f)
        
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
                
            # Extract player name and combine parts until we hit the odds
            name_parts = []
            i = 0
            while i < len(parts) and not parts[i].startswith('+') and not parts[i].startswith('-'):
                name_parts.append(parts[i])
                i += 1
            
            name = " ".join(name_parts)
            odds = parts[i]  # +XXXX odds
            prob = float(parts[-1].strip('%'))/100  # Convert percentage to decimal
            
            profit_if_won = bet_amount * parse_odds(odds)
            ev = calculate_top_10_ev(bet_amount, odds, prob)
            total_top_10_ev += ev
            
            print(f"{name:<25} {odds:<8} {prob:>7.1%} ${profit_if_won:>11.2f} ${ev:>7.2f}")
    
    print("\nSummary:")
    print("-" * 80)
    print(f"Total Winner EV: ${total_winner_ev:.2f}")
    print(f"Total Top 5 EV: ${total_top_5_ev:.2f}")
    print(f"Total Top 10 EV: ${total_top_10_ev:.2f}")
    print(f"Combined EV: ${total_winner_ev + total_top_5_ev + total_top_10_ev:.2f}")
    print(f"Total Investment: ${bet_amount * (num_winner_bets * 3):.2f}")  # Times 3 for all three types of bets
    print(f"Expected ROI: {((total_winner_ev + total_top_5_ev + total_top_10_ev)/(bet_amount * num_winner_bets * 3))*100:.1f}%")

if __name__ == "__main__":
    main()
