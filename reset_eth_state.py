import json
import datetime
import argparse

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Reset ETH trading bot state')
    parser.add_argument('--force', action='store_true', 
                        help='Force reset even if waiting for next candle')
    parser.add_argument('--simulate-crossover', action='store', choices=['BUY', 'SELL'],
                        help='Simulate a crossover of the specified type')
    args = parser.parse_args()

    if args.simulate_crossover:
        # Create a simulated crossover state
        import pandas as pd
        now = datetime.datetime.now()
        
        reset_state = {
            "crossover_detected": True,
            "crossover_type": args.simulate_crossover,
            "crossover_price": 0.0,  # Will be updated on next run
            "crossover_price_index": 4990,
            "crossover_time": now.isoformat(),
            "reset_timestamp": now.isoformat(),
            "simulated": True
        }
        action_msg = f"Simulated {args.simulate_crossover} crossover"
    else:
        # Create a clean reset state
        reset_state = {
            "crossover_detected": False,
            "crossover_type": None,
            "crossover_price": None,
            "crossover_price_index": -1,
            "reset_timestamp": datetime.datetime.now().isoformat()
        }
        action_msg = "Reset"

    # Save the reset state
    try:
        with open('eth_state.json', 'w') as f:
            json.dump(reset_state, f)
        print(f"{action_msg} completed successfully:")
        print(json.dumps(reset_state, indent=2))
        print(f"\nCompleted at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not args.simulate_crossover:
            print("The ETH bot will now start fresh with no active crossover signals.")
        else:
            print(f"The ETH bot will now simulate a {args.simulate_crossover} crossover on next run.")
    except Exception as e:
        print(f"Error updating state file: {str(e)}")

if __name__ == "__main__":
    main() 