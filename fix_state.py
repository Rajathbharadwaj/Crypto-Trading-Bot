import json
import datetime

# Load current state
try:
    with open('bot_state.json', 'r') as f:
        state = json.load(f)
    
    # Get current time
    now = datetime.datetime.now()
    
    # Create a timestamp 15 minutes ago with CURRENT YEAR (not 2025)
    current_year = now.year
    fifteen_min_ago = now - datetime.timedelta(minutes=15)
    
    # Update state
    state['crossover_price_index'] = 4990
    state['crossover_time'] = fifteen_min_ago.isoformat()
    
    print(f"Current time: {now.isoformat()}")
    print(f"Setting crossover time to: {fifteen_min_ago.isoformat()} (15 minutes ago)")
    
    # Save the fixed state
    with open('bot_state.json', 'w') as f:
        json.dump(state, f)
    
    print("State file successfully updated:")
    print(json.dumps(state, indent=2))
    
except Exception as e:
    print(f"Error updating state file: {str(e)}") 