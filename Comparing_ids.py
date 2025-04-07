import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict

def parse_data(file_path):
    """Parse the text data into a list of frames with their IDs"""
    data = []
    pattern = r"Frame \d+ player tracker IDs: \[([^\]]+)\]"
    
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                ids = []
                for x in match.group(1).split(','):
                    x = x.strip()
                    if x == 'None':
                        ids.append(None)
                    elif x.isdigit():
                        ids.append(int(x))
                    else:
                        ids.append(None)
                data.append(ids)
    return data

def calculate_id_changes(data):
    """Calculate how often each player ID (position) changes value across frames"""
    if not data:
        return {}
    
    position_changes = defaultdict(int)
    num_positions = len(data[0])
    
    for pos in range(num_positions):
        prev_id = None
        for frame in data:
            current_id = frame[pos]
            if current_id != prev_id and prev_id is not None:
                position_changes[pos] += 1  # pos represents the player ID
            prev_id = current_id
    
    return position_changes

def plot_id_changes(before_changes, after_changes):
    """Create comparative plot of ID changes"""
    all_ids = sorted(set(before_changes.keys()).union(set(after_changes.keys())))
    
    before_counts = [before_changes.get(id, 0) for id in all_ids]
    after_counts = [after_changes.get(id, 0) for id in all_ids]
    
    plt.figure(figsize=(14, 7))
    bar_width = 0.35
    x = np.arange(len(all_ids))
    
    plt.bar(x - bar_width/2, before_counts, bar_width, 
            label='Before Stabilization', color='royalblue', alpha=0.7)
    plt.bar(x + bar_width/2, after_counts, bar_width, 
            label='After Stabilization', color='limegreen', alpha=0.7)
    
    plt.xlabel('Player ID (Position in List)')
    plt.ylabel('Number of Value Changes')
    plt.title('Player ID Stability: Frequency of Value Changes')
    plt.xticks(x, all_ids)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Calculate and display averages
    avg_before = np.mean(list(before_changes.values())) if before_changes else 0
    avg_after = np.mean(list(after_changes.values())) if after_changes else 0
    
    plt.axhline(avg_before, color='blue', linestyle='--', alpha=0.5)
    plt.axhline(avg_after, color='green', linestyle='--', alpha=0.5)
    
    # Calculate improvement percentage
    improvement_pct = 0
    if avg_before > 0:
        improvement_pct = ((avg_before - avg_after) / avg_before) * 100
    
    plt.text(0.02, 0.95, f'Avg Before: {avg_before:.2f} changes/ID', 
             transform=plt.gca().transAxes, color='blue')
    plt.text(0.02, 0.90, f'Avg After: {avg_after:.2f} changes/ID', 
             transform=plt.gca().transAxes, color='green')
    plt.text(0.02, 0.85, f'Improvement: {improvement_pct:.1f}% stabilization', 
             transform=plt.gca().transAxes, color='purple', weight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return avg_before, avg_after

# Parse both files
before_data = parse_data('ids_before.txt')
after_data = parse_data('ids_after.txt')

# Calculate changes per player ID (position)
before_changes = calculate_id_changes(before_data)
after_changes = calculate_id_changes(after_data)

# Plot and get statistics
avg_before, avg_after = plot_id_changes(before_changes, after_changes)

# Print comprehensive report
print("\n=== Player ID Change Analysis ===")
print(f"Number of player IDs tracked: {max(len(before_changes), len(after_changes))}")
print(f"\nAverage changes per player ID:")
print(f"Before stabilization: {avg_before:.2f}")
print(f"After stabilization: {avg_after:.2f}")

if avg_before > 0:
    improvement = (avg_before - avg_after) / avg_before * 100
    print(f"\nImprovement: {improvement:.1f}% reduction in ID changes")

# Find most and least improved IDs
common_ids = set(before_changes.keys()).intersection(set(after_changes.keys()))
improvements = {id: before_changes[id] - after_changes.get(id, 0) for id in common_ids}

if improvements:
    most_improved = max(improvements.items(), key=lambda x: x[1])
    least_improved = min(improvements.items(), key=lambda x: x[1])
    
    print(f"\nMost improved player ID: {most_improved[0]}")
    print(f"Changes before: {before_changes[most_improved[0]]}, after: {after_changes[most_improved[0]]}")
    print(f"\nLeast improved player ID: {least_improved[0]}")
    print(f"Changes before: {before_changes[least_improved[0]]}, after: {after_changes[least_improved[0]]}")