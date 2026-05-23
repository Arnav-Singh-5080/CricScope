#!/usr/bin/env python3
"""Fix the remaining issues in application.py"""

with open('application.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and remove the problematic section
new_lines = []
skip_until = None
i = 0
while i < len(lines):
    line = lines[i]
    
    # Skip the XGBoost remnants
    if "'wickets_left':  wickets_left," in line:
        # Skip until we find the line with 'input_df = pd.DataFrame'
        skip_until = 'input_df = pd.DataFrame'
        i += 1
        continue
    
    if skip_until and skip_until in line:
        skip_until = None
    
    if skip_until:
        i += 1
        continue
    
    # Skip ALL_MODEL_TEAMS references
    if 'for team in ALL_MODEL_TEAMS:' in line:
        # Skip this line and the next one (the content line)
        i += 2
        continue
    
    # Skip wickets_left references in Analysis section  
    if "'wickets_left': 10 - wickets" in line:
        new_lines.append(line.replace("'wickets_left': 10 - wickets,", ""))
        i += 1
        continue
    
    new_lines.append(line)
    i += 1

with open('application.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✓ Fixed remaining issues")
