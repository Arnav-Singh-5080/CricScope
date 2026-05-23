#!/usr/bin/env python3
"""Fix Analysis page to use pipe.pkl instead of XGBoost"""

with open('application.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and replace the XGBoost section
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # When we find UI_TO_MODEL_NAME, skip until we find the simple input_df creation
    if 'UI_TO_MODEL_NAME = {' in line:
        # Skip all lines until we find 'input_df = pd.DataFrame({'
        # that signals the end of the old XGBoost block
        while i < len(lines) and 'input_df = pd.DataFrame([input_dict])' not in lines[i]:
            i += 1
        # Skip the input_df line too
        if i < len(lines):
            i += 1
        continue
    
    # Fix broken unicode in this section
    if 'ðŸ' in line or '€"' in line:
        line = line.replace('ðŸ', '')
        line = line.replace('€"', '')
    
    new_lines.append(line)
    i += 1

with open('application.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Fixed Analysis XGBoost code")
