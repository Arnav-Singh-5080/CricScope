#!/usr/bin/env python3
"""Comprehensive cleanup of application.py"""

with open('application.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Step 1: Fix all broken unicode characters
broken_chars = {
    'â€"': '-',
    'â€œ': '"',
    'â€': '"',
    'â€™': "'",
    'â†'': '',
    'Â·': '·',
    'Â»': '»',
    'Ã©': 'é',
    '€"': '',
    'ï¸': '',
    'ðŸ': '',
    'š': '',
    '±': '',
    'Å': '',
}

for bad, good in broken_chars.items():
    content = content.replace(bad, good)

# Step 2: Remove duplicate city selectbox
lines = content.split('\n')
seen_city_selectbox = False
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if 'city_analysis = st.selectbox("Venue City"' in line:
        if not seen_city_selectbox:
            seen_city_selectbox = True
            new_lines.append(line)
            new_lines.append(lines[i+1])  # Add ANALYSIS_CITIES line
            i += 2
            continue
        else:
            # Skip duplicate
            i += 1
            # Skip ANALYSIS_CITIES line too if it follows
            if i < len(lines) and 'ANALYSIS_CITIES' in lines[i]:
                i += 1
            continue
    new_lines.append(line)
    i += 1

content = '\n'.join(new_lines)

# Step 3: Fix the Analysis page input_df creation
# Remove the old XGBoost-style input_dict creation
old_input_dict = """input_dict = {
            'target_score':  target,
            'runs_left':     runs_left,
            'balls_left':    balls_left,
            'crr':           crr,
            'rrr':           rrr,
            'wickets_left':  wickets_left,
        }
        for team in ALL_MODEL_TEAMS:
            input_dict[f"bat_{team}"] = 0
        for team in ALL_MODEL_TEAMS:
            input_dict[f"bowl_{team}"] = 0

        # Set the correct bat/bowl flags using the historical name mapping.
        bat_col  = f"bat_{UI_TO_MODEL_NAME[batting_team]}"
        bowl_col = f"bowl_{UI_TO_MODEL_NAME[bowling_team]}"
        input_dict[bat_col]  = 1
        input_dict[bowl_col] = 1

        input_df = pd.DataFrame([input_dict])"""

new_input_df = """input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city_analysis],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [10 - wickets],
            'target': [target],
            'crr': [crr],
            'rrr': [rrr]
        })"""

if old_input_dict in content:
    content = content.replace(old_input_dict, new_input_df)

# Step 4: Remove XGBoost prediction call and replace with pipe prediction
content = content.replace('result = loaded_xgb.predict_proba(input_df)', 'result = pipe.predict_proba(input_df)')

# Step 5: Remove any remaining references to wickets_left = ...
content = content.replace("'wickets_left': wickets_left,", "")
content = content.replace("'wickets_left': 10 - wickets,", "")

# Step 6: Fix the guards/guards comments that reference €"
content = content.replace('Guard 1: batting team has already reached or crossed the target', 'Guard 1: batting team has already reached or crossed the target')

# Step 7: Remove duplicate opening divs for main-pad
# Find and remove duplicates
lines = content.split('\n')
new_lines = []
last_was_main_pad = False
for line in lines:
    if 'st.markdown(\'<div class="main-pad">\'' in line or 'st.markdown(\'<div class="main-pad-analysis">\'' in line:
        if not last_was_main_pad:
            new_lines.append(line)
            last_was_main_pad = True
        continue
    new_lines.append(line)
    if 'class="main-pad' in line:
        last_was_main_pad = True
    else:
        last_was_main_pad = False

content = '\n'.join(new_lines)

with open('application.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Comprehensive cleanup complete")
