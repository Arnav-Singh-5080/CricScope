#!/usr/bin/env python3
"""Comprehensive cleanup of application.py"""

with open('application.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix unicode issues by reading character by character and replacing
import unicodedata

# Replace common broken unicode patterns
replacements = [
    ('\u2013', '-'),  # en-dash
    ('\u201c', '"'),  # left quote
    ('\u201d', '"'),  # right quote  
    ('\u2019', "'"),  # right single quote
    ('\u00b7', '·'),  # middle dot
    ('\u00bb', '»'),  # right guillemet
    ('\u00e9', 'é'),  # e acute
]

for old, new in replacements:
    content = content.replace(old, new)

# Remove any byte sequences that aren't valid UTF-8 text
content = ''.join(c for c in content if ord(c) < 65536 or c in '\n\t ')

# Remove duplicate city selectbox block
import re
pattern = r'# City selection\s+ANALYSIS_CITIES = \[.*?\]\s+city_analysis = st\.selectbox\("Venue City", ANALYSIS_CITIES, key="analysis_city"\)'
matches = list(re.finditer(pattern, content, re.DOTALL))
if len(matches) > 1:
    for match in matches[1:]:
        content = content[:match.start()] + content[match.end():]

# Replace old XGBoost input_dict style with new DataFrame style
old_pattern = r'''input_dict = \{[\s\S]*?input_dict\[bowl_col\] = 1\s*input_df = pd\.DataFrame\(\[input_dict\]\)'''
new_df_code = '''input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city_analysis],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [10 - wickets],
            'target': [target],
            'crr': [crr],
            'rrr': [rrr]
        })'''

content = re.sub(old_pattern, new_df_code, content, flags=re.DOTALL)

# Replace XGBoost predict with pipe predict
content = content.replace('loaded_xgb.predict_proba', 'pipe.predict_proba')

# Remove broken unicode in markdown strings
content = re.sub(r'[^\x00-\x7F]+', '', content)

with open('application.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Cleanup complete")
