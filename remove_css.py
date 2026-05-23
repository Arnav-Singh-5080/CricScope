#!/usr/bin/env python3
"""Fix remaining issues in application.py"""

with open('application.py', 'rb') as f:
    content = f.read()

# Decode as latin-1 to preserve bytes, then fix
content_str = content.decode('utf-8', errors='replace')

# Fix the team-card-abbr CSS block
lines = content_str.split('\n')
new_lines = []
skip_until_next_comment = False

for i, line in enumerate(lines):
    if '.team-card-abbr {' in line:
        # Skip until we find the closing brace
        skip_until_next_comment = True
        continue
    
    if skip_until_next_comment and line.strip() == '}':
        skip_until_next_comment = False
        continue
    
    if skip_until_next_comment:
        continue
    
    new_lines.append(line)

content_str = '\n'.join(new_lines)

# Remove wr-card CSS
lines = content_str.split('\n')
new_lines = []
skip_wr_card = False

for line in lines:
    if '.wr-card {' in line:
        skip_wr_card = True
        continue
    
    if skip_wr_card and line.strip() == '}':
        skip_wr_card = False
        # Don't add this line either
        continue
    
    if skip_wr_card:
        continue
    
    new_lines.append(line)

content_str = '\n'.join(new_lines)

# Remove stadium light div elements
content_str = content_str.replace('<div class="stadium-light-left"></div>', '')
content_str = content_str.replace('<div class="stadium-light-right"></div>', '')

with open('application.py', 'w', encoding='utf-8') as f:
    f.write(content_str)

print("Fixed CSS blocks")
