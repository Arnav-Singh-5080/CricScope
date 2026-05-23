#!/usr/bin/env python3
"""Apply all fixes to application.py"""

with open('application.py', 'r', encoding='utf-8') as f:
    content = f.read()

# FIX 1: Remove mobile responsive CSS block
content = content.replace("""/* ---- MOBILE RESPONSIVE ---- */
@media (max-width: 768px) {
    .hero-wrapper {
        padding: clamp(32px, 6vw, 48px) clamp(16px, 6vw, 32px);
    }
    
    .hero-title {
        font-size: clamp(36px, 12vw, 64px);
    }
    
    .stats-row {
        padding: clamp(16px, 3vw, 24px) clamp(16px, 6vw, 32px);
    }
    
    .stat-pill {
        min-width: calc(50% - 10px);
        flex: 1 1 calc(50% - 10px);
    }
    
    section[data-testid="stSidebar"] {
        width: 260px !important;
    }
}

@media (max-width: 480px) {
    .stat-pill {
        min-width: 100%;
        flex: 1 1 100%;
    }
    
    .metrics-row {
        flex-direction: column;
    }
    
    .metric-chip {
        min-width: 100%;
    }
}

""", "")

# FIX 2: Remove team-card-abbr CSS class
content = content.replace("""/* FIX 2: Team card abbreviation - prevent overflow truncation */
.team-card-abbr {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(14px, 2.5vw, 20px);
    font-weight: 700;
    letter-spacing: 2px;
    margin-top: clamp(10px, 2vw, 16px);
    white-space: nowrap;
    overflow: visible;
}

""", "")

# FIX 3: Replace conf_color with hardcoded color
content = content.replace(
    'conf_color = "#10b981" if conf > 0.75 else "#fbbf24" if conf > 0.55 else "#f87171"',
    '# conf_color removed - using hardcoded color'
)
content = content.replace('{conf_color}', '#d4af37')

# FIX 4: Remove win_stats computation and references
# Find and remove the win_stats line
lines = content.split('\n')
new_lines = []
for line in lines:
    if 'win_stats = compute_win_rates()' in line:
        continue  # Skip this line
    if 'win_stats.get(team_name' in line:
        # Replace with safe values
        new_lines.append(line.replace('s = win_stats.get(team_name, {"wins": 0, "total": 0, "rate": 0})', 's = {"wins": 0, "total": 0, "rate": 0}'))
    else:
        new_lines.append(line)
content = '\n'.join(new_lines)

# FIX 5: Remove XGBoost block from analysis page
# Find the section and remove it
xgb_start = 'loaded_xgb = xgb.XGBClassifier()'
xgb_marker = 'input_dict = {'
if xgb_start in content:
    idx_start = content.find(xgb_start)
    idx_end = content.find('input_dict = {', idx_start)
    if idx_start >= 0 and idx_end >= 0:
        # Find the beginning of the line and end
        line_start = content.rfind('\n', 0, idx_start) + 1
        # Remove everything from line start to 'input_dict = {'
        lines_before = content[:line_start]
        lines_after = content[idx_end:]
        content = lines_before + lines_after

# FIX 6: Replace pipe with sim_pipe in simulator section
# Only in the simulator section (after "if st.session_state.page == \"Simulator\":")
simulator_marker = 'if st.session_state.page == "Simulator":'
if simulator_marker in content:
    idx_simulator = content.find(simulator_marker)
    simulator_section = content[idx_simulator:]
    
    # Replace pipe = load_pipe() with sim_pipe = load_pipe()
    simulator_section = simulator_section.replace('pipe = load_pipe()', 'sim_pipe = load_pipe()')
    # Replace pipe.predict_proba with sim_pipe.predict_proba
    simulator_section = simulator_section.replace('pipe.predict_proba', 'sim_pipe.predict_proba')
    # Replace safe_predict(pipe with safe_predict(sim_pipe
    simulator_section = simulator_section.replace('safe_predict(pipe,', 'safe_predict(sim_pipe,')
    
    content = content[:idx_simulator] + simulator_section

# FIX 7: Fix wickets_left in Analysis page
# Replace 'wickets_left = ...' with direct calculation
content = content.replace('wickets_left = 10 - wickets', '')

# FIX 8: Add city selectbox to Analysis page (before creating input_df)
# Find where to insert it
analysis_marker = 'if st.session_state.page == "Analysis":'
if analysis_marker in content:
    idx_analysis = content.find(analysis_marker)
    # Find the input_df = pd.DataFrame section
    idx_input_df = content.find('input_df = pd.DataFrame({', idx_analysis)
    if idx_input_df >= 0:
        # Find the line before input_df
        line_start = content.rfind('\n', 0, idx_input_df)
        city_selectbox = """
    # City selection
    ANALYSIS_CITIES = ['Abu Dhabi', 'Ahmedabad', 'Bangalore', 'Bengaluru', 'Chennai', 'Delhi', 'Hyderabad', 'Jaipur', 'Kolkata', 'Mumbai', 'Pune', 'Chandigarh', 'Dharamsala', 'Indore', 'Nagpur', 'Ranchi', 'Visakhapatnam']
    city_analysis = st.selectbox("Venue City", ANALYSIS_CITIES, key="analysis_city")
"""
        # Insert before input_df
        content = content[:line_start] + city_selectbox + content[line_start:]
        
        # Now replace 'city': ['Mumbai'] with city_analysis
        content = content.replace("'city': ['Mumbai']", "'city': [city_analysis]")

# FIX 9: Fix broken unicode characters
unicode_replacements = {
    'Â·': '·',
    'â€"': '-',
    'Â»': '»',
    'Ã©': 'é',
    'â€™': "'",
    'â€œ': '"',
    'â€': '"',
}

for old, new in unicode_replacements.items():
    content = content.replace(old, new)

# Remove partial characters
content = content.replace('Å', '')
content = content.replace('ðŸ', '')
content = content.replace('š', '')
content = content.replace('±', '')

# Remove stadium light divs
content = content.replace('<div class="stadium-light-left"></div>', '')
content = content.replace('<div class="stadium-light-right"></div>', '')

with open('application.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ All fixes applied")
