import re

with open('app.py', 'r') as f:
    content = f.read()

# Pattern to match st.markdown call with unsafe_allow_html=True
# We match st.markdown(, unsafe_allow_html=True) and capture the first argument (non-greedy)
pattern = r'st\.markdown\((.*?), unsafe_allow_html=True\)'
# Replace with st.html(\1)
new_content = re.sub(pattern, r'st.html(\1)', content, flags=re.DOTALL)

with open('app.py', 'w') as f:
    f.write(new_content)