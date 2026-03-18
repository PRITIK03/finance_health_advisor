import os

# Get the correct path
file_path = os.path.join('d:/project/Trae', 'finance_health_advisor', 'app.py')
print(f'File path: {file_path}')
print(f'File exists: {os.path.exists(file_path)}')

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Count occurrences before
count_before = content.count('st.container(border=True)')
print(f'Found {count_before} occurrences before replacement')

# Replace all occurrences
new_content = content.replace('st.container(border=True)', 'st.container()')

# Count occurrences after
count_after = new_content.count('st.container(border=True)')
print(f'Found {count_after} occurrences after replacement')

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print('File written successfully')
