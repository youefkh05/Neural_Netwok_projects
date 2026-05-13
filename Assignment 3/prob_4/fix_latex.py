import re

with open('prob4.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace specifically the arabic snippet that broke the compiler, and any stray arabic characters
text = re.sub(r'[\u0600-\u06FF]', '', text)

with open('prob4.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print("Fixed LaTeX Arabic character issue.")
