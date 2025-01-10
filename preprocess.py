import collections

with open("dataset/Tree_Comrades_1936_978-5-17-108545-2.txt", "r") as f:
    lines = f.readlines()

# Remove empty lines
lines = [line.strip() for line in lines if line.strip()]
text = "\n".join(lines)

# with open("dataset/Tree_Comrades_edition_2020_Archipov.txt", "w") as f:
#     f.write(text)

# c = collections.Counter(text)

# Split into train, dev, test (80/10/10) based on number of characters
total_chars = len(text)
train_size = int(0.8 * total_chars)
dev_size = int(0.1 * total_chars)
test_size = total_chars - train_size - dev_size

train_text = text[:train_size]
dev_text = text[train_size : train_size + dev_size]
test_text = text[train_size + dev_size :]

pass
