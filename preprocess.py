import collections
import os
import re

dataset_paths = [
    "dataset/All_Quiet_on_the_Western_Front_1929_AST_978-5-17-105639-1.txt",
    "dataset/All_Quiet_on_the_Western_Front_1929_AST_978-5-17-137374-0.txt",
    "dataset/Station_at_the_Horizon_1928_AST_978-5-17-133322-5.txt",
    "dataset/The_Dream_Room_1920_AST_978-5-17-071518-3.txt",
    "dataset/The_Road_Back_1931.txt",
    "dataset/Tree_Comrades_1936_978-5-17-004252-4.txt",
    "dataset/Tree_Comrades_1936_978-5-17-056963-2.txt",
    "dataset/Tree_Comrades_1936_978-5-17-108545-2.txt",
    "dataset/A_Time_to_Love_and_a_Time_to_Die.txt",
    "dataset/Arch_of_Triumph_1945.txt",
    "dataset/Flotsam_1939.txt",
    "dataset/Heaven_Has_No_Favorites.txt",
    "dataset/Shadows_in_Paradise.txt",
    "dataset/Spark_of_Life_1952.txt",
    "dataset/The_Black_Obelisk.txt",
    "dataset/The_Promised_Land_Vagrius_978-5-9697-0386-5.txt",
    "dataset/The_Night_in_Lisbon.txt",
    "dataset/Gam.txt",
]

alphabet = [
    "\n",
    " ",
    "!",
    ",",
    "-",
    ".",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    "?",
    "а",
    "б",
    "в",
    "г",
    "д",
    "е",
    "ж",
    "з",
    "и",
    "й",
    "к",
    "л",
    "м",
    "н",
    "о",
    "п",
    "р",
    "с",
    "т",
    "у",
    "ф",
    "х",
    "ц",
    "ч",
    "ш",
    "щ",
    "ъ",
    "ы",
    "ь",
    "э",
    "ю",
    "я",
    "ё",  # replace with "е"
    "–",  # replace with "—"
    "—",
    "…",  # replace with "..."
]

cyrillic_pattern = re.compile(r"[а-яА-Я]")


def process_text_file(file_path, output_path=None):
    """
    Process a text file by removing characters not in the specified alphabet
    and replacing certain characters with their equivalents.

    Args:
        file_path (str): Path to the input text file
        output_path (str, optional): Path for the output file. If None, creates a new filename

    Returns:
        str: Path to the processed output file
    """

    # Create a set of lowercase alphabet for faster lookups
    alphabet_lower_set = {char.lower() for char in alphabet}

    # Define character replacements
    replacements = {
        "ё": "е",
        "–": "—",
        "…": "...",
    }

    # If no output path is provided, create one based on the input path
    base_filename = os.path.basename(file_path)
    output_filename = os.path.splitext(base_filename)[0] + ".txt"

    os.makedirs("processed_dataset", exist_ok=True)

    # Construct the full output path
    output_path = os.path.join("processed_dataset", output_filename)

    # Process the file
    with open(file_path, "r", encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", line)
            processed_line = ""
            for char in line:
                char_lower = char.lower()

                # Apply replacements if the character is in the replacement dictionary
                if char in replacements:
                    processed_line += replacements[char]
                # Keep the character if its lowercase version is in the alphabet
                elif char_lower in alphabet_lower_set:
                    processed_line += char
                # Characters not in the alphabet are skipped

            # Only write the line if it contains at least one Cyrillic letter
            if cyrillic_pattern.search(processed_line):
                f_out.write(processed_line)

    return output_path


all_lines = []
for path in dataset_paths:
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    all_lines.extend(lines)

text = "\n".join(all_lines)
c = collections.Counter(text)

processed_files = []

for path in dataset_paths:
    processed_path = process_text_file(path)
    processed_files.append(processed_path)

for path in processed_files:
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    total_lines = len(lines)
    train_size = int(total_lines * 0.8)
    dev_size = int(total_lines * 0.1)

    train_lines = lines[:train_size]
    dev_lines = lines[train_size : train_size + dev_size]
    test_lines = lines[train_size + dev_size :]

    # Write to separate files
    base_path = path.replace(".txt", "")
    with open(f"{base_path}-train.txt", "w") as f:
        f.write("\n".join(train_lines))
    with open(f"{base_path}-dev.txt", "w") as f:
        f.write("\n".join(dev_lines))
    with open(f"{base_path}-test.txt", "w") as f:
        f.write("\n".join(test_lines))
    with open(f"{base_path}.txt", "w") as f:
        f.write("\n".join(lines))
