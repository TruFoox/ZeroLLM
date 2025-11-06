import ftfy

with open("training_data.txt", "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

clean_text = ftfy.fix_text(text)

with open("training_data_clean.txt", "w", encoding="utf-8") as f:
    f.write(clean_text)
