# import os
# import pandas as pd
#
# static_path = "/Users/dibyanshuchatterjee/PycharmProjects/LLM-Pretraining/conversation-corpora/Song-Lyrics/csv/"
#
#
# def process_csv_files(num_files):
#     csv_files = [file for file in os.listdir(static_path) if file.endswith('.csv')]
#     # Limit the number of files to the specified 'num_files'
#     csv_files = csv_files[:num_files]
#     print(csv_files)
#
#     # Create a single output text file for all lyrics
#     output_file_name = "combined_lyrics.txt"
#     with open(output_file_name, "w", encoding="utf-8") as txt_file:
#         for i, file_name in enumerate(csv_files, start=1):
#             try:
#                 df = pd.read_csv(static_path + file_name)
#
#                 for index, row in df.iterrows():
#                     artist = row["Artist"]
#                     title = row["Title"]
#                     lyric = row["Lyric"]
#
#                     # Write artist and track name
#                     txt_file.write(f"Artist: {artist}, Track name: {title}\n")
#
#                     # Write lyrics
#                     txt_file.write(f"{lyric}\n\n")
#
#                 print(f"Lyrics from {file_name} added to {output_file_name}.")
#
#             except Exception as e:
#                 print(f"An error occurred while processing {file_name}: {str(e)}")
#
#     print(f"All lyrics combined and saved in {output_file_name}.")
#
#
# # Input: Ask the user how many files to process
# csv_files_for_file_cnt = [file for file in os.listdir(static_path) if file.endswith('.csv')]
# total_available = len(csv_files_for_file_cnt)
# num_files_to_read = int(input(f"How many CSV files out of {total_available} available files do you want to read? "))
# process_csv_files(num_files_to_read)


from datasets import load_dataset

dataset = load_dataset("wikisql")
train_data = dataset["train"]

# print(type(dataset))
# print(type(train_data['sql'][0]['human_readable']))
# print(len(train_data['sql']))


def process_csv_files():
    # Create a single output text file for all lyrics
    output_file_name = "combined_sql.txt"
    with open(output_file_name, "w", encoding="utf-8") as txt_file:
        for idx in range(0, len(train_data['sql'])):
            try:
                txt_file.write(f"{train_data['sql'][idx]['human_readable']}\n")
            except Exception as e:
                print(f"An error occurred while preparing the corpus: {str(e)}")

    print(f"All sql text combined and saved in {output_file_name}.")


process_csv_files()
