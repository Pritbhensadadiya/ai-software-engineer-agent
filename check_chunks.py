from code_chunker import chunk_repository

chunks = chunk_repository()

print("Total chunks:", len(chunks))
print("\nFirst 10 chunks:")

for chunk in chunks[:10]:
    file_path = chunk["file_path"]
    start_line = chunk["start_line"]
    end_line = chunk["end_line"]
    language = chunk["language"]

    print(f"{file_path}:{start_line}-{end_line} [{language}]")
