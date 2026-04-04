import csv
from langchain_core.documents import Document

def load_csv(data_path):
    documents = []

    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            text = f"{row['question_mm']}\n{row['answer_mm']}"
            metadata = {
                "id": row["id"],
                "topic": row["topic"],
                "difficulty": row["difficulty"],
                "lang": "my",
                "source": row["source_file"]
            }

            documents.append(
                Document(
                    page_content=text.strip(),
                    metadata=metadata
                )
            )

    return documents