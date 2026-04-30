import csv
from langchain_core.documents import Document

def load_csv(data_path):
    documents = []

    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # combine ALL columns into text
            text = "\n".join([f"{k}: {v}" for k, v in row.items()])

            documents.append(
                Document(
                    page_content=text.strip(),
                    metadata=row
                )
            )

    return documents