import json
import time
from sklearn.metrics.pairwise import cosine_similarity
import loader
from pipeline import RAG


def compute_similarity(text1, text2, embedding_model):
    v1 = embedding_model.embed_query(text1)
    v2 = embedding_model.embed_query(text2)
    return cosine_similarity([v1],[v2])[0][0]

def evaluate_retrieval(rag, dataset):
    print("\nEvaluating Retrieval...")

    hit = 0
    total = len(dataset)

    for item in dataset:
        question = item["question"]
        true_answer = item["answer"]

        docs = rag.retrieve(question)

        if any(true_answer in doc for doc in docs):
            hit += 1

    recall = hit / total
    print(f"ecall@{rag.top_k}: {hit}/{total} = {recall:.2f}")

    return recall

def evaluate_generation(rag, dataset):
    print("\n Evaluating Generation...")

    correct = 0
    total = len(dataset)

    for item in dataset:
        question = item["question"]
        true_answer = item["answer"]


        pred = ""
        for token in rag.ask_stream(question):
            pred += token

        if true_answer in pred:
            correct += 1

    acc = correct / total
    print(f"Exact Match Accuracy: {correct}/{total} = {acc:.2f}")

    return acc


def evaluate_semantic(rag, dataset, threshold=0.8):
    print("\n Evaluating Semantic Similarity...")

    correct = 0
    total = len(dataset)

    for item in dataset:
        question = item["question"]
        true_answer = item["answer"]

        pred = ""
        for token in rag.ask_stream(question):
            pred += token

        sim = compute_similarity(pred, true_answer, rag.embedding)

        print(f"[SIM] {sim:.2f} | Q: {question[:30]}")

        if sim >= threshold:
            correct += 1

    score = correct / total
    print(f"Semantic Accuracy: {correct}/{total} = {score:.2f}")

    return score


def evaluate_latency(rag, dataset):
    print("\n Evaluating Latency...")

    times = []

    for item in dataset[:10]:  # sample only
        question = item["question"]

        start = time.time()

        _ = "".join([t for t in rag.ask_stream(question)])

        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)

    print(f"Avg Latency: {avg_time:.2f}s")

    return avg_time


if __name__ == "__main__":
    DATA_PATH = "../dataset/q&a.json" 
    dataset = loader.load_csv(DATA_PATH)
    rag = RAG()
    rag.load_db()

    print("__RAG Evaluation Started__")

    evaluate_retrieval(rag, dataset)
    evaluate_generation(rag, dataset)
    evaluate_semantic(rag, dataset)
    evaluate_latency(rag, dataset)

    print(" Evaluation Complete")