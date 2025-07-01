# ======================================
# File: AskAIEngine.py
# Purpose: Injects Haystack-powered QA into FridayAI for real factual answering.
# ======================================

from haystack.nodes import FARMReader, EmbeddingRetriever
from haystack.document_stores import FAISSDocumentStore
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import PreProcessor
import os

class AskAIEngine:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.store = FAISSDocumentStore(faiss_index_factory_str="Flat")
        self._load_documents()
        self._setup_pipeline()

    def _load_documents(self):
        from haystack.nodes import TextConverter
        converter = TextConverter(remove_numeric_tables=True, valid_languages=["en"])
        preprocessor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            split_by="word",
            split_length=200,
            split_overlap=20,
            split_respect_sentence_boundary=True,
        )

        docs = []
        for fname in os.listdir(self.data_dir):
            if fname.endswith(".txt"):
                path = os.path.join(self.data_dir, fname)
                raw = converter.convert(file_path=path, meta={"name": fname})
                processed = preprocessor.process(raw)
                docs.extend(processed)

        self.store.write_documents(docs)

    def _setup_pipeline(self):
        retriever = EmbeddingRetriever(
            document_store=self.store,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            use_gpu=False
        )
        self.store.update_embeddings(retriever)

        reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
        self.pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)

    def answer(self, question: str) -> str:
        result = self.pipeline.run(query=question, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}})
        answers = result.get("answers", [])

        if answers and answers[0].answer:
            return answers[0].answer
        else:
            return "I'm not sure about that yet — but I'm learning every day."


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    engine = AskAIEngine()
    while True:
        q = input("Ask Friday something: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("Friday:", engine.answer(q))
