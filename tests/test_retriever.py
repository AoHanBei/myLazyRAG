import RAGproject.retriever as retriever_module
from RAGproject import create_retriever


class FakeResult:
    def __init__(self, content: str):
        self.content = content

    def get_content(self):
        return self.content


def test_create_retriever_returns_results(monkeypatch):
    calls = {}

    class FakeDocument:
        def __init__(self, path):
            calls["path"] = path

    class FakeRetriever:
        def __init__(self, doc, group_name, similarity, topk):
            calls["doc"] = doc
            calls["group_name"] = group_name
            calls["similarity"] = similarity
            calls["topk"] = topk

        def __call__(self, query):
            calls["query"] = query
            return [FakeResult("这是一个测试检索结果，包含北京奥运会。")]

    monkeypatch.setattr(
        retriever_module,
        "_load_lazyllm_rag",
        lambda: (FakeDocument, FakeRetriever),
    )

    results = create_retriever("./data_kb", "为我介绍一下北京奥运会")

    assert len(results) == 1
    assert "北京奥运会" in results[0].get_content()
    assert calls["path"] == "./data_kb"
    assert calls["query"] == "为我介绍一下北京奥运会"
    assert calls["group_name"] == "CoarseChunk"
    assert calls["similarity"] == "bm25_chinese"
    assert calls["topk"] == 3


def test_create_retriever_empty_query_returns_list(monkeypatch):
    class FakeDocument:
        def __init__(self, path):
            self.path = path

    class FakeRetriever:
        def __init__(self, doc, group_name, similarity, topk):
            pass

        def __call__(self, query):
            return []

    monkeypatch.setattr(
        retriever_module,
        "_load_lazyllm_rag",
        lambda: (FakeDocument, FakeRetriever),
    )

    results = create_retriever("./data_kb", "")

    assert isinstance(results, list)
