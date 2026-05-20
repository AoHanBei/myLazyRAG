from typing import Any


def _load_lazyllm_rag():
    from lazyllm import Document, Retriever

    return Document, Retriever


def create_retriever(path: str, query: str) -> list[Any]:
    """
    创建并执行检索。

    Args:
        path: 文档路径。
        query: 查询语句。

    Returns:
        list: 检索结果。
    """
    document_cls, retriever_cls = _load_lazyllm_rag()

    doc = document_cls(path)
    retriever = retriever_cls(
        doc,
        group_name="CoarseChunk",
        similarity="bm25_chinese",
        topk=3,
    )
    return retriever(query)
