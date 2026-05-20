from lazyllm import Document, Retriever


def create_retriever(path: str, query: str):
    """
    创建并执行检索。

    Args:
        path: 文档路径。
        query: 查询语句。

    Returns:
        list: 检索结果。
    """
    doc = Document(path)
    retriever = Retriever(
        doc,
        group_name="CoarseChunk",
        similarity="bm25_chinese",
        topk=3,
    )
    return retriever(query)