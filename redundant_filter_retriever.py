from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever
from typing import List


class RedundantFilterRetriever(BaseRetriever):

    embeddings: Embeddings
    chroma: Chroma

    # def __init__(self, embeddings: Embeddings, chroma: Chroma):
    #     super().__init__()
    #     self.embeddings = embeddings
    #     self.chroma = chroma

    def get_relevant_documents(self, query: str) -> List:
        # calculate embeddings for the 'query' string
        emb = self.embeddings.embed_query(query)

        # take embeddings and feed them into max_marginal_relevance_search_by_vector
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8
        )

    async def aget_relevant_documents(self, query: str) -> List:
        # calculate embeddings for the 'query' string asynchronously
        emb = await self.embeddings.embed_query_async(query)

        # take embeddings and feed them into max_marginal_relevance_search_by_vector
        return await self.chroma.max_marginal_relevance_search_by_vector_async(
            embedding=emb,
            lambda_mult=0.8
        )
