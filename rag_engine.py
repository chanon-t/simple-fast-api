import os
import dotenv

from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceEmbeddingOptimizer,
    LongContextReorder,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

from llama_index.core.schema import MetadataMode
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from pinecone import Pinecone
from transformers import AutoTokenizer

dotenv.load_dotenv(override=True)

TOP_SEARCH = 10
HYBRID_SEARCH = True
# CONTEXT_PROMPT = """ You are a SCBX HR (Human Resource) Admin Chatbot, able to have normal interactions, as well as talk\n
#         about Benefit and Documentation related to specific document.\n
#         Here are the relevant documents for the context:\n
#         {context_str}
#         \nInstruction: Use the previous chat history, or the context above, to interact and help the user.\n
#         You are fluent in both English and Thai but you need to answer only Thai language.
# """
CONTEXT_PROMPT = """
    You are an AI chatbot designed to assist employees with various HR-related queries and tasks at SCBX company. Your primary role is to provide accurate, helpful, and timely information tailored to SCBX’s policies and procedures.\n
    Here are the relevant documents for the context:\n
    {context_str}\n
    Instruction: Use the previous chat history, or the context above, to interact and help the user.\n
    You are fluent in both English and Thai, but you will always respond in Thai.\n
    If the information is not available in the knowledge base or if you are unsure, say: "ผมไม่มีข้อมูลนี้ในขณะนี้ กรุณาติดต่อฝ่ายทรัพยากรบุคคลเพื่อขอข้อมูลเพิ่มเติมครับ", Maintain a consistent tone and style in all responses, ending sentences with "ครับ"
"""


class RAGEngine:
    def __init__(
        self,
    ):
        self.chat_engines = {}
        self.chat_v2_engines = {}
        self.index_engines = {}
        self.query_engines = {}

        self.embed_model = CohereEmbedding(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model_name="embed-multilingual-v3.0",
            input_type="search_query",
        )

    def test_query(self, query, model):
        llm = self._get_llm(model)
        resp = llm.complete(query)
        return resp

    def query_messages(self, data):
        if data.engine == "chat":
            chat_engine = self._get_chat_engine(
                namespace=data.namespace,
                llm_name=data.llm,
                custom_system_prompt=data.system_prompt,
            )
            chatMessages = []
            for m in data.messages:
                if m.role == "user":
                    chatMessages.append(
                        ChatMessage(role=MessageRole.USER, content=m.content)
                    )
                elif m.role == "assistant":
                    chatMessages.append(
                        ChatMessage(role=MessageRole.ASSISTANT, content=m.content)
                    )

            response = chat_engine.chat(
                data.query,
                chat_history=chatMessages,
            )
        if data.engine == "chat-v2":
            chat_engine = self._get_chat_v2_engine(
                namespace=data.namespace,
                llm_name=data.llm,
            )
            chatMessages = []
            for m in data.messages:
                if m.role == "user":
                    chatMessages.append(
                        ChatMessage(role=MessageRole.USER, content=m.content)
                    )
                elif m.role == "assistant":
                    chatMessages.append(
                        ChatMessage(role=MessageRole.ASSISTANT, content=m.content)
                    )

            response = chat_engine.chat(
                data.query,
                chat_history=chatMessages,
            )
        elif data.engine == "query":
            query_engine = self._get_query_engine(
                namespace=data.namespace, llm_name=data.llm
            )
            response = query_engine.query(
                data.query,
            )
        elif data.engine == "retrieve":
            retriever = self._get_index(data.namespace, self.embed_model).as_retriever(
                similarity_top_k=5
            )
            nodes = retriever.retrieve(data.query)

            system_prompts = [
                data.system_prompt,
                "\n",
                "These are related knowledge items based on the user’s query, sorted by highest similarity score first:",
            ]

            node_contents = []
            for node in nodes:
                node_contents.append(
                    {
                        "text": node.get_text(),
                        "score": node.get_score(),
                    }
                )
                system_prompts.append(f"- {node.get_text()}")

            system_prompts.append(
                f"\nDo not answer the question if you do not have the information. Suggest consulting other relevant resources for assistance."
            )

            chatMessages = [
                ChatMessage(role=MessageRole.SYSTEM, content="\n".join(system_prompts))
            ]
            for m in data.messages:
                if m.role == "user":
                    chatMessages.append(
                        ChatMessage(role=MessageRole.USER, content=m.content)
                    )
                elif m.role == "assistant":
                    chatMessages.append(
                        ChatMessage(role=MessageRole.ASSISTANT, content=m.content)
                    )
            chatMessages.append(ChatMessage(role=MessageRole.USER, content=data.query))
            llm = self._get_llm(data.llm)
            result = llm.chat(chatMessages)
            response = {"response": result.message.content, "nodes": node_contents}
        return response

    def _get_index(self, namespace, embed_model):
        # if namespace not in self.index_engines.keys():
        #     pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        #     pinecone_index = pc.Index(host=os.environ.get("PINECONE_URL"))
        #     self.index_engines[namespace] = VectorStoreIndex.from_vector_store(
        #         vector_store=PineconeVectorStore(
        #             pinecone_index=pinecone_index,
        #             namespace=namespace,
        #             embed_model=embed_model,
        #             add_sparse_vector=False,
        #         ),
        #         vector_store_query_mode="hybrid" if HYBRID_SEARCH else None,
        #     )
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        pinecone_index = pc.Index(host=os.environ.get("PINECONE_URL"))
        self.index_engines[namespace] = VectorStoreIndex.from_vector_store(
            vector_store=PineconeVectorStore(
                pinecone_index=pinecone_index,
                namespace=namespace,
                embed_model=embed_model,
                add_sparse_vector=False,
            ),
            vector_store_query_mode="hybrid" if HYBRID_SEARCH else None,
        )
        return self.index_engines[namespace]

    def _get_chat_engine(
        self,
        namespace,
        llm_name="gpt",
        custom_system_prompt="default",
        top_k=10,
    ):
        context_prompt = (
            "You are a helpful assistant. You are fluent in both English and Thai but you will always answer in Thai."
            if custom_system_prompt in ("default", "", None)
            else custom_system_prompt
        )

        chat_engine_kwargs = dict(
            chat_mode="condense_plus_context",
            memory=ChatMemoryBuffer.from_defaults(token_limit=3900),
            context_prompt=context_prompt,
            verbose=False,
            embed_model=self.embed_model,
            similarity_top_k=top_k,
            vector_store_query_mode="hybrid",
            alpha=0.6,
            node_postprocessors=[
                LongContextReorder(),
                MetadataReplacementPostProcessor(target_metadata_key="window"),
                CohereRerank(api_key=os.getenv("COHERE_API_KEY"), top_n=20),
                SentenceEmbeddingOptimizer(
                    embed_model=self.embed_model,
                    percentile_cutoff=0.9,
                ),
            ],
        )
        self.chat_engines[namespace] = self._get_index(
            namespace, self.embed_model
        ).as_chat_engine(
            **chat_engine_kwargs,
            llm=self._get_llm(llm_name),
        )

        # if namespace not in self.chat_engines.keys():

        return self.chat_engines[namespace]

    def _get_chat_v2_engine(
        self,
        namespace,
        llm_name="gpt",
    ):
        self.chat_v2_engines[namespace] = self._get_index(
            namespace, self.embed_model
        ).as_chat_engine(
            chat_mode="condense_plus_context",
            similarity_top_k=TOP_SEARCH,
            memory=ChatMemoryBuffer.from_defaults(token_limit=9600),
            context_prompt=CONTEXT_PROMPT,
            verbose=False,
            node_postprocessors=[
                CohereRerank(
                    api_key=os.getenv("COHERE_API_KEY"),
                    model="rerank-multilingual-v3.0",
                    top_n=TOP_SEARCH,
                )
            ],
            embed_model=self.embed_model,
            llm=self._get_llm(llm_name),
        )

        return self.chat_v2_engines[namespace]

    def _get_query_engine(
        self,
        namespace,
        llm_name="gpt",
    ):
        # self.query_engines[namespace] = self._get_index(
        #         namespace, self.embed_model
        #     ).as_query_engine(
        #         llm=self._get_llm(llm_name),
        #     )

        retriever = VectorIndexRetriever(
            index=self._get_index(namespace, self.embed_model),
            similarity_top_k=2,
        )
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.REFINE,
            llm=self._get_llm(llm_name),
        )
        self.query_engines[namespace] = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        # if namespace not in self.chat_engines.keys():

        return self.query_engines[namespace]

    def _get_llm(self, llm_name):
        if llm_name == "typhoon":
            tokenizer = AutoTokenizer.from_pretrained(
                "scb10x/llama-3-typhoon-v1.5x-70b-instruct",
            )
            return OpenAILike(
                api_base=os.getenv("TYPHOON_URL"),
                api_key=os.getenv("TYPHOON_API_KEY"),
                is_chat_model=True,
                context_window=8000,
                max_tokens=768,
                temperature=0,
                model="typhoon-v1.5x-70b-instruct",
                tokenizer=tokenizer,
                is_function_calling_model=False,  # provide this field
                additional_kwargs={
                    "extra_body": {"stop": ["<|eot_id|>"], "repetition_penalty": 1.05}
                },
            )
        if llm_name == "typhoon-rd":
            tokenizer = AutoTokenizer.from_pretrained(
                "scb10x/llama-3-typhoon-v1.5x-70b-instruct",
            )
            return OpenAILike(
                api_base=os.getenv("TYPHOON_RD_URL"),
                api_key=os.getenv("TYPHOON_RD_API_KEY"),
                is_chat_model=True,
                context_window=8000,
                max_tokens=768,
                temperature=0,
                model="typhoon-v1.5x-70b-instruct",
                tokenizer=tokenizer,
                is_function_calling_model=False,  # provide this field
                additional_kwargs={
                    "extra_body": {"stop": ["<|eot_id|>"], "repetition_penalty": 1.05}
                },
            )
        elif llm_name == "gpt":
            return OpenAI(
                is_chat_model=True,
                context_window=8000,
                max_tokens=768,
                temperature=0,
                model="gpt-4o",
                is_function_calling_model=False,
            )
        elif llm_name == "gpt-mini":
            return OpenAI(
                is_chat_model=True,
                context_window=8000,
                max_tokens=768,
                temperature=0,
                model="gpt-4o-mini",
                is_function_calling_model=False,
            )
        elif llm_name == "azure":
            return AzureOpenAI(
                model="gpt-4o",
                deployment_name="rnd-poc",
                api_key=os.getenv("AZURE_API_KEY"),
                azure_endpoint=os.getenv("AZURE_URL"),
                api_version="2024-02-01",
                is_chat_model=True,
                context_window=8000,
                max_tokens=768,
                temperature=0,
                is_function_calling_model=False,
            )
        else:
            raise NotImplementedError()
