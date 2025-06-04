import os
import sys
from icecream import ic
import json
from pathlib import Path
from dotenv import load_dotenv
sys.path.append(str(Path(__file__).parent.parent.parent))

from source.settings import setting
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import NodeWithScore, Node, TextNode
from llama_index.core import Settings
from llama_index.core.postprocessor.types import BaseNodePostprocessor


def MapReducePostProcessor(BaseNodePostprocessor):
    def __init__(self):
        self.setting = setting

        self.llm = OpenAI(
            model=self.setting.model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            logprobs=None,
            default_headers={},
        )
        Settings.llm = self.llm

    def _get_prompt_mapping():
        return [
                ChatMessage(
                    role="system",
                    content="Use the following portion of a long document to see if any of the text is relevant to answer the question.\
                        Return any relevant text verbatim.",
                ),
                ChatMessage(
                    role="user",
                    content="{context} \
                            Human: {question}"
                            .format(WHOLE_DOCUMENT=whole_document, CHUNK_CONTENT=chunk.text
                    ),
                ),
            ]


    def _mapping(node: NodeWithScore, query_bundle: QueryBundle):
        messages = [
                ChatMessage(
                    role="system",
                    content="Use the following portion of a long document to see if any of the text is relevant to answer the question.\
                        Return any relevant text verbatim.",
                ),
                ChatMessage(
                    role="user",
                    content="{context}\n\
                            Human: {question}"
                            .format(
                                context=node.text, 
                                question=query_bundle.query_str
                        ),
                ),
            ]

        response = self.llm.chat(messages)
        mapping_content = response.message.content
        return mapping_content

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore], 
        query_bundle:QueryBundle
    )->List[NodeWithScore]:
        '''
        Postprocess node
        '''
        mapping_nodes = [NodeWithScore(node=Node(text=self._mapping(node, query_bundle))) for node in nodes]
        return mapping_nodes

