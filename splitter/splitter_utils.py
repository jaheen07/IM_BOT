from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Chapter"),
    ("##", "Section"),
    ("###", "Subsection"),
    ("####", "Sub-subsection")
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

