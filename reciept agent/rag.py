import shutil
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_BASE_DIR = Path("chroma_db")

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CACHED EMBEDDING MODEL
# ============================================================

@lru_cache(maxsize=1)
def get_embeddings():
    """
    Load the HuggingFace embedding model once.

    Subsequent calls reuse the same model instead
    of loading it again.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# ============================================================
# RECEIPT DOCUMENTS
# ============================================================

def create_receipt_documents(
    order_id,
    purchase_date,
    items
):
    """
    Create exactly one document for each receipt item.
    """

    documents = []

    for item in items:

        text = (
            f"Order ID: {order_id}\n"
            f"Purchase Date: {purchase_date}\n"
            f"Item: {item['name']}\n"
            f"Category: {item['category']}\n"
            f"Price: {item['price']}"
        )

        document = Document(
            page_content=text,
            metadata={
                "source": "receipt",
                "order_id": str(
                    order_id or ""
                ),
                "item_name": str(
                    item["name"]
                ),
                "category": str(
                    item["category"]
                ),
            }
        )

        documents.append(
            document
        )

    return documents


# ============================================================
# POLICY DOCUMENTS
# ============================================================

def create_policy_documents(
    policy_text
):
    """
    Chunk the policy separately from receipt items.
    """

    if not policy_text:
        return []

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30
        )
    )

    chunks = splitter.split_text(
        policy_text
    )

    documents = []

    for index, chunk in enumerate(
        chunks
    ):

        document = Document(
            page_content=chunk,
            metadata={
                "source": "policy",
                "chunk": index,
            }
        )

        documents.append(
            document
        )

    return documents


# ============================================================
# SAFE PATH NAME
# ============================================================

def safe_name(value):
    """
    Convert username/order ID into a safe directory name.
    """

    value = str(
        value or "unknown"
    )

    allowed = []

    for character in value:

        if (
            character.isalnum()
            or character in "-_"
        ):
            allowed.append(
                character
            )

        else:
            allowed.append(
                "_"
            )

    return "".join(
        allowed
    )


# ============================================================
# VECTOR STORE
# ============================================================

def build_vectorstore(
    username,
    order_id,
    purchase_date,
    items,
    policy_text
):
    """
    Build a Chroma vector store scoped to one
    shopper and one uploaded receipt.
    """

    username_safe = safe_name(
        username
    )

    order_safe = safe_name(
        order_id
    )

    persist_directory = (
        CHROMA_BASE_DIR
        / username_safe
        / order_safe
    )

    # Remove the previous vector store for
    # this same shopper/order before rebuilding.
    if persist_directory.exists():

        shutil.rmtree(
            persist_directory,
            ignore_errors=True
        )

    persist_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Receipt documents
    # --------------------------------------------------------

    receipt_documents = (
        create_receipt_documents(
            order_id=order_id,
            purchase_date=purchase_date,
            items=items,
        )
    )

    # --------------------------------------------------------
    # Policy documents
    # --------------------------------------------------------

    policy_documents = (
        create_policy_documents(
            policy_text
        )
    )

    documents = (
        receipt_documents
        + policy_documents
    )

    if not documents:

        raise ValueError(
            "No receipt or policy documents "
            "were available for indexing."
        )

    # IMPORTANT:
    # This returns the SAME cached embedding
    # model after its first initialization.
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(
            persist_directory
        ),
        collection_name=(
            "shopper_receipt"
        ),
    )

    return vectorstore


# ============================================================
# RETRIEVER
# ============================================================

def get_retriever(
    vectorstore
):
    """
    Create a small retriever.

    k=3 is enough for our small receipt/policy dataset
    and avoids unnecessary retrieval work.
    """

    return vectorstore.as_retriever(
        search_kwargs={
            "k": 3
        }
    )


# ============================================================
# RETRIEVE CONTEXT
# ============================================================

def retrieve_context(
    vectorstore,
    query
):
    """
    Retrieve relevant receipt and policy context.
    """

    if vectorstore is None:
        return ""

    retriever = get_retriever(
        vectorstore
    )

    documents = retriever.invoke(
        query
    )

    if not documents:
        return ""

    return "\n\n".join(
        document.page_content
        for document in documents
    )


# ============================================================
# RETRIEVAL TEST
# ============================================================

def test_retrieval(
    vectorstore,
    query=(
        "When can I return the jacket?"
    )
):
    """
    Simple test helper for assignment/demo validation.
    """

    return retrieve_context(
        vectorstore,
        query
    )