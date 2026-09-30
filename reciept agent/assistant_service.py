import time
from llm import get_llm
from rag import retrieve_context

from tools_backup import (
    calculate_return_window,
    lookup_order_status,
    lookup_warranty_policy,
)


# ============================================================
# RECEIPT CONTEXT
# ============================================================

def build_receipt_context(order_id, purchase_date, df):

    if df is None or df.empty:
        return "No receipt is currently available."

    lines = [
        f"Order ID: {order_id or 'Not found'}",
        f"Purchase Date: {purchase_date}",
        "Purchased Items:"
    ]

    for _, row in df.iterrows():

        lines.append(
            f"- Item: {row['Item']}\n"
            f"  Category: {row['Category']}\n"
            f"  Price: {row['Price']}\n"
            f"  Return Deadline: {row.get('Return', 'Unknown')}\n"
            f"  Days Remaining: {row.get('Days', 'Unknown')}\n"
            f"  Return Status: {row.get('Status', 'Unknown')}"
        )

    return "\n".join(lines)


# ============================================================
# FAST ITEM DETECTION - NO LLM CALL
# ============================================================

def find_item_fast(question, df):
    """
    Finds a purchased item using verified receipt data.

    No LLM call is made here.
    """

    if df is None or df.empty:
        return None

    question_lower = question.casefold()

    item_names = (
        df["Item"]
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # EXACT/FULL ITEM MATCH
    # --------------------------------------------------------

    for item in item_names:

        if item.casefold() in question_lower:
            return item

    # --------------------------------------------------------
    # WORD-BASED MATCH
    # Example:
    # "Can I return the jacket?"
    # matches "Blue Jacket"
    # --------------------------------------------------------

    best_item = None
    best_score = 0

    ignored_words = {
        "the",
        "a",
        "an",
        "my",
        "this",
        "that",
        "new",
        "blue",
        "black",
        "white",
        "red",
    }

    question_words = set(
        question_lower
        .replace("?", " ")
        .replace(".", " ")
        .replace(",", " ")
        .split()
    )

    for item in item_names:

        item_words = set(
            item.casefold().split()
        )

        useful_words = (
            item_words - ignored_words
        )

        score = len(
            useful_words
            & question_words
        )

        if score > best_score:
            best_score = score
            best_item = item

    if best_score > 0:
        return best_item

    return None


# ============================================================
# FAST INTENT DETECTION - NO LLM CALL
# ============================================================

def detect_intent_fast(question):
    """
    Fast local intent detection.

    This does not perform return calculations.
    It only determines the type of shopper request.
    """

    q = question.casefold()

    # --------------------------------------------------------
    # COMPLAINT
    # Check before return because a complaint may contain
    # words such as refund/replacement.
    # --------------------------------------------------------

    complaint_phrases = [
        "complaint",
        "complain",
        "damaged",
        "broken",
        "wrong item",
        "wrong product",
        "defective",
        "not working",
        "doesn't work",
        "does not work",
        "arrived damaged",
        "want replacement",
    ]

    if any(
        phrase in q
        for phrase in complaint_phrases
    ):
        return "COMPLAINT"

    # --------------------------------------------------------
    # ORDER STATUS
    # --------------------------------------------------------

    status_phrases = [
        "where is my order",
        "track my order",
        "track order",
        "order status",
        "shipping status",
        "delivery status",
        "tracking number",
        "when will my order arrive",
        "when will it arrive",
        "delivery date",
        "carrier",
        "shipped",
    ]

    if any(
        phrase in q
        for phrase in status_phrases
    ):
        return "ORDER_STATUS"

    # --------------------------------------------------------
    # WARRANTY
    # --------------------------------------------------------

    warranty_phrases = [
        "warranty",
        "warranty claim",
        "under warranty",
        "warranty period",
        "warranty coverage",
        "warranty terms",
    ]

    if any(
        phrase in q
        for phrase in warranty_phrases
    ):
        return "WARRANTY"

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return_phrases = [
        "return",
        "send back",
        "give back",
        "return it",
        "return this",
        "return my",
        "refund",
        "eligible",
        "return window",
        "return deadline",
        "can i send",
        "can i give",
        "don't want",
        "do not want",
        "does not fit",
        "doesn't fit",
        "too small",
        "too large",
        "too big",
    ]

    if any(
        phrase in q
        for phrase in return_phrases
    ):
        return "RETURN"

    # --------------------------------------------------------
    # POLICY
    # --------------------------------------------------------

    policy_phrases = [
        "policy",
        "return rule",
        "return rules",
        "store rules",
        "terms",
    ]

    if any(
        phrase in q
        for phrase in policy_phrases
    ):
        return "POLICY"

    # --------------------------------------------------------
    # RECEIPT
    # --------------------------------------------------------

    receipt_phrases = [
        "receipt",
        "show receipt",
        "view receipt",
        "what did i buy",
        "what did i purchase",
        "items i bought",
        "purchased items",
    ]

    if any(
        phrase in q
        for phrase in receipt_phrases
    ):
        return "RECEIPT"

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_phrases = [
        "summary",
        "summarize",
        "summarise",
        "purchase summary",
        "order summary",
    ]

    if any(
        phrase in q
        for phrase in summary_phrases
    ):
        return "SUMMARY"

    return "OTHER"


# ============================================================
# RETURN TOOL
# ============================================================

def get_return_information(
    item_name,
    purchase_date,
    df
):

    if not item_name:
        return None

    if df is None or df.empty:
        return None

    matching = df[
        df["Item"]
        .astype(str)
        .str.casefold()
        ==
        item_name.casefold()
    ]

    if matching.empty:
        return None

    row = matching.iloc[0]

    return (
        calculate_return_window.invoke(
            {
                "input_str":
                    f"{row['Item']}|"
                    f"{purchase_date}|"
                    f"{row['Category']}"
            }
        )
    )


# ============================================================
# WARRANTY TOOL
# ============================================================

def get_warranty_information(
    item_name,
    df
):

    if not item_name:
        return None

    if df is None or df.empty:
        return None

    matching = df[
        df["Item"]
        .astype(str)
        .str.casefold()
        ==
        item_name.casefold()
    ]

    if matching.empty:
        return None

    category = str(
        matching.iloc[0]["Category"]
    )

    return (
        lookup_warranty_policy.invoke(
            {
                "category": category
            }
        )
    )


# ============================================================
# SAFE RAG RETRIEVAL
# ============================================================

def get_retrieved_context(
    vectorstore,
    question
):

    if vectorstore is None:
        return ""

    try:

        return retrieve_context(
            vectorstore,
            question
        )

    except Exception:

        return ""


# ============================================================
# MAIN ASSISTANT
# ============================================================

def answer_question(
    question,
    vectorstore,
    order_id,
    purchase_date,
    df,
    language="English"
):
    total_start = time.time()

    # ========================================================
    # FAST LOCAL UNDERSTANDING
    # NO OLLAMA CALL
    # ========================================================

    intent = detect_intent_fast(
        question
    )

    item_name = find_item_fast(
        question,
        df
    )

    # ========================================================
    # RAG
    # ========================================================

    rag_start = time.time()

    retrieved_context = get_retrieved_context(
        vectorstore,
        question
    )

    print(
        "RAG TIME:",
        round(time.time() - rag_start, 2),
        "seconds"
    )

    # ========================================================
    # VERIFIED RECEIPT
    # ========================================================

    receipt_context = (
        build_receipt_context(
            order_id,
            purchase_date,
            df
        )
    )

    # ========================================================
    # RETURN TOOL
    # ========================================================

    return_result = None

    if (
        intent == "RETURN"
        and item_name
    ):

        return_result = (
            get_return_information(
                item_name,
                purchase_date,
                df
            )
        )

    # ========================================================
    # ORDER STATUS TOOL
    # ========================================================

    order_result = None

    if intent == "ORDER_STATUS":

        order_result = (
            lookup_order_status.invoke(
                {
                    "order_id":
                        order_id or ""
                }
            )
        )

    # ========================================================
    # WARRANTY TOOL
    # ========================================================

    warranty_result = None

    if (
        intent == "WARRANTY"
        and item_name
    ):

        warranty_result = (
            get_warranty_information(
                item_name,
                df
            )
        )

    # ========================================================
    # VERIFIED EVIDENCE
    # ========================================================

    evidence = f"""
VERIFIED RECEIPT
----------------
{receipt_context}

RETRIEVED POLICY / RECEIPT CONTEXT
----------------------------------
{retrieved_context}

INTENT
------
{intent}

REFERRED ITEM
-------------
{item_name if item_name else "Not identified"}

RETURN TOOL RESULT
------------------
{return_result if return_result else "Not applicable"}

ORDER STATUS RESULT
-------------------
{order_result if order_result else "Not applicable"}

WARRANTY RESULT
---------------
{warranty_result if warranty_result else "Not applicable"}
"""

    # ========================================================
    # ONLY OLLAMA CALL
    # ========================================================

    llm = get_llm()

    prompt = f"""
You are a Smart Order & Warranty Assistant.

Answer the shopper's question using ONLY
the verified evidence below.

SHOPPER QUESTION:
{question}

VERIFIED EVIDENCE:
{evidence}

RULES:

- Never invent a product.
- Never invent a purchase date.
- Never invent a return period.
- Never invent a return deadline.
- Never calculate return dates yourself.
- Use only the deterministic return tool result
  for return calculations.
- Never invent warranty coverage or duration.
- Never invent order status.
- Never invent tracking information.
- Never invent carrier information.
- Never invent delivery information.
- If information is unavailable, clearly say so.
- If an item cannot be identified, ask the shopper
  which purchased item they mean.
- Date eligibility does not by itself guarantee
  final return eligibility.
- The supplied return policy also requires the item
  to be unused and in original packaging.
- Keep the answer short.
- You MUST write the entire answer in {language}.
- Translate all explanatory text into {language}.
- Keep product names, dates, order IDs, prices, and numbers unchanged.
- Preserve product names, dates, order IDs and
  numbers exactly.

Answer:
"""

    try:

        response = llm.invoke(
            prompt
        )

        answer = (
            response.content.strip()
        )

    except Exception as error:

        answer = (
            "The local AI model could not "
            f"generate the response: {error}"
        )

    print(
    "TOTAL ASSISTANT TIME:",
     round(time.time() - total_start, 2),
    "seconds"
    )

    return {
        "answer":
            answer,

        "intent":
            intent,

        "item":
            item_name,

        "return_result":
            return_result,

        "order_result":
            order_result,

        "warranty_result":
            warranty_result,
    }