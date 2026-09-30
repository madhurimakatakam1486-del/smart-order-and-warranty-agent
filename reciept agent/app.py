import json
from pathlib import Path

import pandas as pd
import streamlit as st

from receipt_parser import parse_receipt, extract_text_from_image
from tools import (
    calculate_return_window,
    calculate_warranty_window,
    lookup_warranty_policy,
    lookup_order_status,
    create_return_request,
    create_warranty_claim,
    generate_complaint,
)
from rag import build_vectorstore
from assistant_service import answer_question


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Order & Warranty Agent",
    page_icon="🛍️",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

POLICY_FILE = DATA_DIR / "policy.txt"
USERS_FILE = BASE_DIR / "users.json"


# ============================================================
# LANGUAGE OPTIONS
# ============================================================

LANGUAGES = {
    "English": "English",
    "తెలుగు": "Telugu",
    "हिन्दी": "Hindi",
}


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_SESSION_VALUES = {
    "logged_in": False,
    "username": None,

    "receipt_processed": False,
    "receipt_name": None,
    "receipt_text": "",

    "order_id": None,
    "purchase_date": None,
    "items": [],
    "receipt_df": None,

    "vectorstore": None,

    "chat_history": [],
    "last_result": None,

    "selected_item": None,
    "active_action": None,
}


for key, value in DEFAULT_SESSION_VALUES.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# USERS
# ============================================================

def load_users():
    """
    Load users from users.json.

    Supports both old and new account formats.
    """

    if not USERS_FILE.exists():
        return {}

    try:

        with open(
            USERS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:
        return {}


def save_users(users):
    """
    Save users to users.json.
    """

    with open(
        USERS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            users,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# PASSWORD VERIFICATION
# ============================================================

def verify_password(
    users,
    username,
    password
):
    """
    Supports the old account format:

        "madhurima": "1234"

    and the new account format:

        "username": {
            "password": "...",
            "security_question": "...",
            "recovery_answer": "..."
        }
    """

    if username not in users:
        return False

    user_data = users[username]

    # Old account format
    if isinstance(user_data, str):
        return user_data == password

    # New account format
    if isinstance(user_data, dict):
        return (
            user_data.get("password")
            == password
        )

    return False


# ============================================================
# POLICY
# ============================================================

def load_policy():
    """
    Load the supplied store policy.
    """

    if not POLICY_FILE.exists():
        return ""

    try:

        return POLICY_FILE.read_text(
            encoding="utf-8"
        )

    except Exception:
        return ""


# ============================================================
# RESET RECEIPT SESSION
# ============================================================

def reset_receipt_state():

    st.session_state.receipt_processed = False
    st.session_state.receipt_name = None
    st.session_state.receipt_text = ""

    st.session_state.order_id = None
    st.session_state.purchase_date = None
    st.session_state.items = []

    st.session_state.receipt_df = None
    st.session_state.vectorstore = None

    st.session_state.chat_history = []
    st.session_state.last_result = None

    st.session_state.selected_item = None
    st.session_state.active_action = None


# ============================================================
# AUTHENTICATION PAGE
# ============================================================

def show_login():

    st.title(
        "🛍️ Smart Order & Warranty Agent"
    )

    st.caption(
        "Upload your receipt, check return windows, "
        "track orders, and get grounded shopping support."
    )

    login_tab, signup_tab, forgot_tab = st.tabs(
        [
            "Login",
            "Create Account",
            "Forgot Password",
        ]
    )

    # ========================================================
    # LOGIN
    # ========================================================

    with login_tab:

        username = st.text_input(
            "Username",
            key="login_username"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button(
            "Login",
            type="primary",
            use_container_width=True
        ):

            users = load_users()

            username = username.strip()

            if verify_password(
                users,
                username,
                password
            ):

                st.session_state.logged_in = True
                st.session_state.username = username

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

    # ========================================================
    # CREATE ACCOUNT
    # ========================================================

    with signup_tab:

        st.subheader(
            "Create Account"
        )

        new_username = st.text_input(
            "Choose Username",
            key="signup_username"
        )

        new_password = st.text_input(
            "Choose Password",
            type="password",
            key="signup_password"
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            key="signup_confirm"
        )

        security_questions = [
            "What is your favorite city?",
            "What was the name of your first school?",
            "What is your favorite food?",
        ]

        security_question = st.selectbox(
            "Security Question",
            security_questions,
            key="signup_security_question"
        )

        recovery_answer = st.text_input(
            "Recovery Answer",
            type="password",
            key="signup_recovery_answer",
            help=(
                "Remember this answer. "
                "You will need it if you forget "
                "your password."
            )
        )

        if st.button(
            "Create Account",
            use_container_width=True
        ):

            users = load_users()

            new_username = (
                new_username.strip()
            )

            if not new_username:

                st.warning(
                    "Please enter a username."
                )

            elif not new_password:

                st.warning(
                    "Please enter a password."
                )

            elif len(new_password) < 4:

                st.warning(
                    "Password must contain at least "
                    "4 characters."
                )

            elif (
                new_password
                != confirm_password
            ):

                st.error(
                    "Passwords do not match."
                )

            elif not recovery_answer.strip():

                st.warning(
                    "Please enter a recovery answer."
                )

            elif new_username in users:

                st.error(
                    "That username already exists."
                )

            else:

                users[new_username] = {
                    "password":
                        new_password,

                    "security_question":
                        security_question,

                    "recovery_answer":
                        recovery_answer
                        .strip()
                        .casefold(),
                }

                save_users(users)

                st.success(
                    "✅ Account created successfully. "
                    "You can now log in."
                )

    # ========================================================
    # FORGOT PASSWORD
    # ========================================================

    with forgot_tab:

        st.subheader(
            "Reset Password"
        )

        recovery_username = st.text_input(
            "Username",
            key="forgot_username"
        )

        username_key = (
            recovery_username.strip()
        )

        if username_key:

            users = load_users()

            if username_key not in users:

                st.error(
                    "Account not found."
                )

            else:

                user_data = users[
                    username_key
                ]

                # --------------------------------------------
                # OLD USER FORMAT
                # --------------------------------------------

                if isinstance(
                    user_data,
                    str
                ):

                    st.warning(
                        "Password recovery is not "
                        "configured for this account. "
                        "Log in using your existing password "
                        "and configure recovery from Profile."
                    )

                # --------------------------------------------
                # NEW USER FORMAT
                # --------------------------------------------

                elif isinstance(
                    user_data,
                    dict
                ):

                    question = (
                        user_data.get(
                            "security_question"
                        )
                    )

                    if not question:

                        st.warning(
                            "Password recovery is not "
                            "configured for this account."
                        )

                    else:

                        st.info(
                            f"Security Question: "
                            f"**{question}**"
                        )

                        answer = st.text_input(
                            "Recovery Answer",
                            type="password",
                            key="forgot_answer"
                        )

                        new_reset_password = (
                            st.text_input(
                                "New Password",
                                type="password",
                                key="forgot_new_password"
                            )
                        )

                        confirm_reset_password = (
                            st.text_input(
                                "Confirm New Password",
                                type="password",
                                key="forgot_confirm_password"
                            )
                        )

                        if st.button(
                            "Reset Password",
                            type="primary",
                            use_container_width=True
                        ):

                            saved_answer = (
                                user_data.get(
                                    "recovery_answer",
                                    ""
                                )
                            )

                            entered_answer = (
                                answer
                                .strip()
                                .casefold()
                            )

                            if (
                                entered_answer
                                != saved_answer
                            ):

                                st.error(
                                    "Incorrect recovery answer."
                                )

                            elif not new_reset_password:

                                st.warning(
                                    "Please enter a new password."
                                )

                            elif (
                                len(
                                    new_reset_password
                                )
                                < 4
                            ):

                                st.warning(
                                    "Password must contain "
                                    "at least 4 characters."
                                )

                            elif (
                                new_reset_password
                                !=
                                confirm_reset_password
                            ):

                                st.error(
                                    "New passwords do not match."
                                )

                            else:

                                users[
                                    username_key
                                ][
                                    "password"
                                ] = (
                                    new_reset_password
                                )

                                save_users(
                                    users
                                )

                                st.success(
                                    "✅ Password reset "
                                    "successfully. "
                                    "You can now log in "
                                    "with your new password."
                                )


# ============================================================
# SIDEBAR
# ============================================================

def show_sidebar():

    with st.sidebar:

        st.header(
            "🛍️ Smart Agent"
        )

        st.write(
            f"Signed in as "
            f"**{st.session_state.username}**"
        )

        language_label = st.selectbox(
            "🌐 Language",
            list(
                LANGUAGES.keys()
            )
        )

        language = LANGUAGES[
            language_label
        ]

        st.divider()

        page = st.radio(
            "Navigation",
            [
                "Agent",
                "Order Status",
                "Profile",
            ]
        )

        st.divider()

        if st.button(
            "Logout",
            use_container_width=True
        ):

            reset_receipt_state()

            st.session_state.logged_in = False
            st.session_state.username = None

            st.rerun()

    return page, language


# ============================================================
# PROCESS RECEIPT
# ============================================================

def process_receipt(
    uploaded_file
):

    uploaded_file.seek(0)

    receipt_text = (
        extract_text_from_image(
            uploaded_file
        )
    )

    uploaded_file.seek(0)

    order_id, purchase_date, items = (
        parse_receipt(
            uploaded_file
        )
    )

    if not receipt_text:

        st.error(
            "I could not read text "
            "from this receipt."
        )

        return False

    if not purchase_date:

        st.error(
            "I could not identify "
            "the purchase date."
        )

        return False

    if not items:

        st.error(
            "I could not identify "
            "any purchased items."
        )

        return False

    rows = []

    # ========================================================
    # AUTOMATIC RETURN CALCULATION
    # ========================================================

    for item in items:

        result = (
            calculate_return_window.invoke(
                {
                    "input_str":
                        f"{item['name']}|"
                        f"{purchase_date}|"
                        f"{item['category']}"
                }
            )
        )

        warranty_result = (
            calculate_warranty_window.invoke(
                {
                    "input_str":
                        f"{item['name']}|"
                        f"{purchase_date}|"
                        f"{item['category']}"
                }
            )
        )

                # Warranty information
        if warranty_result.get("success"):

            if warranty_result.get("warranty_available"):

                warranty_deadline = warranty_result.get(
                    "warranty_deadline"
                )

                warranty_days = warranty_result.get(
                    "warranty_days_remaining"
                )

                if warranty_result.get("warranty_expired"):
                    warranty_status = "Expired"
                else:
                    warranty_status = "Under Warranty"

            else:

                warranty_deadline = "No warranty"
                warranty_days = None
                warranty_status = "No Warranty"

        else:

            warranty_deadline = "Not specified"
            warranty_days = None
            warranty_status = "Policy unavailable"

        if result.get("success"):

            if result.get("expired"):

                status = "Expired"

            elif result.get(
                "expiring_soon"
            ):

                status = (
                    "Expiring Soon"
                )

            else:

                status = (
                    "Within Date Window"
                )

            rows.append(
                {
                    "Item":
                        item["name"],

                    "Category":
                        item["category"],

                    "Price":
                        item["price"],

                    "Return":
                        result.get(
                            "return_deadline"
                        ),

                    "Days":
                        result.get(
                            "days_remaining"
                        ),

                    "Status":
                        status,

                    "Warranty":
                        warranty_deadline,

                    "Warranty Days":
                        warranty_days,

                    "Warranty Status":
                        warranty_status,

                    "Policy Found":
                        True,
                }
            )

        else:

            rows.append(
                {
                    "Item":
                        item["name"],

                    "Category":
                        item["category"],

                    "Price":
                        item["price"],

                    "Return":
                        "Not specified",

                    "Days":
                        None,

                    "Status":
                        "Policy unavailable",

                    "Warranty":
                        warranty_deadline,

                    "Warranty Days":
                        warranty_days,

                    "Warranty Status":
                        warranty_status,

                    "Policy Found":
                        False,
                }
            )

    df = pd.DataFrame(
        rows
    )

    policy_text = load_policy()

    if not policy_text:

        st.error(
            "Policy file was not found "
            "at data/policy.txt."
        )

        return False

    # ========================================================
    # BUILD RAG VECTOR STORE
    # ========================================================

    try:

        vectorstore = (
            build_vectorstore(
                username=(
                    st.session_state.username
                ),
                order_id=(
                    order_id
                    or "unknown-order"
                ),
                purchase_date=(
                    purchase_date
                ),
                items=items,
                policy_text=(
                    policy_text
                ),
            )
        )

    except Exception as error:

        st.error(
            "Unable to build the "
            f"knowledge base: {error}"
        )

        return False

    # ========================================================
    # STORE SESSION DATA
    # ========================================================

    st.session_state.receipt_processed = True

    st.session_state.receipt_name = (
        uploaded_file.name
    )

    st.session_state.receipt_text = (
        receipt_text
    )

    st.session_state.order_id = (
        order_id
    )

    st.session_state.purchase_date = (
        purchase_date
    )

    st.session_state.items = (
        items
    )

    st.session_state.receipt_df = (
        df
    )

    st.session_state.vectorstore = (
        vectorstore
    )

    st.session_state.chat_history = []
    st.session_state.last_result = None
    st.session_state.active_action = None

    return True


# ============================================================
# PURCHASE SUMMARY
# ============================================================

def show_purchase_summary():

    df = (
        st.session_state.receipt_df
    )

    if df is None or df.empty:
        return

    st.subheader(
        "🧾 Purchase Summary"
    )
    st.caption(
    "Quick overview of your order, return deadlines "
    "and warranty coverage."
)

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "Order ID",
        st.session_state.order_id
        or "Not found"
    )

    col2.metric(
        "Purchase Date",
        st.session_state.purchase_date
    )

    col3.metric(
        "Items",
        len(df)
    )
    return_active = len(
    df[df["Days"] > 0]
)

    return_expiring = len(
    df[
        (df["Days"] > 0)
        & (df["Days"] < 7)
    ]
)

    under_warranty = len(
    df[
        df["Warranty Status"]
        == "Under Warranty"
    ]
)

    status_col1, status_col2, status_col3 = (
    st.columns(3)
)

    status_col1.metric(
    "↩️ Return Windows Open",
    return_active
)

    status_col2.metric(
    "⚠️ Expiring Soon",
    return_expiring
)

    status_col3.metric(
    "🛡️ Under Warranty",
    under_warranty
)

    display_df = (
    df[
        [
            "Item",
            "Category",
            "Price",
            "Return",
            "Days",
            "Status",
            "Warranty",
            "Warranty Days",
            "Warranty Status",
        ]
    ].copy()
)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # PROACTIVE ALERTS
    # ========================================================

    for _, row in df.iterrows():

        days = row["Days"]

        if pd.isna(days):
            continue

        days = int(days)

        if days <= 0:

            st.error(
                f"⛔ {row['Item']}: "
                f"the return window has expired. "
                f"Return deadline: {row['Return']}."
            )

        elif days < 7:

            day_word = (
                "day"
                if days == 1
                else "days"
            )

            st.warning(
                f"🔔 {row['Item']}: "
                f"your return window ends "
                f"in {days} {day_word} "
                f"({row['Return']})."
            )

    st.caption(
        "Date eligibility alone does not guarantee "
        "a return. The supplied policy also requires "
        "the item to be unused and in original packaging."
    )


# ============================================================
# AI CHAT
# ============================================================

def show_chat(
    language
):
    st.subheader(
    "💬 Ask Your Smart Agent"
)

    st.caption(
    "Ask questions about returns, warranties, "
    "order status, or any item on your receipt."
)

    st.info(
    '💡 Try asking: "Can I return my Blue Jacket?" '
    'or "Is my Wireless Mouse still under warranty?"'
)
    

    for message in (
        st.session_state.chat_history
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    question = st.chat_input(
        "Ask about your purchase, "
        "return, warranty, or order..."
    )

    if question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Checking your receipt "
                "and policy..."
            ):

                try:

                    result = (
                        answer_question(
                            question=question,

                            vectorstore=(
                                st.session_state
                                .vectorstore
                            ),

                            order_id=(
                                st.session_state
                                .order_id
                            ),

                            purchase_date=(
                                st.session_state
                                .purchase_date
                            ),

                            df=(
                                st.session_state
                                .receipt_df
                            ),

                            language=language,
                        )
                    )

                    answer = (
                        result["answer"]
                    )

                    st.markdown(
                        answer
                    )

                    st.session_state.last_result = (
                        result
                    )

                    if result.get(
                        "item"
                    ):

                        st.session_state.selected_item = (
                            result["item"]
                        )

                    st.session_state.chat_history.append(
                        {
                            "role":
                                "assistant",

                            "content":
                                answer,
                        }
                    )

                except Exception as error:

                    st.error(
                        f"Assistant error: "
                        f"{error}"
                    )

                    return

    show_dynamic_actions()


# ============================================================
# DYNAMIC ACTIONS
# ============================================================

def show_dynamic_actions():

    result = (
        st.session_state.last_result
    )

    if not result:
        return

    intent = result.get(
        "intent"
    )

    item = result.get(
        "item"
    )

    return_result = result.get(
        "return_result"
    )

    warranty_result = result.get(
        "warranty_result"
    )

    st.divider()

    st.subheader(
        "Available Actions"
    )

    # ========================================================
    # RETURN
    # ========================================================

    if (
        intent == "RETURN"
        and return_result
    ):

        if (
            return_result.get(
                "success"
            )
            and
            not return_result.get(
                "expired"
            )
        ):

            col1, col2, col3 = (
                st.columns(3)
            )

            with col1:

                if st.button(
                    "📦 Start Return",
                    use_container_width=True
                ):

                    st.session_state.active_action = (
                        "return"
                    )

                    st.session_state.selected_item = (
                        item
                    )

                    st.rerun()

            with col2:

                if st.button(
                    "🧾 View Receipt",
                    use_container_width=True
                ):

                    st.session_state.active_action = (
                        "receipt"
                    )

                    st.rerun()

            with col3:

                if st.button(
                    "💬 Contact Support",
                    use_container_width=True
                ):

                    st.session_state.active_action = (
                        "complaint"
                    )

                    st.rerun()

        else:

            col1, col2 = (
                st.columns(2)
            )

            with col1:

                if st.button(
                    "📖 Why am I not eligible?",
                    use_container_width=True
                ):

                    st.session_state.active_action = (
                        "ineligible"
                    )

                    st.rerun()

            with col2:

                if st.button(
                    "💬 Contact Support",
                    use_container_width=True
                ):

                    st.session_state.active_action = (
                        "complaint"
                    )

                    st.rerun()

    # ========================================================
    # WARRANTY
    # ========================================================

    elif intent == "WARRANTY":

        if (
            warranty_result
            and warranty_result.get(
                "found"
            )
        ):

            col1, col2 = (
                st.columns(2)
            )

            with col1:

                if st.button(
                    "🛡️ Start Warranty Claim",
                    use_container_width=True
                ):

                    st.session_state.active_action = (
                        "warranty"
                    )

                    st.rerun()

            with col2:

                if st.button(
                    "📋 View Warranty Terms",
                    use_container_width=True
                ):

                    st.session_state.active_action = (
                        "warranty_terms"
                    )

                    st.rerun()

        else:

            if st.button(
                "💬 Contact Support",
                use_container_width=True
            ):

                st.session_state.active_action = (
                    "complaint"
                )

                st.rerun()

    # ========================================================
    # COMPLAINT
    # ========================================================

    elif intent == "COMPLAINT":

        if st.button(
            "📝 Generate Complaint",
            use_container_width=True
        ):

            st.session_state.active_action = (
                "complaint"
            )

            st.rerun()

    # ========================================================
    # RECEIPT
    # ========================================================

    elif intent in {
        "RECEIPT",
        "SUMMARY",
    }:

        if st.button(
            "🧾 View Receipt",
            use_container_width=True
        ):

            st.session_state.active_action = (
                "receipt"
            )

            st.rerun()


# ============================================================
# RETURN WORKFLOW
# ============================================================

def show_return_workflow():

    item_name = (
        st.session_state.selected_item
    )

    if not item_name:

        st.warning(
            "Please ask about the item "
            "you want to return first."
        )

        return

    df = (
        st.session_state.receipt_df
    )

    matching = df[
        df["Item"]
        .astype(str)
        .str.casefold()
        ==
        item_name.casefold()
    ]

    if matching.empty:

        st.error(
            "The selected item was not "
            "found on the receipt."
        )

        return

    row = matching.iloc[0]

    st.subheader(
        f"📦 Return Request — "
        f"{item_name}"
    )

    if row["Status"] == "Expired":

        st.error(
            "The date-based return window "
            f"ended on {row['Return']}."
        )

        return

    if not bool(
        row["Policy Found"]
    ):

        st.warning(
            "A return period is not available "
            "for this item category."
        )

        return

    st.info(
        f"Return deadline: "
        f"**{row['Return']}**"
    )

    st.write(
        "The supplied policy requires "
        "the item to be **unused and "
        "in original packaging**."
    )

    unused = st.checkbox(
        "The item is unused."
    )

    original_packaging = (
        st.checkbox(
            "The item is in its "
            "original packaging."
        )
    )

    reason = st.text_area(
        "Reason for return",
        placeholder=(
            "Example: The jacket "
            "does not fit."
        )
    )

    if st.button(
        "Prepare Return Request",
        type="primary"
    ):

        if not unused:

            st.warning(
                "The policy requires "
                "the item to be unused."
            )

        elif not original_packaging:

            st.warning(
                "The policy requires "
                "original packaging."
            )

        elif not reason.strip():

            st.warning(
                "Please enter the "
                "reason for return."
            )

        else:

            request = (
                create_return_request(
                    st.session_state.order_id,
                    item_name,
                    reason.strip()
                )
            )

            st.success(
                "✅ Return request prepared."
            )

            st.json(
                request
            )

            st.caption(
                "This demonstration prepares "
                "the request locally. "
                "It does not submit the return "
                "to an external retailer."
            )

    if st.button(
        "← Back to Agent"
    ):

        st.session_state.active_action = None
        st.rerun()


# ============================================================
# RECEIPT VIEW
# ============================================================

def show_receipt():

    st.subheader(
        "🧾 Uploaded Receipt"
    )

    if (
        st.session_state.receipt_text
    ):

        st.text_area(
            "Extracted receipt text",

            st.session_state.receipt_text,

            height=300,

            disabled=True
        )

    else:

        st.info(
            "Receipt text is unavailable."
        )

    if st.button(
        "← Back to Agent"
    ):

        st.session_state.active_action = None
        st.rerun()


# ============================================================
# INELIGIBILITY EXPLANATION
# ============================================================

def show_ineligibility():

    result = (
        st.session_state.last_result
    )

    st.subheader(
        "📖 Return Eligibility"
    )

    if not result:

        st.info(
            "No return result is "
            "currently available."
        )

        return

    return_result = (
        result.get(
            "return_result"
        )
    )

    if not return_result:

        st.info(
            "No verified return calculation "
            "is available."
        )

    elif not return_result.get(
        "success"
    ):

        st.warning(
            return_result.get(
                "error",
                "No matching return "
                "policy was found."
            )
        )

    elif return_result.get(
        "expired"
    ):

        st.error(
            "The return window ended on "
            f"{return_result.get('return_deadline')}."
        )

    else:

        st.info(
            "The item is currently within "
            "the date-based return window."
        )

    if st.button(
        "← Back to Agent"
    ):

        st.session_state.active_action = None
        st.rerun()


# ============================================================
# COMPLAINT GENERATOR
# ============================================================

def show_complaint_generator():

    st.subheader(
        "📝 Complaint Generator"
    )

    df = (
        st.session_state.receipt_df
    )

    if df is None or df.empty:

        st.warning(
            "Upload a receipt first."
        )

        return

    item_names = (
        df["Item"]
        .astype(str)
        .tolist()
    )

    default_index = 0

    selected_item = (
        st.session_state.selected_item
    )

    if selected_item in item_names:

        default_index = (
            item_names.index(
                selected_item
            )
        )

    item = st.selectbox(
        "Product",
        item_names,
        index=default_index
    )

    problem = st.text_area(
        "What happened?",
        placeholder=(
            "Example: The mouse "
            "arrived damaged."
        )
    )

    resolution = st.selectbox(
        "Requested resolution",
        [
            "Replacement",
            "Refund",
            "Repair",
            "Other",
        ]
    )

    custom_resolution = ""

    if resolution == "Other":

        custom_resolution = (
            st.text_input(
                "Describe the resolution "
                "you want"
            )
        )

    final_resolution = (
        custom_resolution.strip()
        if resolution == "Other"
        else resolution
    )

    if st.button(
        "Generate Complaint",
        type="primary"
    ):

        result = generate_complaint(
            order_id=(
                st.session_state.order_id
            ),

            item_name=item,

            purchase_date=(
                st.session_state.purchase_date
            ),

            problem=problem.strip(),

            requested_resolution=(
                final_resolution
            ),
        )

        if result.get(
            "success"
        ):

            st.success(
                "✅ Complaint draft generated."
            )

            st.text_area(
                "Complaint",
                result["complaint"],
                height=280
            )

        else:

            st.warning(
                result.get(
                    "message",
                    "More information "
                    "is required."
                )
            )

    if st.button(
        "← Back to Agent"
    ):

        st.session_state.active_action = None
        st.rerun()


# ============================================================
# WARRANTY
# ============================================================

def show_warranty():

    st.subheader(
        "🛡️ Warranty"
    )

    st.warning(
        "The supplied policy document does "
        "not specify a warranty duration. "
        "A warranty deadline cannot be "
        "calculated without verified "
        "warranty terms."
    )

    if st.button(
        "← Back to Agent"
    ):

        st.session_state.active_action = None
        st.rerun()


# ============================================================
# AGENT PAGE
# ============================================================

def show_agent_page(
    language
):

    st.markdown(
    """
    <div style="
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
        background: linear-gradient(135deg, #1f4e78, #2e75b6);
        color: white;
    ">
        <h1 style="margin:0;">
            🛍️ Smart Order & Warranty Agent
        </h1>
        <p style="margin:8px 0 0 0; font-size:17px;">
            Your AI-powered assistant for returns, warranties
            and order support
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

    st.markdown(
    """
    ### 📄 Upload Your Receipt
    Upload a receipt or order confirmation to automatically
    check **return deadlines, warranty coverage and order details**.
    """
)
    uploaded_file = st.file_uploader(
        "Choose a receipt image",
        type=[
            "png",
            "jpg",
            "jpeg",
        ]
    )

    if uploaded_file is not None:

        new_receipt = (
            not st.session_state
            .receipt_processed
            or
            st.session_state
            .receipt_name
            != uploaded_file.name
        )

        if new_receipt:

            with st.spinner(
                "Reading receipt and "
                "checking return windows..."
            ):

                success = (
                    process_receipt(
                        uploaded_file
                    )
                )

            if success:

                st.success(
                    "Receipt processed "
                    "successfully."
                )

                st.rerun()

    if not (
        st.session_state
        .receipt_processed
    ):

        st.markdown(
    """
    <div style="
        text-align:center;
        padding:30px;
        border:2px dashed #d0d7de;
        border-radius:14px;
        margin-top:15px;
    ">
        <div style="font-size:40px;">🧾</div>
        <h3>No receipt uploaded yet</h3>
        <p style="color:#6b7280;">
            Upload your receipt above to instantly check
            return deadlines, warranty coverage and order details.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

        return

    show_purchase_summary()

    st.divider()

    action = (
        st.session_state.active_action
    )

    if action == "return":

        show_return_workflow()
        return

    if action == "receipt":

        show_receipt()
        return

    if action == "complaint":

        show_complaint_generator()
        return

    if action == "ineligible":

        show_ineligibility()
        return

    if action in {
        "warranty",
        "warranty_terms",
    }:

        show_warranty()
        return

    show_chat(
        language
    )


# ============================================================
# ORDER STATUS
# ============================================================

def show_order_status_page():

    st.title(
        "🚚 Order Status"
    )

    order_id = st.text_input(
        "Enter Order ID",
        placeholder="Example: TM10045"
    )

    if st.button(
        "Check Order",
        type="primary"
    ):

        if not order_id.strip():

            st.warning(
                "Please enter an order ID."
            )

            return

        result = (
            lookup_order_status.invoke(
                {
                    "order_id":
                        order_id.strip()
                }
            )
        )

        if result.get(
            "found"
        ):

            st.success(
                f"Order "
                f"{result['order_id']} "
                f"found."
            )

            col1, col2 = (
                st.columns(2)
            )

            with col1:

                st.write(
                    "**Status:**",
                    result.get(
                        "status"
                    )
                    or "Not available"
                )

                st.write(
                    "**Carrier:**",
                    result.get(
                        "carrier"
                    )
                    or "Not available"
                )

                st.write(
                    "**Ship Date:**",
                    result.get(
                        "ship_date"
                    )
                    or "Not available"
                )

            with col2:

                st.write(
                    "**Tracking Number:**",
                    result.get(
                        "tracking_number"
                    )
                    or "Not available"
                )

                st.write(
                    "**Estimated Delivery:**",
                    result.get(
                        "estimated_delivery"
                    )
                    or "Not available"
                )

        else:

            st.error(
                result.get(
                    "message",
                    "Order not found."
                )
            )


# ============================================================
# PROFILE
# ============================================================

def show_profile_page():

    st.title(
        "👤 Profile"
    )

    st.write(
        "**Username:**",
        st.session_state.username
    )

    if (
        st.session_state
        .receipt_processed
    ):

        st.write(
            "**Current Order:**",
            st.session_state.order_id
            or "Not found"
        )

        st.write(
            "**Purchase Date:**",
            st.session_state.purchase_date
        )

        df = (
            st.session_state
            .receipt_df
        )

        if df is not None:

            st.write(
                "**Items in current receipt:**",
                len(df)
            )

    else:

        st.info(
            "No receipt is currently loaded."
        )

    st.divider()

    # ========================================================
    # PASSWORD RECOVERY SETUP
    # ========================================================

    st.subheader(
        "🔐 Password Recovery"
    )

    users = load_users()

    current_username = (
        st.session_state.username
    )

    current_user = users.get(
        current_username
    )

    recovery_questions = [
        "What is your favorite city?",
        "What was the name of your first school?",
        "What is your favorite food?",
    ]

    # ========================================================
    # OLD ACCOUNT FORMAT
    # ========================================================

    if isinstance(
        current_user,
        str
    ):

        st.info(
            "Password recovery is not "
            "configured for this account."
        )

        question = st.selectbox(
            "Choose Security Question",
            recovery_questions,
            key="profile_security_question"
        )

        answer = st.text_input(
            "Recovery Answer",
            type="password",
            key="profile_recovery_answer"
        )

        if st.button(
            "Set Up Password Recovery",
            type="primary"
        ):

            if not answer.strip():

                st.warning(
                    "Please enter a "
                    "recovery answer."
                )

            else:

                # Keep existing password
                existing_password = (
                    current_user
                )

                users[
                    current_username
                ] = {

                    "password":
                        existing_password,

                    "security_question":
                        question,

                    "recovery_answer":
                        answer
                        .strip()
                        .casefold(),
                }

                save_users(
                    users
                )

                st.success(
                    "✅ Password recovery "
                    "configured."
                )

                st.rerun()

    # ========================================================
    # NEW ACCOUNT FORMAT
    # ========================================================

    elif isinstance(
        current_user,
        dict
    ):

        question = (
            current_user.get(
                "security_question"
            )
        )

        if question:

            st.success(
                "✅ Password recovery "
                "is configured."
            )

            st.write(
                "**Security Question:**",
                question
            )

        else:

            st.warning(
                "Password recovery "
                "is not configured."
            )

    st.divider()

    # ========================================================
    # CLEAR RECEIPT
    # ========================================================

    if st.button(
        "Clear Current Receipt"
    ):

        reset_receipt_state()

        st.success(
            "Current receipt cleared."
        )

        st.rerun()


# ============================================================
# MAIN
# ============================================================

def main():

    if not (
        st.session_state.logged_in
    ):

        show_login()
        return

    page, language = (
        show_sidebar()
    )

    if page == "Agent":

        show_agent_page(
            language
        )

    elif page == "Order Status":

        show_order_status_page()

    elif page == "Profile":

        show_profile_page()


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    main()