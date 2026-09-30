from datetime import datetime, timedelta
from langchain_core.tools import tool
import pandas as pd


# ============================================================
# POLICY DATA
# These values match policy.txt
# ============================================================

RETURN_POLICY = {
    "electronics": 30,
    "clothing": 15,
    "accessories": 10
}

# The supplied policy.txt does not specify warranty durations.
# Warranty duration is therefore NOT invented here.
WARRANTY_POLICY = {}


# ============================================================
# RETURN WINDOW CALCULATOR TOOL
# ============================================================

@tool
def calculate_return_window(input_str: str) -> dict:
    """
    Deterministically calculates an item's return window.

    Input format:
    item_name|purchase_date|category

    Example:
    laptop|2026-09-01|electronics

    Calculation:
    Return deadline = purchase date + policy days.

    An item is flagged as expiring soon when fewer than
    7 days remain in the return window.

    Return policy values come from the supplied store policy.
    The tool does not invent a policy for unknown categories.
    """

    try:
        parts = input_str.split("|")

        if len(parts) != 3:
            return {
                "success": False,
                "error": (
                    "Invalid input. Expected format: "
                    "item_name|YYYY-MM-DD|category"
                )
            }

        item_name = parts[0].strip()
        purchase_date = parts[1].strip()
        category = parts[2].strip().lower()

        # ----------------------------------------------------
        # Validate policy category
        # ----------------------------------------------------

        if category not in RETURN_POLICY:
            return {
                "success": False,
                "item": item_name,
                "category": category,
                "policy_found": False,
                "error": (
                    f"No return policy was found for "
                    f"category '{category}'."
                )
            }

        return_days = RETURN_POLICY[category]

        # ----------------------------------------------------
        # Parse purchase date
        # ----------------------------------------------------

        purchase = datetime.strptime(
            purchase_date,
            "%Y-%m-%d"
        ).date()

        today = datetime.today().date()

        # ----------------------------------------------------
        # Deterministic calculation
        # ----------------------------------------------------

        return_deadline = purchase + timedelta(
            days=return_days
        )

        days_remaining = (
            return_deadline - today
        ).days

        expired = days_remaining <= 0

        # Requirement: FEWER than 7 days
        expiring_soon = (
            0 < days_remaining < 7
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {
            "success": True,
            "policy_found": True,

            "item": item_name,
            "category": category,
            "purchase_date": str(purchase),

            "return_policy_days": return_days,
            "return_deadline": str(return_deadline),
            "return_days_remaining": days_remaining,

            # Backward-compatible names
            "days_remaining": days_remaining,
            "expired": expired,

            "return_expired": expired,
            "return_eligible_by_date": not expired,
            "expiring_soon": expiring_soon,

            # Policy condition from policy.txt
            "return_condition": (
                "Item must be unused and in original packaging."
            ),

            # We cannot determine final eligibility solely
            # from dates because condition must also be met.
            "eligibility_note": (
                "Date eligibility is satisfied only if the item "
                "also meets the store policy condition."
            )
        }

    except ValueError:
        return {
            "success": False,
            "error": (
                "Invalid purchase date. "
                "Expected YYYY-MM-DD."
            )
        }

    except Exception as error:
        return {
            "success": False,
            "error": (
                "Unable to calculate return window: "
                f"{str(error)}"
            )
        }


# ============================================================
# WARRANTY INFORMATION TOOL
# ============================================================

@tool
def lookup_warranty_policy(category: str) -> dict:
    """
    Looks up warranty information available in the supplied
    policy data.

    The supplied policy document currently does not specify
    warranty durations, so this tool must not invent one.
    """

    category = category.strip().lower()

    if category in WARRANTY_POLICY:
        return {
            "found": True,
            "category": category,
            "warranty_days": WARRANTY_POLICY[category]
        }

    return {
        "found": False,
        "category": category,
        "message": (
            "Warranty duration is not specified in the "
            "available policy document."
        )
    }


# ============================================================
# ORDER STATUS LOOKUP TOOL
# ============================================================

@tool
def lookup_order_status(order_id: str) -> dict:
    """
    Looks up an order in data/orders.csv.

    If the order exists, verified order information is returned.

    If the order does not exist, a safe not-found response is
    returned. The tool never fabricates order information.
    """

    try:
        if not order_id or not order_id.strip():
            return {
                "found": False,
                "message": "Please provide an order ID."
            }

        requested_order_id = order_id.strip()

        # ----------------------------------------------------
        # Read mock database
        # ----------------------------------------------------

        df = pd.read_csv("data/orders.csv")

        required_columns = [
            "order_id",
            "status",
            "ship_date",
            "carrier",
            "tracking_number",
            "estimated_delivery"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            return {
                "found": False,
                "message": (
                    "Order database is missing columns: "
                    + ", ".join(missing_columns)
                )
            }

        # ----------------------------------------------------
        # Normalize order IDs
        # ----------------------------------------------------

        df["order_id"] = (
            df["order_id"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        normalized_order_id = (
            requested_order_id
            .lower()
        )

        result = df[
            df["order_id"] == normalized_order_id
        ]

        if result.empty:
            return {
                "found": False,
                "order_id": requested_order_id,
                "message": (
                    f"Order {requested_order_id} was not found."
                )
            }

        row = result.iloc[0]

        # ----------------------------------------------------
        # Use verified stored status
        # ----------------------------------------------------

        status = str(row["status"])

        def safe_value(value):
            if pd.isna(value):
                return None
            return str(value)

        return {
            "found": True,
            "order_id": requested_order_id,
            "status": status,
            "ship_date": safe_value(
                row["ship_date"]
            ),
            "carrier": safe_value(
                row["carrier"]
            ),
            "tracking_number": safe_value(
                row["tracking_number"]
            ),
            "estimated_delivery": safe_value(
                row["estimated_delivery"]
            )
        }

    except FileNotFoundError:
        return {
            "found": False,
            "message": (
                "Order database was not found at "
                "data/orders.csv."
            )
        }

    except Exception as error:
        return {
            "found": False,
            "message": (
                "Unable to read the order database: "
                f"{str(error)}"
            )
        }


# ============================================================
# RETURN REQUEST HELPER
# ============================================================

def create_return_request(
    order_id,
    item_name,
    reason
):
    """
    Creates a local return-request object.

    This does not claim that a real retailer return has been
    submitted. It prepares the request for the demo workflow.
    """

    return {
        "type": "return",
        "order_id": order_id,
        "item": item_name,
        "reason": reason,
        "status": "Prepared",
        "created_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }


# ============================================================
# WARRANTY CLAIM HELPER
# ============================================================

def create_warranty_claim(
    order_id,
    item_name,
    problem
):
    """
    Creates a local warranty-claim draft.

    It does not claim that the warranty provider accepted
    or received the claim.
    """

    return {
        "type": "warranty",
        "order_id": order_id,
        "item": item_name,
        "problem": problem,
        "status": "Draft",
        "created_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }


# ============================================================
# COMPLAINT GENERATOR
# ============================================================

def generate_complaint(
    order_id,
    item_name,
    purchase_date,
    problem,
    requested_resolution
):
    """
    Generates a complaint using only supplied/verified data.
    Missing information is not invented.
    """

    missing = []

    if not order_id:
        missing.append("order number")

    if not item_name:
        missing.append("product")

    if not purchase_date:
        missing.append("purchase date")

    if not problem:
        missing.append("problem")

    if not requested_resolution:
        missing.append("requested resolution")

    if missing:
        return {
            "success": False,
            "missing": missing,
            "message": (
                "More information is required: "
                + ", ".join(missing)
            )
        }

    complaint = f"""Customer Complaint

Order Number: {order_id}
Product: {item_name}
Purchase Date: {purchase_date}
Problem: {problem}
Requested Resolution: {requested_resolution}

I am requesting assistance with the issue described above.
Please review my order and advise on the appropriate next steps.
"""

    return {
        "success": True,
        "complaint": complaint
    }