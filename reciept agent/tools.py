from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ORDERS_FILE = BASE_DIR / "data" / "orders.csv"


# ============================================================
# VERIFIED SAMPLE POLICY
# ============================================================

RETURN_POLICY = {
    "electronics": 30,
    "clothing": 15,
    "accessories": 10,
}

WARRANTY_POLICY = {
    "electronics": 365,
    "clothing": None,
    "accessories": 180,
}

RETURN_CONDITION = (
    "Item must be unused and in original packaging."
)

WARRANTY_CONDITION = (
    "Warranty claims apply to manufacturing defects only. "
    "Physical damage, misuse, accidental damage, and normal "
    "wear and tear are not covered under warranty."
)


# ============================================================
# RETURN WINDOW CALCULATOR
# ============================================================

@tool
def calculate_return_window(input_str: str) -> dict:
    """
    Deterministically calculates an item's return window.

    Input format:
        item_name|YYYY-MM-DD|category

    Calculation:
        return deadline = purchase date + policy return days

    The return is treated as expired when zero or fewer
    days remain, according to this project's eligibility rule.

    Expiring soon means more than zero but fewer than
    seven days remain.
    """

    try:

        parts = input_str.split("|")

        if len(parts) != 3:

            return {
                "success": False,
                "error": (
                    "Invalid input. Expected format: "
                    "item_name|YYYY-MM-DD|category"
                ),
            }

        item_name = parts[0].strip()
        purchase_date = parts[1].strip()
        category = parts[2].strip().lower()

        # ----------------------------------------------------
        # CHECK POLICY
        # ----------------------------------------------------

        if category not in RETURN_POLICY:

            return {
                "success": False,
                "item": item_name,
                "category": category,
                "policy_found": False,
                "error": (
                    "No return policy was found for "
                    f"category '{category}'."
                ),
            }

        return_days = RETURN_POLICY[
            category
        ]

        purchase = datetime.strptime(
            purchase_date,
            "%Y-%m-%d"
        ).date()

        today = datetime.today().date()

        return_deadline = (
            purchase
            + timedelta(
                days=return_days
            )
        )

        days_remaining = (
            return_deadline
            - today
        ).days

        # ----------------------------------------------------
        # PROJECT ELIGIBILITY RULE
        # ----------------------------------------------------

        expired = (
            days_remaining <= 0
        )

        expiring_soon = (
            0 < days_remaining < 7
        )

        return {
            "success": True,
            "policy_found": True,

            "item": item_name,
            "category": category,

            "purchase_date":
                str(purchase),

            "return_policy_days":
                return_days,

            "return_deadline":
                str(return_deadline),

            "return_days_remaining":
                days_remaining,

            # Compatibility with existing app
            "days_remaining":
                days_remaining,

            "expired":
                expired,

            "return_expired":
                expired,

            "return_eligible_by_date":
                not expired,

            "expiring_soon":
                expiring_soon,

            "return_condition":
                RETURN_CONDITION,

            "eligibility_note": (
                "Date eligibility is satisfied only if "
                "the item also meets the store policy "
                "condition."
            ),
        }

    except ValueError:

        return {
            "success": False,
            "error": (
                "Invalid purchase date. "
                "Expected YYYY-MM-DD."
            ),
        }

    except Exception as error:

        return {
            "success": False,
            "error": (
                "Unable to calculate return window: "
                f"{str(error)}"
            ),
        }


# ============================================================
# WARRANTY WINDOW CALCULATOR
# ============================================================

@tool
def calculate_warranty_window(
    input_str: str
) -> dict:
    """
    Deterministically calculates an item's warranty window.

    Input format:
        item_name|YYYY-MM-DD|category

    Calculation:
        warranty deadline =
        purchase date + warranty policy days

    No warranty duration is invented for categories
    where the policy explicitly says there is no warranty.
    """

    try:

        parts = input_str.split("|")

        if len(parts) != 3:

            return {
                "success": False,
                "error": (
                    "Invalid input. Expected format: "
                    "item_name|YYYY-MM-DD|category"
                ),
            }

        item_name = parts[0].strip()
        purchase_date = parts[1].strip()
        category = parts[2].strip().lower()

        # ----------------------------------------------------
        # UNKNOWN CATEGORY
        # ----------------------------------------------------

        if category not in WARRANTY_POLICY:

            return {
                "success": False,
                "policy_found": False,
                "warranty_available": False,
                "item": item_name,
                "category": category,
                "error": (
                    "No warranty policy was found for "
                    f"category '{category}'."
                ),
            }

        warranty_days = (
            WARRANTY_POLICY[
                category
            ]
        )

        # ----------------------------------------------------
        # EXPLICIT NO WARRANTY
        # ----------------------------------------------------

        if warranty_days is None:

            return {
                "success": True,
                "policy_found": True,
                "warranty_available": False,

                "item": item_name,
                "category": category,

                "purchase_date":
                    purchase_date,

                "warranty_policy_days":
                    None,

                "warranty_deadline":
                    None,

                "warranty_days_remaining":
                    None,

                "warranty_expired":
                    None,

                "under_warranty_by_date":
                    False,

                "message": (
                    "The policy states that this "
                    "category has no warranty."
                ),
            }

        # ----------------------------------------------------
        # CALCULATE WARRANTY
        # ----------------------------------------------------

        purchase = datetime.strptime(
            purchase_date,
            "%Y-%m-%d"
        ).date()

        today = datetime.today().date()

        warranty_deadline = (
            purchase
            + timedelta(
                days=warranty_days
            )
        )

        warranty_days_remaining = (
            warranty_deadline
            - today
        ).days

        warranty_expired = (
            warranty_days_remaining < 0
        )

        under_warranty = (
            not warranty_expired
        )

        return {
            "success": True,
            "policy_found": True,
            "warranty_available": True,

            "item": item_name,
            "category": category,

            "purchase_date":
                str(purchase),

            "warranty_policy_days":
                warranty_days,

            "warranty_deadline":
                str(warranty_deadline),

            "warranty_days_remaining":
                warranty_days_remaining,

            "warranty_expired":
                warranty_expired,

            "under_warranty_by_date":
                under_warranty,

            "warranty_condition":
                WARRANTY_CONDITION,

            "eligibility_note": (
                "Being within the warranty period does "
                "not by itself prove claim eligibility. "
                "The reported issue must satisfy the "
                "warranty conditions."
            ),
        }

    except ValueError:

        return {
            "success": False,
            "error": (
                "Invalid purchase date. "
                "Expected YYYY-MM-DD."
            ),
        }

    except Exception as error:

        return {
            "success": False,
            "error": (
                "Unable to calculate warranty window: "
                f"{str(error)}"
            ),
        }


# ============================================================
# WARRANTY POLICY LOOKUP
# ============================================================

@tool
def lookup_warranty_policy(
    category: str
) -> dict:
    """
    Looks up verified warranty terms for a category.
    """

    category = (
        category.strip().lower()
    )

    if category not in WARRANTY_POLICY:

        return {
            "found": False,
            "category": category,
            "message": (
                "No warranty policy was found "
                "for this category."
            ),
        }

    warranty_days = (
        WARRANTY_POLICY[
            category
        ]
    )

    if warranty_days is None:

        return {
            "found": True,
            "category": category,
            "warranty_available": False,
            "warranty_days": None,
            "message": (
                "The policy states that this "
                "category has no warranty."
            ),
        }

    return {
        "found": True,
        "category": category,
        "warranty_available": True,
        "warranty_days":
            warranty_days,
        "condition":
            WARRANTY_CONDITION,
    }


# ============================================================
# ORDER STATUS LOOKUP
# ============================================================

@tool
def lookup_order_status(
    order_id: str
) -> dict:
    """
    Looks up an order in data/orders.csv.

    The tool returns only information present
    in the mock order database and never
    fabricates an order status.
    """

    try:

        if (
            not order_id
            or not order_id.strip()
        ):

            return {
                "found": False,
                "message": (
                    "Please provide an order ID."
                ),
            }

        requested_order_id = (
            order_id.strip()
        )

        if not ORDERS_FILE.exists():

            return {
                "found": False,
                "message": (
                    "Order database was not found "
                    "at data/orders.csv."
                ),
            }

        df = pd.read_csv(
            ORDERS_FILE
        )

        required_columns = [
            "order_id",
            "status",
            "ship_date",
            "carrier",
            "tracking_number",
            "estimated_delivery",
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
                    "Order database is missing "
                    "columns: "
                    + ", ".join(
                        missing_columns
                    )
                ),
            }

        df["order_id"] = (
            df["order_id"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        result = df[
            df["order_id"]
            ==
            requested_order_id.lower()
        ]

        if result.empty:

            return {
                "found": False,
                "order_id":
                    requested_order_id,
                "message": (
                    f"Order "
                    f"{requested_order_id} "
                    f"was not found."
                ),
            }

        row = result.iloc[0]

        def safe_value(value):

            if pd.isna(value):
                return None

            return str(value)

        return {
            "found": True,

            "order_id":
                requested_order_id,

            "status":
                safe_value(
                    row["status"]
                ),

            "ship_date":
                safe_value(
                    row["ship_date"]
                ),

            "carrier":
                safe_value(
                    row["carrier"]
                ),

            "tracking_number":
                safe_value(
                    row["tracking_number"]
                ),

            "estimated_delivery":
                safe_value(
                    row[
                        "estimated_delivery"
                    ]
                ),
        }

    except Exception as error:

        return {
            "found": False,
            "message": (
                "Unable to read the order database: "
                f"{str(error)}"
            ),
        }


# ============================================================
# RETURN REQUEST
# ============================================================

def create_return_request(
    order_id,
    item_name,
    reason
):
    """
    Prepare a local demonstration return request.
    """

    return {
        "type":
            "return",

        "order_id":
            order_id,

        "item":
            item_name,

        "reason":
            reason,

        "status":
            "Prepared",

        "created_at":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
    }


# ============================================================
# WARRANTY CLAIM
# ============================================================

def create_warranty_claim(
    order_id,
    item_name,
    problem
):
    """
    Prepare a local demonstration warranty claim.
    """

    return {
        "type":
            "warranty",

        "order_id":
            order_id,

        "item":
            item_name,

        "problem":
            problem,

        "status":
            "Draft",

        "created_at":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
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
    Generate a complaint only from verified order data
    and shopper-provided issue/resolution information.
    """

    missing = []

    if not order_id:
        missing.append(
            "order number"
        )

    if not item_name:
        missing.append(
            "product"
        )

    if not purchase_date:
        missing.append(
            "purchase date"
        )

    if not problem:
        missing.append(
            "problem"
        )

    if not requested_resolution:
        missing.append(
            "requested resolution"
        )

    if missing:

        return {
            "success": False,
            "missing": missing,
            "message": (
                "More information is required: "
                + ", ".join(
                    missing
                )
            ),
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
        "complaint": complaint,
    }