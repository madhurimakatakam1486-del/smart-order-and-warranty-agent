import logging
import re
from decimal import Decimal, InvalidOperation

from PIL import Image
import pytesseract


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

# Update this path only if Tesseract is installed somewhere else.
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# SUPPORTED POLICY CATEGORIES
# ============================================================

SUPPORTED_CATEGORIES = {
    "electronics",
    "clothing",
    "accessories",
}


# ============================================================
# CATEGORY DETECTION
# ============================================================

def detect_category(item_name):
    """
    Detects a supported return-policy category from an item name.

    Supported categories:
    - electronics
    - clothing
    - accessories

    If the category cannot be determined safely, returns
    'unknown'. This prevents the application from inventing
    a return policy for an unsupported category.
    """

    if not item_name:
        return "unknown"

    name = str(item_name).lower().strip()

    electronics_keywords = [
        "laptop",
        "mouse",
        "keyboard",
        "monitor",
        "headphone",
        "headphones",
        "earphone",
        "earphones",
        "earbuds",
        "tablet",
        "computer",
        "charger",
        "speaker",
        "webcam",
    ]

    clothing_keywords = [
        "jacket",
        "shirt",
        "t-shirt",
        "tshirt",
        "jeans",
        "trouser",
        "trousers",
        "dress",
        "coat",
        "sweater",
        "hoodie",
        "pants",
        "kurta",
        "top",
    ]

    accessories_keywords = [
        "bag",
        "handbag",
        "wallet",
        "belt",
        "scarf",
        "cap",
        "hat",
        "backpack",
        "purse",
    ]

    if any(keyword in name for keyword in electronics_keywords):
        return "electronics"

    if any(keyword in name for keyword in clothing_keywords):
        return "clothing"

    if any(keyword in name for keyword in accessories_keywords):
        return "accessories"

    return "unknown"


# ============================================================
# OCR
# ============================================================

def extract_text_from_image(uploaded_file):
    """
    Extracts text from an uploaded receipt image using Tesseract OCR.

    Returns:
        str: Extracted receipt text.
    """

    try:
        uploaded_file.seek(0)

        image = Image.open(uploaded_file)

        text = pytesseract.image_to_string(image)

        if not text or not text.strip():
            logger.warning("OCR returned empty text.")
            return ""

        return text.strip()

    except Exception as error:
        logger.error("OCR failed: %s", error)
        return ""


# ============================================================
# PRICE PARSING
# ============================================================

def parse_price(value):
    """
    Converts a price string into a float.

    Examples:
        $49.99  -> 49.99
        ₹999.00 -> 999.00
        1,299.50 -> 1299.50
    """

    if value is None:
        return None

    cleaned = re.sub(
        r"[^\d.\-]",
        "",
        str(value).replace(",", "")
    )

    if not cleaned:
        return None

    try:
        return float(Decimal(cleaned))

    except (InvalidOperation, ValueError):
        return None


# ============================================================
# PURCHASE DATE EXTRACTION
# ============================================================

def extract_purchase_date(text):
    """
    Extracts a purchase date from receipt text.

    The function normalizes supported dates to YYYY-MM-DD.
    """

    if not text:
        return None

    patterns = [
        # YYYY-MM-DD
        (
            r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b",
            "ymd"
        ),

        # DD-MM-YYYY or DD/MM/YYYY
        (
            r"\b(\d{1,2})[-/](\d{1,2})[-/](20\d{2})\b",
            "dmy"
        ),
    ]

    for pattern, date_type in patterns:

        match = re.search(pattern, text)

        if not match:
            continue

        try:
            if date_type == "ymd":
                year = int(match.group(1))
                month = int(match.group(2))
                day = int(match.group(3))

            else:
                day = int(match.group(1))
                month = int(match.group(2))
                year = int(match.group(3))

            # Validate date
            from datetime import date

            parsed_date = date(
                year,
                month,
                day
            )

            return parsed_date.strftime("%Y-%m-%d")

        except ValueError:
            continue

    return None


# ============================================================
# ORDER ID EXTRACTION
# ============================================================

def extract_order_id(text):
    """
    Extracts an order ID from common receipt formats.

    Examples:
        Order ID: TM10045
        Order Number: TM10045
        Order No: TM10045
        Order #: TM10045
    """

    if not text:
        return None

    patterns = [
        r"(?i)\border\s+id\s*[:#-]?\s*([A-Z0-9_-]+)",
        r"(?i)\border\s+number\s*[:#-]?\s*([A-Z0-9_-]+)",
        r"(?i)\border\s+no\.?\s*[:#-]?\s*([A-Z0-9_-]+)",
        r"(?i)\border\s*#\s*([A-Z0-9_-]+)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text
        )

        if match:

            order_id = (
                match.group(1).strip()
            )

            return order_id

    return None

# ============================================================
# ITEM LINE FILTER
# ============================================================

def should_skip_line(line):
    """
    Returns True for receipt metadata lines that should not
    be interpreted as purchased products.
    """

    if not line:
        return True

    lower_line = line.lower().strip()

    skip_terms = [
        "order id",
        "order number",
        "order no",
        "purchase date",
        "date:",
        "subtotal",
        "grand total",
        "total:",
        "tax:",
        "shipping:",
        "payment",
        "warranty",
        "return policy",
        "return:",
    ]

    return any(
        term in lower_line
        for term in skip_terms
    )


# ============================================================
# ITEM PARSING
# ============================================================

def parse_items(text):
    """
    Parses purchased items from receipt text.

    Supported formats include:

        Wireless Mouse | electronics | 29.99

        Jacket | clothing | 79.99

        Backpack | accessories | 45.00

    It also attempts to parse lines where the price appears
    at the end:

        Wireless Mouse $29.99

    Returns:
        list[dict]
    """

    items = []

    if not text:
        return items

    lines = text.splitlines()

    for raw_line in lines:

        line = raw_line.strip()

        if should_skip_line(line):
            continue

        # ----------------------------------------------------
        # FORMAT 1:
        # Item | category | price
        # ----------------------------------------------------

        if "|" in line:

            parts = [
                part.strip()
                for part in line.split("|")
            ]

            if len(parts) >= 3:

                name = parts[0]
                supplied_category = parts[1].lower()
                price = parse_price(parts[-1])

                if not name or price is None:
                    continue

                if supplied_category in SUPPORTED_CATEGORIES:
                    category = supplied_category
                else:
                    category = detect_category(name)

                items.append({
                    "name": name,
                    "category": category,
                    "price": price,
                })

                continue

        # ----------------------------------------------------
        # FORMAT 2:
        # Item name followed by price
        #
        # Examples:
        # Jacket $79.99
        # Mouse 29.99
        # ----------------------------------------------------

        price_match = re.search(
            r"(?:₹|\$|€|£)?\s*"
            r"(\d[\d,]*\.\d{2})\s*$",
            line
        )

        if not price_match:
            continue

        price = parse_price(
            price_match.group(1)
        )

        if price is None:
            continue

        name = line[
            :price_match.start()
        ].strip(" |-:\t")

        if not name:
            continue

        # Avoid treating pure numeric lines as products.
        if not re.search(r"[A-Za-z]", name):
            continue

        category = detect_category(name)

        items.append({
            "name": name,
            "category": category,
            "price": price,
        })

    return items


# ============================================================
# MAIN RECEIPT PARSER
# ============================================================

def parse_receipt(uploaded_file):
    """
    Main receipt parser.

    Extracts:
    - Order ID
    - Purchase date
    - Purchased items
    - Category
    - Price

    Returns:
        (order_id, purchase_date, items)
    """

    if uploaded_file is None:
        return None, None, []

    try:
        # ----------------------------------------------------
        # Extract receipt text
        # ----------------------------------------------------

        text = extract_text_from_image(
            uploaded_file
        )

        if not text:
            logger.warning(
                "No readable text was found in receipt."
            )

            return None, None, []

        # ----------------------------------------------------
        # Extract verified receipt fields
        # ----------------------------------------------------

        order_id = extract_order_id(text)

        purchase_date = extract_purchase_date(
            text
        )

        items = parse_items(text)

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        logger.info(
            "Receipt parsed: order_id=%s, "
            "purchase_date=%s, items=%d",
            order_id,
            purchase_date,
            len(items),
        )

        return (
            order_id,
            purchase_date,
            items
        )

    except Exception as error:

        logger.exception(
            "Unexpected receipt parsing error: %s",
            error
        )

        return None, None, []