import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

from receipt_parser import extract_text_from_image, parse_receipt_text
from tools import calculate_return_window, lookup_order_status

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide")

# ---------------- LIGHT UI ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f8fafc, #eef2f7);
    color: #1e293b;
}
.block-container {padding: 2rem;}
div[data-testid="stMetric"], div[data-testid="stDataFrame"], div.stPlotlyChart, div.stAlert {
    background: rgba(255,255,255,0.75);
    border-radius: 12px;
    padding: 15px;
    border: 1px solid rgba(0,0,0,0.05);
}
button {border-radius: 8px !important; background: white !important;}
[data-testid="stSidebar"] {background: rgba(255,255,255,0.9);}
input, textarea {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
    border-radius: 8px !important;
    padding: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- POLICY ----------------
POLICY = {
    "electronics": 30,
    "clothing": 15,
    "accessories": 10,
    "default": 7
}

# ---------------- SERVICE LINKS ----------------
SERVICE_CENTERS = {
    "dell": "https://www.dell.com/support/home",
    "hp": "https://support.hp.com",
    "apple": "https://support.apple.com",
    "lenovo": "https://support.lenovo.com",
    "logitech": "https://support.logi.com",
    "default": "https://www.google.com/search?q=service+center+near+me"
}

def get_service_link(item):
    item = item.lower()
    for brand in SERVICE_CENTERS:
        if brand in item:
            return SERVICE_CENTERS[brand]
    return SERVICE_CENTERS["default"]

# ---------------- USERS ----------------
USER_FILE = "users.json"

def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    return json.load(open(USER_FILE))

def save_users(users):
    json.dump(users, open(USER_FILE, "w"))

users = load_users()

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "data" not in st.session_state:
    st.session_state.data = {}

# ---------------- AUTH ----------------
if not st.session_state.logged_in:

    st.title("🔐 Login / Signup")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if u in users and users[u] == p:
                st.session_state.logged_in = True
                st.session_state.username = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        nu = st.text_input("New Username")
        np = st.text_input("New Password", type="password")

        if st.button("Signup"):
            if nu in users:
                st.error("User exists")
            else:
                users[nu] = np
                save_users(users)
                st.success("Account created")

    st.stop()

# ---------------- USER DATA ----------------
username = st.session_state.username

if username not in st.session_state.data:
    st.session_state.data[username] = {"history": [], "chat": []}

store = st.session_state.data[username]

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio("Navigation", ["Dashboard", "Profile"])
st.sidebar.write(f"👤 {username}")

# ---------------- HISTORY ----------------
selected = None

if store["history"]:
    st.sidebar.markdown("## 📜 Orders")

    labels = [
        f"{h['order_id']} | {h['date']} | ${h['total']}"
        for h in store["history"]
    ]

    selected_label = st.sidebar.selectbox("Select Order", labels)

    selected = next(
        h for h in store["history"]
        if f"{h['order_id']} | {h['date']} | ${h['total']}" == selected_label
    )

    if st.sidebar.button("🗑 Clear History"):
        store["history"] = []
        st.rerun()

# ---------------- PROFILE ----------------
if page == "Profile":

    st.title("👤 Profile")

    history = store["history"]
    spend = sum(h["df"]["Price"].sum() for h in history) if history else 0

    st.metric("Receipts", len(history))
    st.metric("Total Spend", f"${round(spend,2)}")

    if history:
        df_all = pd.concat([h["df"] for h in history])

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(df_all, names="Category", values="Price"), use_container_width=True)
        with col2:
            st.plotly_chart(px.bar(df_all, x="Item", y="Price"), use_container_width=True)

    st.subheader("Change Password")

    old = st.text_input("Current Password", type="password")
    new = st.text_input("New Password", type="password")

    if st.button("Update Password"):
        if users.get(username) != old:
            st.error("Wrong password")
        else:
            users[username] = new
            save_users(users)
            st.success("Password updated")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.stop()

# ---------------- MAIN ----------------
st.title("🛒 Smart Order & Warranty Assistant")

left, right = st.columns([2,1])

df = None
order_id = None

# ---------------- LEFT ----------------
with left:

    file = st.file_uploader("Upload Receipt", ["txt","png","jpg"])

    if file:
        if file.type.startswith("image"):
            text = extract_text_from_image(file)
        else:
            text = file.read().decode()

        order_id, date, items = parse_receipt_text(text)

        rows = []
        for item in items:
            res = calculate_return_window.invoke({
                "input_str": f"{item['name']}|{date}|{item['category']}"
            })

            rows.append({
                "Item": item["name"],
                "Category": item["category"],
                "Price": float(item["price"]),
                "Return": res["return_deadline"],
                "Warranty": res["warranty_deadline"],
                "Days": res["days_remaining"],
                "Status": "Expired" if res["expired"]
                          else "Expiring Soon" if res["expiring_soon"]
                          else "Safe"
            })

        df = pd.DataFrame(rows)

        if not any(h["order_id"] == order_id for h in store["history"]):
            store["history"].append({
                "order_id": order_id,
                "df": df,
                "text": text,
                "date": date,
                "total": round(df["Price"].sum(), 2)
            })

    elif selected:
        df = selected["df"]
        order_id = selected["order_id"]

    if df is not None:

        c1, c2, c3 = st.columns(3)
        c1.metric("Items", len(df))
        c2.metric("Expiring Soon", len(df[df["Status"]=="Expiring Soon"]))
        c3.metric("Total Spend", f"${df['Price'].sum()}")

        st.dataframe(df, use_container_width=True)

        st.subheader("Alerts")

        for _, r in df.iterrows():
            if r["Status"] == "Expired":
                st.error(f"{r['Item']} expired")
            elif r["Status"] == "Expiring Soon":
                st.warning(f"{r['Item']} → {r['Days']} days left")

# ---------------- RIGHT ----------------
with right:

    st.subheader("AI Assistant")

    q = st.text_input("Ask about your receipt")

    if q and df is not None:

        q_lower = q.lower()
        match = None

        for _, r in df.iterrows():
            if any(word in r["Item"].lower() for word in q_lower.split()):
                match = r
                break

        if "policy" in q_lower:

            if match is not None:
                category = match["Category"].lower()
                days_allowed = POLICY.get(category, POLICY["default"])

                ans = f"""
📦 Return Policy for {match['Item']}:

- Category: {category.title()}
- Allowed return period: {days_allowed} days
- Return deadline: {match['Return']}
"""
            else:
                ans = """
📦 General Return Policy:
- Electronics → 30 days
- Clothing → 15 days
- Accessories → 10 days
"""

        elif "return" in q_lower and match is not None:

            link = get_service_link(match["Item"])

            if match["Status"] == "Expired":
                if match["Days"] >= 0:
                    ans = f"❌ Return expired. Warranty till {match['Warranty']} → {link}"
                else:
                    ans = f"❌ Return & warranty expired. Repair → {link}"
            else:
                ans = f"✅ Return before {match['Return']} ({match['Days']} days left)"

        elif "warranty" in q_lower:

            if match is not None:
                status = "Active" if match["Days"] >= 0 else "Expired"
                ans = f"🛠 Warranty till {match['Warranty']} ({status})"
            else:
                ans = "Specify item for warranty details"

        elif "status" in q_lower:
            res = lookup_order_status.invoke({"order_id": order_id})
            ans = res["status"] if res["found"] else "Not found"

        else:
            ans = "Ask about return policy, warranty, or order status"

        store["chat"].append((q, ans))

    for q,a in store["chat"][::-1]:
        st.write("🧑", q)
        st.write("🤖", a)
        st.write("---")

    st.subheader("Check Order")

    oid = st.text_input("Order ID")

    if st.button("Check Status"):
        res = lookup_order_status.invoke({"order_id": oid})
        if res["found"]:
            st.success(res["status"])
        else:
            st.error("Not found")