# 🛍️ Smart Order & Warranty Agent

An AI-powered shopping support agent that analyzes uploaded receipts or order confirmations, calculates return and warranty windows, provides proactive deadline alerts, checks order status, and answers shopper questions using RAG and deterministic Python tools.

## 🎯 Problem Statement

Shoppers often struggle to track return deadlines, understand warranty coverage, locate order information, and interpret store policies.

The Smart Order & Warranty Agent brings these tasks together in one application. A shopper uploads a receipt, and the system extracts purchase information, calculates important deadlines, retrieves relevant policy information, and provides grounded answers through a conversational interface.

## ✨ Key Features

- 📄 Receipt and order confirmation upload
- 🔍 Automatic extraction of order and product information
- ↩️ Deterministic return-window calculation
- 🛡️ Warranty-window calculation
- ⚠️ Proactive alerts for approaching return deadlines
- 📦 Mock order-status lookup
- 💬 Natural-language shopping assistant
- 🧠 Retrieval-Augmented Generation (RAG)
- 🔎 Semantic retrieval using vector embeddings
- 🗄️ Chroma vector database
- 🛠️ LangChain tool calling
- 🤖 Local LLM using Ollama / Llama
- 🌐 English, Telugu and Hindi response support
- 📝 Complaint generation
- 🔐 User authentication
- 📊 Structured purchase summary dashboard
- ❌ Graceful handling of unavailable information

## 🧠 How It Works

The application follows this flow:

1. The shopper signs in.
2. The shopper uploads a receipt or order confirmation.
3. Receipt information is extracted and parsed.
4. Individual purchased items are converted into documents.
5. Policy documents are split into relevant chunks.
6. Embeddings are generated for the documents.
7. Documents are stored in ChromaDB.
8. Relevant information is retrieved based on the shopper's question.
9. Deterministic Python tools calculate return and warranty windows.
10. Order status is retrieved from the mock order database.
11. The LLM receives only verified evidence and generates a grounded response.

## 🏗️ Architecture

```text
Shopper
   │
   ▼
Streamlit UI
   │
   ├── Receipt Upload
   │
   ▼
Receipt Parser / OCR
   │
   ├── Order ID
   ├── Purchase Date
   └── Purchased Items
   │
   ▼
RAG Pipeline
   │
   ├── Document Chunking
   ├── HuggingFace Embeddings
   ├── ChromaDB
   └── Retriever
   │
   ▼
Smart Agent
   │
   ├── Return Window Tool
   ├── Warranty Window Tool
   ├── Warranty Policy Tool
   └── Order Status Tool
   │
   ▼
Ollama / Llama LLM
   │
   ▼
Grounded Shopper Response
```

## 🛠️ Technology Stack

- **Python** – Core application logic
- **Streamlit** – Web user interface
- **LangChain** – Agent and tool integration
- **Ollama** – Local LLM execution
- **Llama 3.2** – Language model
- **ChromaDB** – Vector database
- **HuggingFace Embeddings** – Semantic embeddings
- **Pandas** – Receipt and order data processing
- **OCR / Image Processing** – Receipt text extraction
- **CSV** – Mock order-status database

## 🔧 Agent Tools

The agent uses deterministic Python tools instead of relying on the LLM to guess important information.

### Return Window Calculator

Calculates the return deadline and remaining return days based on:

- Product category
- Purchase date
- Return policy

### Warranty Window Calculator

Calculates warranty coverage and warranty expiration dates using defined warranty rules.

### Warranty Policy Lookup

Retrieves verified warranty-policy information.

### Order Status Lookup

Looks up order information from the mock order database rather than allowing the LLM to invent tracking or delivery information.

## 🧠 RAG Implementation

The project uses Retrieval-Augmented Generation to ground AI responses.

Receipt items are stored as individual documents, while policy information is split into separate chunks.

The RAG pipeline includes:

```text
Documents
   ↓
Text Splitting
   ↓
Embeddings
   ↓
Chroma Vector Store
   ↓
Retriever
   ↓
Relevant Evidence
   ↓
LLM Response
```

This helps the agent answer questions using relevant receipt and policy information.

## 🛡️ Hallucination Prevention

The system is designed to prevent the AI from inventing critical shopper information.

The LLM is instructed not to invent:

- Products
- Purchase dates
- Return periods
- Return deadlines
- Warranty coverage
- Order status
- Tracking information
- Carrier information
- Delivery information

Return and warranty calculations are handled by deterministic Python logic.

If verified information is unavailable, the agent informs the shopper instead of fabricating an answer.

## 💬 Example Questions

Users can ask questions such as:

```text
Can I return my Blue Jacket?

When does my return window expire?

Is my Wireless Mouse still under warranty?

What is the status of my order?

What warranty coverage does my Backpack have?

My product is defective. Is it covered by warranty?
```

The agent can also support questions that require information from multiple parts of the system.

## ⚠️ Proactive Alerts

The application automatically checks return deadlines after processing the receipt.

For example:

```text
⚠️ Blue Jacket return window ends in 5 days.

❌ Black Backpack return window has expired.
```

This allows the system to proactively help shoppers instead of waiting for them to ask.

## 🌐 Multilingual Support

The Smart Order & Warranty Agent supports responses in:

- English
- తెలుగు (Telugu)
- हिन्दी (Hindi)

Product names, order IDs, dates, prices and other important values are preserved while explanatory responses can be generated in the selected language.

## 📊 User Experience

The Streamlit interface includes:

- Professional application header
- Receipt upload workflow
- Purchase summary dashboard
- Return status indicators
- Warranty status indicators
- Proactive deadline warnings
- Conversational AI interface
- Loading indicators
- Success and error messages
- Empty-state guidance
- Dynamic shopper actions

## 📁 Project Structure

```text
Smart-Order-Warranty-Agent/
│
├── app.py
├── agent.py
├── assistant_service.py
├── llm.py
├── rag.py
├── receipt_parser.py
├── tools.py
├── users.json
│
├── data/
│   └── orders.csv
│
├── policies/
│   └── return_policy.txt
│
└── README.md
```

> Project structure may vary slightly depending on the local setup.

## 🚀 Running the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Smart-Order-Warranty-Agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama

Install Ollama and make sure it is running locally.

### 4. Download the Llama model

```bash
ollama pull llama3.2
```

### 5. Run the application

```bash
streamlit run app.py
```

Open the Streamlit URL displayed in the terminal.

## 🧪 Example Demo Flow

For a demonstration:

1. Sign in to the application.
2. Upload a sample receipt.
3. View the automatically generated Purchase Summary.
4. Check return and warranty status.
5. Observe proactive return-deadline alerts.
6. Ask the Smart Agent a return question.
7. Ask a warranty question.
8. Ask for order status.
9. Change the response language.
10. Demonstrate how the system avoids unsupported answers.

## 🔮 Future Improvements

Future versions could include:

- Integration with real retailer APIs
- Real-time shipment tracking
- Email/SMS return reminders
- Support for additional receipt formats
- Dynamic retailer-policy extraction
- Improved authentication and password security
- Cloud deployment
- More languages
- Faster local LLM inference
- Automated return and warranty claim submission

This project demonstrates how **RAG, LLM agents, deterministic tools, vector databases and a user-friendly interface** can be combined to solve a practical e-commerce customer-support problem.

The goal is not just to create a chatbot, but to create an intelligent shopping agent that provides reliable, actionable and grounded assistance.
