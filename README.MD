
# Draftwise — AI-Powered LinkedIn Post Writer & Reviewer

**Write better LinkedIn posts with an AI writer and an AI reviewer working together.**

Draftwise is an AI-powered LinkedIn content creation application built using **Python, Streamlit, LangGraph, OpenAI, Groq, and Tavily**.

It uses a multi-agent workflow in which one AI agent generates LinkedIn posts while another evaluates them against predefined quality criteria. If a draft does not meet the required standards, it is sent back to the writer along with the reviewer's feedback for improvement.

Rather than generating a post once and leaving the user to refine it manually, Draftwise creates an iterative writing and reviewing loop that continues until the post is approved or the maximum number of attempts is reached.

---

## Table of Contents

- [Draftwise — AI-Powered LinkedIn Post Writer \& Reviewer](#draftwise--ai-powered-linkedin-post-writer--reviewer)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [How It Works](#how-it-works)
    - [1. Writer Agent](#1-writer-agent)
    - [2. Conditional Web Research](#2-conditional-web-research)
    - [3. Reviewer Agent](#3-reviewer-agent)
    - [4. Iterative Improvement](#4-iterative-improvement)
  - [Architecture](#architecture)
  - [Tech Stack](#tech-stack)
  - [Project Structure](#project-structure)
    - [File Description](#file-description)
  - [Installation and Setup](#installation-and-setup)
    - [Prerequisites](#prerequisites)
    - [Step 1: Clone the Repository](#step-1-clone-the-repository)
    - [Step 2: Create a Virtual Environment](#step-2-create-a-virtual-environment)
    - [Step 3: Create the Requirements File](#step-3-create-the-requirements-file)
    - [Step 4: Install Dependencies](#step-4-install-dependencies)
    - [Step 5: Run the Application](#step-5-run-the-application)
  - [Running the Application](#running-the-application)
  - [Deployment](#deployment)
    - [Deployment Steps](#deployment-steps)
  - [API Keys and Security](#api-keys-and-security)
  - [Future Improvements](#future-improvements)
  - [Author](#author)
  - [License](#license)

---

## Features

- **AI-Powered Writing:** Generates engaging LinkedIn posts based on user-provided topics and selected writing tones.
- **Multi-Agent Workflow:** Uses separate writer and reviewer agents with distinct responsibilities.
- **Iterative Review Loop:** Automatically sends rejected drafts back to the writer along with actionable feedback.
- **Conditional Web Research:** Uses Tavily Search to retrieve relevant information when a topic requires current context.
- **Multiple Writing Tones:** Choose from Thoughtful, Bold, Story-driven, Data-led, and Friendly.
- **Live Workflow Visualization:** Follow the writer, research, reviewer, and approval stages as they execute.
- **Draft History:** View previous drafts, reviewer feedback, and approval status.
- **Word Count Tracking:** Displays the generated draft's word count against the target of 150–200 words.
- **Copy to Clipboard:** Copy the generated LinkedIn post directly.
- **PDF Export:** Download the generated draft as a formatted PDF.
- **Customizable Settings:** Configure your display name and maximum number of writing attempts.
- **API Key Validation:** Enter and verify OpenAI, Groq, and Tavily API keys directly through the application.

---

## How It Works

Draftwise follows a LangGraph-powered workflow consisting of four main stages.

### 1. Writer Agent

The writer agent uses OpenAI's `gpt-4o-mini` model to generate LinkedIn posts.

It follows a predefined writing structure:

- A strong opening hook.
- One clear and valuable takeaway.
- Short, readable paragraphs.
- Approximately 150–200 words.
- An engaging closing question or call to action.
- No hashtags.

The user can select a preferred writing tone before generation.

### 2. Conditional Web Research

When the writer determines that a topic requires current information, it can invoke the Tavily Search tool.

The retrieved information is passed back to the writer so it can incorporate relevant context into the draft.

If research is unnecessary, the workflow skips this stage.

### 3. Reviewer Agent

The reviewer agent uses Groq's `openai/gpt-oss-120b` model to evaluate the generated post.

It checks the draft against seven criteria:

1. Strength of the opening hook.
2. Clarity and value of the main takeaway.
3. Readability and paragraph structure.
4. Word count.
5. Quality of the closing question or CTA.
6. Professional yet human tone.
7. Absence of hashtags.

The reviewer returns one of two verdicts:

- **APPROVED:** The post meets the review criteria.
- **REJECTED:** The post requires further improvement.

### 4. Iterative Improvement

If a draft is rejected, the reviewer's feedback is passed back to the writer agent.

The writer generates an improved version addressing the identified issues, and the reviewer evaluates it again.

This process continues until:

- The reviewer approves the draft, or
- The configured maximum number of attempts is reached.

The application maintains a history of drafts and their corresponding feedback.

---

## Architecture

The application uses LangGraph to coordinate the writer, research, and reviewer workflow.

```text
                  USER INPUT
                      |
                      v
                WRITER AGENT
                (GPT-4o-mini)
                      |
                      v
              Conditional Search
                      |
             +--------+--------+
             |                 |
          Required          Not Required
             |                 |
             v                 |
         TAVILY SEARCH          |
             |                 |
             v                 |
       WRITER CONTINUES         |
             |                 |
             +--------+--------+
                      |
                      v
                EXTRACT DRAFT
                      |
                      v
               REVIEWER AGENT
               (Groq / GPT-OSS)
                      |
                      v
                  EVALUATION
                      |
               +------+------+
               |             |
            APPROVED       REJECTED
               |             |
               v             v
             FINISH      WRITER AGENT
                             |
                       Improved Draft
                             |
                             v
                          REVIEWER
```

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application language |
| Streamlit | Interactive frontend and application interface |
| LangGraph | Agent orchestration and workflow management |
| LangChain Core | Message handling and agent integration |
| OpenAI GPT-4o-mini | LinkedIn post generation |
| Groq GPT-OSS-120B | Draft evaluation and reviewing |
| Tavily Search | Conditional web research |
| Requests | API key validation through HTTP requests |
| ReportLab | PDF generation |
| HTML & CSS | Custom interface styling |
| JavaScript | Clipboard functionality |

---

## Project Structure

```text
Draftwise/
│
├── app.py
├── requirements.txt
└── README.md
```

### File Description

| File | Description |
|---|---|
| `app.py` | Main Streamlit application containing the UI, LangGraph workflow, writer and reviewer agents, and supporting functionality. |
| `requirements.txt` | Contains the Python dependencies required to run the project. |
| `README.md` | Project documentation and setup instructions. |

---

## Installation and Setup

Follow these steps to run Draftwise locally.

### Prerequisites

Make sure you have the following installed:

- Python 3.10 or later.
- pip.
- Git.

You will also need API keys from the following platforms:

| Service | Purpose | API Key |
|---|---|---|
| OpenAI | Writer agent | [Get API Key](https://platform.openai.com/api-keys) |
| Groq | Reviewer agent | [Get API Key](https://console.groq.com/keys) |
| Tavily | Web research | [Get API Key](https://app.tavily.com/) |

**Note:** OpenAI API access is separately billed. Groq and Tavily usage is subject to their respective account limits and pricing.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/draftwise.git
```

Navigate into the project directory:

```bash
cd draftwise
```

Replace `YOUR_USERNAME` with your GitHub username and `draftwise` with your repository name if different.

### Step 2: Create a Virtual Environment

Create an isolated Python environment:

```bash
python -m venv venv
```

Activate it:

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### Step 3: Create the Requirements File

Create a file named `requirements.txt` in the project root directory.

Add the following dependencies:

```text
streamlit>=1.40
requests
reportlab
langchain-core
langchain-groq
langchain-openai
langchain-tavily
langgraph
```

### Step 4: Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

---

## Running the Application

Once the application opens:

1. Enter your OpenAI API key.
2. Enter your Tavily API key.
3. Enter your Groq API key.
4. Click **Verify keys**.
5. Once the keys are verified, configure your writing preferences.
6. Enter the topic for your LinkedIn post.
7. Select your preferred writing tone.
8. Start generating your post.

Draftwise will execute the LangGraph workflow, generate a draft, review it, and refine it if necessary.

You can then copy the generated post or export it as a PDF.

---

## Deployment

Draftwise can be deployed using **Streamlit Community Cloud**.

### Deployment Steps

1. Push your project to a GitHub repository.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in using your GitHub account.
4. Select **Create app**.
5. Choose your Draftwise repository.
6. Select the `main` branch.
7. Set the main file path to `app.py`.
8. Click **Deploy**.

Once deployment completes, Streamlit provides a public URL through which the application can be accessed.

Since Draftwise accepts API keys directly through its interface, users can supply their own credentials when using the application.

---

## API Keys and Security

- API keys are entered through password-type input fields.
- Keys are validated before the workflow starts.
- Credentials are held in Streamlit session state.
- The application does not intentionally write API keys to local files.
- Error messages are sanitized to avoid displaying entered API keys.

**Important:** Streamlit session state is server-side application state, not browser-only storage. API keys are transmitted to the hosted application server and are used to make requests to the respective providers. Only use this application if you trust its deployment and hosting environment.

Never commit API keys, passwords, or other credentials to GitHub.

---

## Future Improvements

Potential enhancements for future versions include:

- Support for additional LLM providers.
- More granular writing-style customization.
- Advanced post-quality scoring.
- Multiple post variations from a single topic.
- Saving and retrieving previous writing sessions.
- User authentication and personalized workspaces.
- Additional export formats.

---

## Author

**Draftwise** — An AI-powered LinkedIn writing and reviewing workflow.

Built with Python, Streamlit, LangGraph, and multiple LLM integrations.

---

## License

This project is available under the MIT License.

If you intend to use the MIT License, add a `LICENSE` file containing the standard MIT License text and your copyright details.
