# Tableau LangChain Starter Kit

A powerful integration that brings AI functionality to Tableau Server or Tableau Cloud using LangChain, enabling natural language interactions with the data you trust in Tableau.

This repo is an implementation of [tableau_langchain](https://github.com/tableau/tableau_langchain) and it's [PyPi registry](https://pypi.org/project/langchain-tableau/).

## 🚀 Features

- Natural language querying of Tableau data
- Available via Web interface or Dashboard extension
- Support for both Tableau Server and Tableau Cloud

## 📋 Prerequisites

Before you begin, ensure you have the following:

- **Tableau Server Version 2025.1** or later OR **Tableau Cloud**, a free Tableau Cloud trial is available via the [Tableau Developer Program](https://www.tableau.com/en-gb/developer)
- **Python 3.12+** - [Download Python](https://python.org/downloads/)
- **Git** - [Download Git](https://git-scm.com/downloads/)
- **API credentials** for your chosen AI model (OpenAI, etc.)

## ⚠️ Warning

When using this code, data from Tableau will be sent to an external AI model (by default, OpenAI). For learning and testing, it is strongly recommended to use the Superstore dataset included with Tableau.

If you need to process sensitive or proprietary information, consider configuring the tool to use a local AI model instead of an external service. This approach ensures your data remains within your organisation’s infrastructure and reduces the risk of data exposure.

## 📺 Step Up Guide 

<a href="https://www.youtube.com/watch?v=2sPJfxeoJ90">
    <img src="https://img.youtube.com/vi/2sPJfxeoJ90/maxresdefault.jpg" width ="80%">
  </a>

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com//tableau-langchain-starter-kit.git
cd tableau-langchain-starter-kit
```

### 2. Create Virtual Environment

Creating a virtual environment helps isolate project dependencies:

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

💡 **Tip:** You should see `(venv)` at the beginning of your command prompt when the virtual environment is active.

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

If you encounter any installation issues, try upgrading pip first:
```bash
pip install --upgrade pip
```

## ⚙️ Configuration

### Environment Variables Setup

1. Copy the template environment file:
```bash
cp .env_template .env
```

2. Open the `.env` file in your preferred text editor and configure the following variables:

```
# Model Providers
OPENAI_API_KEY='from OpenAI developer portal'
MODEL_PROVIDER='openai'

# LangSmith
LANGCHAIN_TRACING='true'
LANGCHAIN_API_KEY="from Langsmith app"
LANGCHAIN_PROJECT="Langsmith project name"

# Tableau Server / Cloud
TABLEAU_DOMAIN='your Tableau Cloud or Server domain'
TABLEAU_SITE='your Tableau site'
TABLEAU_JWT_CLIENT_ID='from Connected App configuration page'
TABLEAU_JWT_SECRET_ID='from Connected App configuration page'
TABLEAU_JWT_SECRET='from Connected App configuration page'
TABLEAU_API_VERSION='3.21'
TABLEAU_USER='user account for the Agent'
DATASOURCE_LUID='unique identifier for a data source found via the graphql metadata API'
```

⚠️ **Security Note:** Never commit your `.env` file to version control. It's already included in `.gitignore`.

## 🏃‍♂️ Running the Application

### Testing Mode (Command Line)

Perfect for testing your configuration and running quick experiments:

```
python main.py
```

This mode allows you to:
- Test your Tableau connection
- Verify AI service integration
- Run sample queries from the command line

### Web Interface Mode

Launch the full web application with dashboard extension support:

```bash
python web_app.py
```

Once running, open your browser and navigate to:
- **Local development:** `http://localhost:8000`
- The application will display the correct URL in the terminal

You will now be able to ask questions in natural language like:
   - "What are the trends in customer satisfaction?"
   - "Compare revenue between Q1 and Q2"
   - "Show me outliers in the sales data"

### Dashboard Extension

Launch the full web application with dashboard extension support:

```bash
python web_app.py
```

Once running, open your Tableau workbook, or the [Superstore Dashboard](dashboard_extension\Superstore.twbx)

On a dashboard page, in the bottom left menu drag a dashboard exention, local extension, and select [tableau_langchain.trex](dashboard_extension\tableau_langchain.trex) from the dashboard_extension folder. 


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Get Involved

- Check out the [Tableau LangChain](https://github.com/tableau/tableau_langchain) repo for further developments
- Join the [#tableau-langchain](https://tableau-datadev.slack.com/archives/C07LMAVG4N6) conversation on Slack. Sign up to the [DataDev Slack channel here.](https://tabsoft.co/JoinTableauDev)

## 🙏 Acknowledgments

- [Tableau LangChain](https://github.com/tableau/tableau_langchain) the team developing the tools
- [LangChain](https://langchain.com/) for the AI framework
- [Tableau](https://tableau.com/) for the visualization platform
- All contributors who have helped improve this project

---

**⭐ If you find this project helpful, please consider giving it a star!**
