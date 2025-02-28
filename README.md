# Azure OpenAI Multiple Response Generator

A Streamlit application that sends the same prompt to Azure OpenAI multiple times with customizable parameters, visualizing the diversity of responses.

## Features

- Send the same prompt to Azure OpenAI multiple times (up to 50 requests)
- Adjust temperature to control randomness in responses
- Choose between different model deployments (gpt-4o, gpt-4o-mini)
- Select from preset system prompts or create your own
- Visualize response diversity with interactive charts
- View a summary table of all unique responses and their frequencies

## Requirements

- Python 3.8+
- Azure OpenAI API access
- Valid Azure OpenAI deployments for gpt-4o and gpt-4o-mini

## Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/yourusername/azure-openai-multiple-responses.git
   cd azure-openai-multiple-responses