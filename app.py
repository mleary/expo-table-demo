import streamlit as st
from utils.openai_client import OpenAIClientWrapper
import pandas as pd
import altair as alt
from collections import Counter

# Set page to wide mode
st.set_page_config(layout="wide")

def main():
    st.title("Azure OpenAI Multiple Response Generator")
    
    # Initialize OpenAI client
    openai_client = OpenAIClientWrapper()
    
    # Sidebar for system prompt
    with st.sidebar:
        st.subheader("System Prompt")
        
        # Dropdown for preset prompts
        preset_option = st.selectbox(
            "Choose a preset:",
            ["Research Assistant", "Concise Answers (10 words max)"]
        )
        
        # Add some space
        st.write("")
        st.write("")
        
        # Default values based on selection
        if preset_option == "Research Assistant":
            default_prompt = "You are a helpful research assistant. Provide thorough and informative responses to questions."
        else:  # Concise Answers
            default_prompt = "Answer the user's question in 10 words or less. Do not include explanations and only provide one answer. Provide only the direct answer in as few words as possible."
        
        # Editable text area that updates based on selection
        system_prompt = st.text_area(
            "Edit system prompt:",
            value=default_prompt,
            height=150
        )
    
    # Main content area
    # Get user inputs
    question = st.text_area("Enter your question:", height=150)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    with col2:
        num_calls = st.slider("Number of times to call the LLM:", min_value=1, max_value=50, value=1)
    
    with col3:
        deployment_name = st.selectbox(
            "Select model deployment:",
            options=["gpt-4o", "gpt-4o-mini"],
            index=0
        )

    if st.button("Submit"):
        if question:
            # Create a placeholder for status updates
            status_placeholder = st.empty()
            status_placeholder.info("Request underway, please wait...")
                    
            with st.spinner(f"Generating {num_calls} responses..."):
                responses = openai_client.call_llm(question, temperature, num_calls, deployment_name, system_prompt)
            
            # Clear the status message once completed
            status_placeholder.empty()
            
            # Count occurrences of each unique response
            response_counts = Counter(responses)
            unique_count = len(response_counts)
            
            # Create a DataFrame for the chart
            chart_data = pd.DataFrame({
                'Response': list(response_counts.keys()),
                'Count': list(response_counts.values())
            }).sort_values('Count', ascending=False)
            
            # Display statistics
            st.subheader(f"Response Diversity: {unique_count} unique responses out of {num_calls} total")
            
            # Display the chart
            if len(chart_data) > 0:
                # Truncate long responses for the chart labels
                chart_data['Display'] = chart_data['Response'].apply(lambda x: (x[:30] + '...') if len(x) > 30 else x)
                
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('Display:N', 
                          title='Response', 
                          sort='-y',
                          axis=alt.Axis(
                              labelAngle=-45,  # Angle the labels 45 degrees
                              labelLimit=200   # Allow longer label text
                          )),
                    y=alt.Y('Count:Q', title='Frequency'),
                    tooltip=['Display', 'Count']
                ).properties(
                    height=300
                )
                
                st.altair_chart(chart, use_container_width=True)
            
            # Display results as a table instead of expanders
            st.subheader("Response Summary:")
            
            # Create a dataframe with full responses and their counts
            results_df = pd.DataFrame({
                'Response': list(response_counts.keys()),
                'Count': list(response_counts.values())
            }).sort_values('Count', ascending=False)
            
            # Display the table
            st.dataframe(results_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()