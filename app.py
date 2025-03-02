import streamlit as st
from utils.openai_client import OpenAIClientWrapper
import pandas as pd
import altair as alt
from collections import Counter

# Set page config
st.set_page_config(
    page_title="LLM Consistency Analyzer",
    page_icon="🤖",
    layout="wide"
)

# Apply custom CSS for better spacing and styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        padding-top: 2rem !important;  /* Added top padding */
    }
    .subheader {
        font-size: 1.7rem !important;  /* Same size as before */
        font-style: italic !important;  /* Italicize the text */
        margin-bottom: 1.5rem !important;
    }
    /* Add more top padding to the main container */
    .block-container {
        padding-top: 3rem !important;
    }
    /* Ensure button styling */
    .stButton>button {
        width: 100%;
        background-color: #4CAF50 !important; /* Green button */
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Title with emoji
    st.markdown('<p class="main-header">🔄 Assessing LLM Outputs for Consistency 📊</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Compare how models respond to the same prompt across multiple runs</p>', unsafe_allow_html=True)
    
    # Initialize OpenAI client
    openai_client = OpenAIClientWrapper()
    
    # Create two-column layout for main interface
    col1, col2 = st.columns([1, 2])
    
    # Left column for controls (without card)
    with col1:
        # Model selection
        st.subheader("🤖 Model Selection")
        deployment_name = st.selectbox(
            "Select model deployment:",
            options=["gpt-4o", "gpt-4o-mini"],
            index=0
        )
        
        # Parameters
        st.subheader("⚙️ Parameters")
        temperature = st.slider(
            "Temperature:", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )
        
        num_calls = st.slider(
            "Number of runs:", 
            min_value=1, 
            max_value=50, 
            value=10,
            help="How many times to call the model with the same prompt"
        )
        
    # Right column for prompt input (without card)
    with col2:
        st.subheader("💬 Question")
        question = st.text_area(
            "Enter your question:",
            placeholder="Type your question here. Try something that might have multiple valid answers, like 'What's a good name for a pet turtle?'",
            height=120
        )
        
        # Move button to be under the question input
        submit_button = st.button("🚀 Generate Responses", use_container_width=True)
    
    # Sidebar for system prompt
    with st.sidebar:
        st.subheader("🧠 System Prompt")
        
        # Dropdown for preset prompts with Concise Answers as default
        preset_option = st.selectbox(
            "Choose a preset:",
            ["Concise Answers (10 words max)", "Research Assistant"],
            index=0
        )
        
        # Default values based on selection
        if preset_option == "Research Assistant":
            default_prompt = "You are a helpful research assistant. Provide thorough and informative responses to questions."
        else:  # Concise Answers
            default_prompt = "Answer the user's question in 10 words or less. Do not include explanations. Provide only the direct answer in as few words as possible."
        
        # Editable text area that updates based on selection
        system_prompt = st.text_area(
            "Edit system prompt:",
            value=default_prompt,
            height=150
        )
        
        # Additional tips in sidebar
        st.divider()
        st.caption("💡 **Tips:**")
        st.caption("• Higher temperatures produce more varied responses")
        st.caption("• Try different system prompts to see how they affect consistency")
        st.caption("• For best results, ask clear, specific questions")
    
    # Results area
    if submit_button and question:
        # Create a placeholder for status updates
        st.divider()
        status_placeholder = st.empty()
        status_placeholder.info("🔄 Request underway, please wait...")
                
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
        
        # Results in tabs
        tab1, tab2 = st.tabs(["📊 Visualization", "📋 Full Results"])
        
        with tab1:
            # Display statistics
            st.markdown(f"### Response Diversity: {unique_count} unique out of {num_calls} total")
            
            # Calculate consistency percentage
            consistency = 100 * (num_calls - unique_count + 1) / num_calls if num_calls > 1 else 100
            st.metric("Consistency Score", f"{consistency:.1f}%", 
                      help="Higher percentages indicate more consistent responses")
            
            # Display the chart
            if len(chart_data) > 0:
                # For chart display, use more readable truncation
                chart_data['Display'] = chart_data['Response'].apply(lambda x: (x[:40] + '...') if len(x) > 40 else x)
                
                # Create horizontal bar chart with improved readability
                chart = alt.Chart(chart_data).mark_bar().encode(
                    y=alt.Y('Display:N', 
                          title='Response', 
                          sort='-x',
                          axis=alt.Axis(labelLimit=400)),
                    x=alt.X('Count:Q', 
                          title='Frequency',
                          axis=alt.Axis(
                              tickMinStep=1,
                              titleFontSize=14,   # Larger font for axis title
                              titleFontWeight='bold',  # Bold font for axis title
                              labelFontSize=12    # Larger font for tick labels
                          )),
                    tooltip=['Response', 'Count']
                ).properties(
                    height=max(300, len(chart_data) * 30)
                )
                
                st.altair_chart(chart, use_container_width=True)
        
        with tab2:
            # Display results as a table
            st.markdown("### Response Summary")
            
            # Create a dataframe with full responses and their counts
            results_df = pd.DataFrame({
                'Response': list(response_counts.keys()),
                'Count': list(response_counts.values())
            }).sort_values('Count', ascending=False)
            
            # Add percentage column
            results_df['Percentage'] = results_df['Count'] / num_calls * 100
            results_df['Percentage'] = results_df['Percentage'].apply(lambda x: f"{x:.1f}%")
            
            # Display the table
            st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    elif submit_button:
        st.warning("⚠️ Please enter a question first.")

if __name__ == "__main__":
    main()