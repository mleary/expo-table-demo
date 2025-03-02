import streamlit as st
from utils.openai_client import OpenAIClientWrapper
import pandas as pd
import altair as alt
from collections import Counter
import time

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
        font-size: 1.7rem !important;
        font-style: italic !important;
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
    /* Improve section spacing */
    .section-header {
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    /* Improve card styling */
    .card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #eaeaea;
    }
    /* Adjust table styling */
    .dataframe-container {
        margin-top: 1rem;
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
    
    # Left column for controls
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
        
        # Add advanced options in an expander
        with st.expander("🔧 Advanced Options"):
            max_tokens = st.slider(
                "Max response length:", 
                min_value=50, 
                max_value=500, 
                value=150,
                step=10,
                help="Maximum number of tokens in each response"
            )
        
    # Right column for prompt input
    with col2:
        st.subheader("💬 Question")
        question = st.text_area(
            "Enter your question:",
            placeholder="Type your question here. Try something that might have multiple valid answers, like 'What's a good name for a pet turtle?'",
            height=120
        )
        
        # Example questions
        with st.expander("📝 Example questions"):
            st.caption("• What's a good name for a pet turtle?")
            st.caption("• What is the capital of France?")
            st.caption("• Give me a one-line JavaScript function to reverse a string")
            st.caption("• What's the best programming language to learn first?")
        
        # Generate button
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
            default_prompt = "Answer the user's question in 10 words or less. Do not include explanations. Provide only the direct answer in as few words as possible. I understand questions might be subjective. Do not tell me that, give me an answer. I understand it might be subjective. If you say subjective, I will fine you a days work."
        
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
    
    # Clear spacing before results
    st.write("")
    
    # Results area
    if submit_button and question:
        # Add a visual separator
        st.divider()
        
        # Create a placeholder for status updates
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        status_placeholder.info("🔄 Request underway, please wait...")
        
        # Start timer
        start_time = time.time()
        
        # Create progress bar
        progress_bar = progress_placeholder.progress(0)
        
        # Process responses in batches to show progress
        responses = []
        batch_size = max(1, num_calls // 10)  # Show at least 10 progress updates
        
        for i in range(0, num_calls, batch_size):
            # Calculate batch end (handling the last batch)
            batch_end = min(i + batch_size, num_calls)
            batch_count = batch_end - i
            
            # Call the API for this batch
            batch_responses = openai_client.call_llm(
                question, 
                temperature, 
                batch_count, 
                deployment_name, 
                system_prompt
            )
            
            # Add batch responses to the full list
            responses.extend(batch_responses)
            
            # Update progress
            progress_bar.progress(len(responses) / num_calls)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Clear the status message and progress bar
        status_placeholder.empty()
        progress_placeholder.empty()
        
        # Show completion message with time
        st.success(f"✅ Generated {num_calls} responses in {processing_time:.1f} seconds")
        
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
            # Create two columns for metrics
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                # Display unique response count
                st.metric(
                    "Unique Responses", 
                    f"{unique_count} of {num_calls}",
                    help="Number of different responses received"
                )
            
            with metric_col2:
                # Calculate consistency percentage
                consistency = 100 * (1 - (unique_count - 1) / num_calls) if num_calls > 1 else 100
                st.metric(
                    "Consistency Score", 
                    f"{consistency:.1f}%", 
                    help="Higher percentages indicate more consistent responses"
                )
            
            # Display the chart
            if len(chart_data) > 0:
                st.subheader("Response Distribution")
                
                # For chart display, use more readable truncation
                chart_data['Display'] = chart_data['Response'].apply(lambda x: (x[:40] + '...') if len(x) > 40 else x)
                
                # Create horizontal bar chart with improved readability
                chart = alt.Chart(chart_data).mark_bar().encode(
                    y=alt.Y(
                        'Display:N', 
                        title='Response', 
                        sort='-x',
                        axis=alt.Axis(labelLimit=400)
                    ),
                    x=alt.X(
                        'Count:Q', 
                        title='Frequency',
                        axis=alt.Axis(
                            tickMinStep=1,     # Force whole number ticks
                            values=list(range(1, max(chart_data['Count']) + 1)),  # Explicitly set whole numbers
                            titleFontSize=16,   # Larger font for axis title
                            titlePadding=15,    # Add padding to the title
                            titleFontWeight='bold',  # Bold font for axis title
                            labelFontSize=14,   # Larger font for tick labels
                            labelColor='#333333' # Darker color for better contrast
                        )
                    ),
                    tooltip=['Response', 'Count']
                ).properties(
                    height=max(300, len(chart_data) * 30)
                )
                
                st.altair_chart(chart, use_container_width=True)
        
        with tab2:
            # Display results as a table
            st.subheader("Response Summary")
            
            # Create a dataframe with full responses and their counts
            results_df = pd.DataFrame({
                'Response': list(response_counts.keys()),
                'Count': list(response_counts.values())
            }).sort_values('Count', ascending=False)
            
            # Add percentage column
            results_df['Percentage'] = results_df['Count'] / num_calls * 100
            results_df['Percentage'] = results_df['Percentage'].apply(lambda x: f"{x:.1f}%")
            
            # Display the table
            st.dataframe(
                results_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Response": st.column_config.TextColumn(
                        "Response",
                        width="large",
                    ),
                    "Count": st.column_config.NumberColumn(
                        "Count",
                        format="%d",
                    ),
                    "Percentage": st.column_config.TextColumn(
                        "Percentage",
                        width="small",
                    ),
                }
            )
    
    elif submit_button:
        st.warning("⚠️ Please enter a question first.")

if __name__ == "__main__":
    main()