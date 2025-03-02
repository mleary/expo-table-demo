import os
from openai import AzureOpenAI

class OpenAIClientWrapper:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )

    def call_llm(self, prompt, temperature, num_calls, deployment_name, system_prompt="You are a helpful assistant.", 
                max_tokens=150, top_p=1.0, frequency_penalty=0.0):
        responses = []
        for _ in range(num_calls):
            response = self.client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty
            )
            responses.append(response.choices[0].message.content.strip())
        return responses