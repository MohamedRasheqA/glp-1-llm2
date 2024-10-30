from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from openai import OpenAI
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from datetime import datetime

app = Flask(__name__, static_url_path='/', static_folder='static')
CORS(app)
load_dotenv()

@app.route('/')
def home():
    return app.send_static_file('index.html')

class GLP1Bot:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pplx_api_key = os.getenv("PPLX_API_KEY")
        self.pplx_model = os.getenv("PPLX_MODEL", "llama-3.1-sonar-large-128k-online")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        if not self.openai_api_key or not self.pplx_api_key:
            raise ValueError("Required API keys not found in environment variables")
            
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.pplx_headers = {
            "Authorization": f"Bearer {self.pplx_api_key}",
            "Content-Type": "application/json"
        }
        
        # System prompts remain the same as in original code
        self.pplx_system_prompt = """You are a medical information assistant specialized in GLP-1 medications.
        Provide detailed, evidence-based information about GLP-1 medications, focusing on medical accuracy.
        Cover important aspects such as:
        - Mechanism of action
        - Proper usage and administration
        - Expected outcomes and timeframes
        - Potential side effects and management
        - Drug interactions and contraindications
        - Storage requirements
        - Lifestyle modifications for optimal results"""
        self.gpt_validation_prompt = """You are a medical content validator. Review and enhance the following information about GLP-1 medications.
        Ensure the response is:
        1. Medically accurate and evidence-based
        2. Well-structured with clear sections
        3. Includes appropriate medical disclaimers
        4. Easy to understand for patients
        5. Comprehensive yet concise
        6. Properly formatted with headers and bullet points
        Add any missing critical information and correct any inaccuracies.
        Always maintain a professional yet approachable tone."""

    def get_pplx_response(self, query: str) -> Optional[str]:
        """Get initial response from PPLX API"""
        try:
            payload = {
                "model": self.pplx_model,
                "messages": [
                    {"role": "system", "content": self.pplx_system_prompt},
                    {"role": "user", "content": query}
                ],
                "temperature": float(os.getenv("TEMPERATURE", "0.1")),
                "max_tokens": int(os.getenv("MAX_TOKENS", "1000"))
            }
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=self.pplx_headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[red]Error communicating with PPLX: {str(e)}[/red]")
            return None

    def validate_with_gpt(self, pplx_response: str, original_query: str) -> Optional[str]:
        """Validate and enhance PPLX response using GPT"""
        try:
            validation_prompt = f"""
            Original query: {original_query}
            PPLX Response to validate:
            {pplx_response}
            Please validate and enhance this response according to medical standards and best practices.
            Ensure all information is accurate and properly structured.
            """
            completion = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self.gpt_validation_prompt},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=float(os.getenv("TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("MAX_TOKENS", "1500"))
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"[red]Error validating with GPT: {str(e)}[/red]")
            return None

    def format_response(self, response: str) -> str:
        """Format the response with safety disclaimer"""
        if not response:
            return "I apologize, but I couldn't generate a response at this time. Please try again."
        safety_disclaimer = """
        IMPORTANT MEDICAL DISCLAIMER:
        - This information is for educational purposes only
        - Consult your healthcare provider for personalized medical advice
        - Follow your prescribed treatment plan
        - Report any side effects to your healthcare provider
        - Individual results may vary
        - Never modify your medication regimen without professional guidance
        """
        if "disclaimer" not in response.lower():
            response += safety_disclaimer
        return response

    def categorize_query(self, query: str) -> str:
        """Categorize the user query"""
        categories = {
            "dosage": ["dose", "dosage", "how to take", "when to take", "injection", "administration"],
            "side_effects": ["side effect", "adverse", "reaction", "problem", "issues", "symptoms"],
            "benefits": ["benefit", "advantage", "help", "work", "effect", "weight", "glucose"],
            "storage": ["store", "storage", "keep", "refrigerate", "temperature"],
            "lifestyle": ["diet", "exercise", "lifestyle", "food", "alcohol", "eating"],
            "interactions": ["interaction", "drug", "medication", "combine", "mixing"],
            "cost": ["cost", "price", "insurance", "coverage", "afford"]
        }
        query_lower = query.lower()
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        return "general"

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query through both PPLX and GPT"""
        try:
            if not user_query.strip():
                return {
                    "status": "error",
                    "message": "Please enter a valid question."
                }
            # Step 1: Get initial response from PPLX
            print("\n:magnifying_glass: Retrieving information from medical knowledge base...")
            pplx_response = self.get_pplx_response(user_query)
            if not pplx_response:
                return {
                    "status": "error",
                    "message": "Failed to retrieve information from knowledge base."
                }
            # Step 2: Validate and enhance with GPT
            print(":white_tick: Validating and enhancing information...")
            validated_response = self.validate_with_gpt(pplx_response, user_query)
            if not validated_response:
                return {
                    "status": "error",
                    "message": "Failed to validate information."
                }
            # Format final response
            query_category = self.categorize_query(user_query)
            formatted_response = self.format_response(validated_response)
            return {
                "status": "success",
                "query_category": query_category,
                "original_query": user_query,
                "pplx_response": pplx_response,
                "response": formatted_response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}"
            }

bot = GLP1Bot()

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get('query')
        
        if not query:
            return jsonify({
                "status": "error",
                "message": "No query provided"
            }), 400

        response = bot.process_query(query)
        return jsonify(response)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)