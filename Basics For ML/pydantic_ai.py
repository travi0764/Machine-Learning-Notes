import os 
import json 
import base64
from pathlib import Path 
from typing import Annotated, List, Dict, Any, Optional
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models import KnownModelName
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from dotenv import load_dotenv
import asyncio
import fitz
from openai import AsyncAzureOpenAI
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIChatModel
from generate_token import generate_token
from schema import Section, FormJSON

load_dotenv()

# Paths
PDF_PATH = Path("Agent2/pdfs/Cosentyx_New.pdf")
IMAGES_DIR = Path("Agent3/images")
OUTPUT_DIR = Path("Agent3/output")
RULES_DIR = Path("Agent3/rules")
PROMPTS_DIR = Path("Agent3/prompts")

# Create directories
IMAGES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
RULES_DIR.mkdir(exist_ok=True)
PROMPTS_DIR.mkdir(exist_ok=True)

# Azure OpenAI Setup
OPENAI_ENDPOINT = "https://api-tst.nonprod.az.mckesson.com/common/mcktech/az-openai-proxy-enhanced/"
OPENAI_API_VERSION = "2024-08-01-preview"
GPT_MODEL = "mckopenai-gpt5-2025-08-07"

token = generate_token()
client = AsyncAzureOpenAI(
    azure_endpoint=OPENAI_ENDPOINT, 
    api_version=OPENAI_API_VERSION, 
    azure_ad_token=token
)

MODEL = OpenAIChatModel(GPT_MODEL, provider=OpenAIProvider(openai_client=client))

# Agent with improved system prompt
agent = Agent(
    MODEL,
    system_prompt="""You are an expert at converting PDF forms into precise Form-JSON structures.

YOUR WORKFLOW (FOLLOW IN ORDER):
1. First, call 'load_rules_and_prompts' to get extraction rules and prompts
2. Call 'extract_text_with_azure' to get OCR text from the PDF
3. Analyze the OCR text to find which page(s) contain "Prescription Information"
4. Call 'pdf_to_images' to convert PDF pages to images
5. Call 'load_page_image' with the correct page number to view the relevant page
6. Cross-reference the visual image with OCR text to extract accurate data
7. Apply the rules and prompts strictly when structuring the output

CRITICAL RULES:
- Your output MUST be 100% valid Section matching the Pydantic schema
- Never invent fields. Use "Not found" or leave optional fields null if unclear
- Follow section rules from the loaded rules file exactly
- Use prompts from the loaded prompts file to guide extraction
- Always verify extracted data against BOTH OCR text AND page images
- If information conflicts between OCR and image, trust the image""",
    output_type=Section,
    retries=2
)


@agent.tool_plain
def pdf_to_images(pdf_path: str) -> dict:
    """Convert PDF to images and save them. Returns info about converted pages."""
    print("Converting PDF to images...")
    pdf_path = Path(pdf_path)
    print(f"PDF Path: {pdf_path}")
    print(f"Output Folder: {IMAGES_DIR}")
    
    if not pdf_path.exists():
        return {"error": f"PDF not found at {pdf_path}", "pages_converted": 0}
    
    doc = fitz.open(pdf_path)
    pages_info = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=200)
        image_path = IMAGES_DIR / f"page_{page_num + 1}.png"
        pix.save(image_path)
        pages_info.append({
            "page_number": page_num + 1,
            "image_path": str(image_path)
        })
    
    doc.close()
    return {
        "total_pages": len(doc),
        "pages_converted": len(pages_info),
        "pages": pages_info,
        "message": f"Successfully converted {len(pages_info)} pages to images"
    }


@agent.tool
async def load_page_image(ctx: RunContext[None], page_number: int) -> tuple[str, bytes]:
    """Load a specific page image for visual analysis. Returns (description, image_bytes)."""
    image_path = IMAGES_DIR / f"page_{page_number}.png"
    
    if not image_path.exists():
        raise ValueError(f"Image for page {page_number} not found at {image_path}")
    
    with open(image_path, "rb") as file:
        image_bytes = file.read()
    
    return (f"Page {page_number} image", image_bytes)


@agent.tool_plain
def extract_text_with_azure(pdf_path: str) -> dict:
    """Extract text from PDF using Azure Document Intelligence with page information."""
    print("Extracting text with Azure Document Intelligence...")
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        return {"error": f"PDF not found at {pdf_path}"}
    
    loader = AzureAIDocumentIntelligenceLoader(
        file_path=str(pdf_path),
        api_key=os.getenv("DOCUMENT_INTELLIGENCE_KEY"),
        api_endpoint=os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT"),
        api_model="prebuilt-layout",
        analysis_features=["ocrHighResolution"],
    )
    
    docs = loader.load()
    
    # Organize text by pages
    full_text = "\n\n".join([d.page_content for d in docs])
    
    # Try to identify pages with metadata
    pages_text = []
    for i, doc in enumerate(docs):
        page_info = {
            "page_number": i + 1,
            "content": doc.page_content,
            "has_prescription_info": "prescription" in doc.page_content.lower()
        }
        pages_text.append(page_info)
    
    return {
        "full_text": full_text,
        "pages": pages_text,
        "total_pages": len(docs),
        "message": "OCR extraction complete"
    }


@agent.tool_plain
def load_rules_and_prompts() -> dict:
    """Load extraction rules and prompts for the Prescription Information section."""
    print("Loading rules and prompts...")
    
    rules = {}
    for file in RULES_DIR.glob("*.txt"):
        section = file.stem
        rules[section] = file.read_text(encoding="utf-8")
    
    prompts = {}
    for file in PROMPTS_DIR.glob("*.txt"):
        prompts[file.stem] = file.read_text(encoding="utf-8")
    
    return {
        "rules": rules,
        "prompts": prompts,
        "message": f"Loaded {len(rules)} rules and {len(prompts)} prompts"
    }


def dump_json(data, file_path):
    """Dump data to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


async def main():
    """Main execution function."""
    print("Starting PDF extraction process...")
    
    # Run the agent
    result = await agent.run(
        f"""Extract the Prescription Information section from: {PDF_PATH}
        
Follow the workflow:
1. Load rules and prompts first
2. Extract text with Azure Document Intelligence
3. Identify which page(s) have Prescription Information
4. Convert PDF to images
5. Load the relevant page image(s)
6. Cross-reference OCR text with page images
7. Apply rules and prompts to extract accurate data
8. Generate the final JSON structure"""
    )
    
    print("\n" + "="*50)
    print("Extraction Result:")
    print("="*50)
    
    # Save output
    output_data = result.output.model_dump()
    dump_json(output_data, OUTPUT_DIR / "summary.json")
    print(f"\nOutput saved to: {OUTPUT_DIR / 'summary.json'}")
    
    # Save usage info
    print("\nUsage:")
    print(result.usage())
    
    # Save raw messages
    with open(OUTPUT_DIR / "summary_raw.json", "w", encoding="utf-8") as f:
        f.write(result.all_messages_json().decode("utf-8"))
    print(f"Raw messages saved to: {OUTPUT_DIR / 'summary_raw.json'}")
    
    return result


if __name__ == "__main__":
    result = asyncio.run(main())
    print("\n✓ Processing complete!")
