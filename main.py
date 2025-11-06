from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import io
import os
from dotenv import load_dotenv
import google.generativeai as genai
import uvicorn

# --- 1. Load Environment Variables ---
load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please create a .env file and add your key.")

# Initialize the Gemini client
genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash-image"

app = FastAPI(
    title="ACP Sheet Application API",
    description="API for applying ACP sheets to building walls",
    version="1.0.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def generate_multimodal_response(prompt: str, input_images: list[Image.Image]) -> list[tuple[str, str]]:
    """
    Calls the Gemini API with a text prompt and multiple input images,
    and processes a multimodal response (text and/or generated images).
    
    HOW IT WORKS:
    1. Takes your prompt (instructions) and images
    2. Formats them for Gemini API
    3. Sends to Gemini for processing
    4. Receives response (can be text, image, or both)
    5. Returns structured list of (type, content) tuples
    
    Example return: [('text', 'Processing complete'), ('image', <PIL.Image>)]
    """
    model = genai.GenerativeModel(MODEL_NAME)
    
    content_parts = []
    
    # Add images first
    for img in input_images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        content_parts.append(img)
    
    # Add prompt text
    content_parts.append(prompt)
    
    response_content = []

    try:
        response = model.generate_content(content_parts)

        if hasattr(response, 'parts') and response.parts:
            for part in response.parts:
                if hasattr(part, 'text') and part.text:
                    response_content.append(('text', part.text))
                elif hasattr(part, 'inline_data') and part.inline_data:
                    mime_type = part.inline_data.mime_type
                    image_bytes = part.inline_data.data
                    if "image" in mime_type:
                        try:
                            generated_img = Image.open(io.BytesIO(image_bytes))
                            response_content.append(('image', generated_img))
                        except Exception as e:
                            response_content.append(('text', f"Error displaying generated image: {e}"))
                    else:
                        response_content.append(('text', f"Received inline data of type {mime_type} that is not an image."))
        elif hasattr(response, 'text'):
            # Handle simple text response
            response_content.append(('text', response.text))
        else:
            response_content.append(('text', "Model generated an empty response."))
            
    except Exception as e:
        response_content.append(('text', f"An error occurred: {e}"))
    
    return response_content

@app.get("/")
async def read_root():
    """API Information"""
    return {
        "message": "ACP Sheet Application API",
        "version": "1.0.0",
        "model": MODEL_NAME,
        "endpoints": {
            "GET /": "API Information (this page)",
            "GET /health": "Health check",
            "POST /apply_acp": "Apply ACP sheet to wall image",
            "POST /apply_acp_single": "Apply ACP sheet using single prompt (faster but less precise)"
        },
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": MODEL_NAME}


@app.post("/apply_acp")
async def apply_acp_endpoint(wall_image: UploadFile = File(...), acp_sheet_image: UploadFile = File(...)):
    """
    TWO-STEP APPROACH (More Accurate):
    Applies the ACP sheet to the wall image using a two-step process:
    
    Step 1: Generate a precise mask identifying wall surfaces
    Step 2: Apply ACP texture to masked areas only
    
    This approach gives better control and more accurate results.
    """
    try:
        wall_img = Image.open(wall_image.file).convert('RGB')
        acp_img = Image.open(acp_sheet_image.file).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # ========== STEP 1: Generate the mask (IMPROVED PROMPT) ==========
    mask_prompt = """
    TASK: Create a precise segmentation mask for wall surfaces in this building image.
    
    REQUIREMENTS:
    - Generate a black and white binary mask
    - WHITE areas (255): Main building wall surfaces that should receive ACP cladding
    - BLACK areas (0): Everything else (windows, doors, roof, sky, ground, people, vehicles)
    
    IMPORTANT:
    - Identify all exterior wall surfaces accurately
    - Exclude window frames, glass, doors completely
    - Exclude roof, gutters, and architectural details
    - Keep edges sharp and precise
    - The mask should be the same size as the input image
    
    OUTPUT: Return ONLY the binary mask image (no text, no explanations)
    """
    
    mask_response_parts = generate_multimodal_response(mask_prompt, [wall_img])

    mask_image = None
    for part_type, content in mask_response_parts:
        if part_type == 'image':
            mask_image = content
            break
    
    if not mask_image:
        text_errors = [content for part_type, content in mask_response_parts if part_type == 'text']
        if text_errors:
            return {"error": f"Could not generate mask: {'. '.join(text_errors)}"}
        return {"error": "Could not generate mask."}

    # ========== STEP 2: Apply ACP Sheet (IMPROVED PROMPT) ==========
    apply_prompt = """
    TASK: Apply the ACP (Aluminum Composite Panel) texture to the building wall using the provided mask.
    
    YOU ARE GIVEN:
    1. Original building image
    2. ACP sheet texture/pattern
    3. Binary mask (white = walls to cover, black = preserve original)
    
    INSTRUCTIONS:
    - Apply the ACP sheet texture ONLY to the WHITE areas of the mask
    - Maintain realistic perspective and depth of the original building
    - Blend the texture naturally with lighting conditions
    - Preserve shadows, highlights, and architectural features
    - Keep all BLACK mask areas completely unchanged (windows, doors, background)
    - Ensure seamless edges where ACP meets windows/doors
    - The result should look photorealistic and professionally rendered
    
    OUTPUT: Return ONLY the final composite image showing the building with ACP applied (no text, no annotations)
    """

    response_parts = generate_multimodal_response(apply_prompt, [wall_img, acp_img, mask_image])

    for part_type, content in response_parts:
        if part_type == 'image':
            img_byte_arr = io.BytesIO()
            content.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            return StreamingResponse(img_byte_arr, media_type="image/png")

    text_errors = [content for part_type, content in response_parts if part_type == 'text']
    if text_errors:
        return {"error": f"Could not generate final image: {'. '.join(text_errors)}"}
        
    return {"error": "Could not generate the final image."}


@app.post("/apply_acp_single")
async def apply_acp_single_prompt(wall_image: UploadFile = File(...), acp_sheet_image: UploadFile = File(...)):
    """
    SINGLE-STEP APPROACH (Faster but less precise):
    Applies the ACP sheet directly using a single comprehensive prompt.
    
    Use this if speed is more important than precision.
    """
    try:
        wall_img = Image.open(wall_image.file).convert('RGB')
        acp_img = Image.open(acp_sheet_image.file).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # ========== SINGLE COMPREHENSIVE PROMPT (IMPROVED) ==========
    single_prompt = """
    TASK: Apply the provided ACP (Aluminum Composite Panel) sheet texture to the building walls in the first image.
    
    STEP-BY-STEP PROCESS:
    1. Analyze the building image and identify all wall surfaces
    2. Exclude windows, doors, glass surfaces, roof, and surroundings
    3. Apply the ACP texture from the second image to identified wall surfaces only
    4. Maintain perspective, depth, and lighting of the original building
    5. Blend naturally with architectural features
    
    CRITICAL REQUIREMENTS:
    ✓ Apply ACP ONLY to exterior wall surfaces
    ✓ Preserve ALL windows (frames and glass) in their original state
    ✓ Preserve ALL doors in their original state
    ✓ Keep background (sky, trees, ground, vehicles, people) completely unchanged
    ✓ Maintain realistic shadows and highlights
    ✓ Ensure the result looks photorealistic and professionally rendered
    ✓ Keep proper perspective distortion matching the building's angle
    
    IMPORTANT:
    - The ACP texture should conform to the building's geometry
    - Edge transitions between ACP and windows/doors must be clean
    - Lighting on the ACP should match the original image's lighting conditions
    
    OUTPUT: Return a single composite image showing the building with ACP sheet applied to walls only.
    """

    response_parts = generate_multimodal_response(single_prompt, [wall_img, acp_img])

    for part_type, content in response_parts:
        if part_type == 'image':
            img_byte_arr = io.BytesIO()
            content.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            return StreamingResponse(img_byte_arr, media_type="image/png")

    text_errors = [content for part_type, content in response_parts if part_type == 'text']
    if text_errors:
        return {"error": f"Could not generate image: {'. '.join(text_errors)}"}
        
    return {"error": "Could not generate the image."}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    print("🚀 Starting FastAPI server...")
    print(f"📍 Port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)