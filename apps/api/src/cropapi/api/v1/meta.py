from fastapi import APIRouter

router = APIRouter()

# Stubbed label space. Phase 2.6 sources this from the loaded model's
# artifact manifest instead of a hardcoded list.
_SUPPORTED_CROPS = ["tomato", "potato", "corn", "apple", "grape"]
_SUPPORTED_DISEASES = [
    {"crop": "tomato", "disease": "early_blight"},
    {"crop": "tomato", "disease": "late_blight"},
    {"crop": "potato", "disease": "late_blight"},
    {"crop": "corn", "disease": "common_rust"},
]


@router.get("/meta")
async def get_meta() -> dict:
    return {
        "model_version": "cropnet-v0.3.1",
        "supported_crops": _SUPPORTED_CROPS,
        "supported_diseases": _SUPPORTED_DISEASES,
    }