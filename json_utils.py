"""
json_utils.py
Utility functions for structured JSON extraction and schema mapping
from industrial machinery documents (e.g., Alibaba product listings).
"""
import json
import re
import logging
import jsonschema
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# At the top of json_utils.py
from copy import deepcopy

DEFAULT_FULL_SCHEMA_BASE = {
    "IndustryType": {"id": None, "name": "", "description": ""},
    "PlantCapacity": {"id": None, "industry_type_id": None, "capacity_label": "", "description": ""},
    "ProcessFlow": [],
    "Product": {"id": None, "industry_type_id": None, "name": "", "description": "", "image_url": ""},
    "ProductRecipe": [],
    "Machine": {
        "id": None,
        "process_flow_id": None,
        "industry_type_id": None,
        "name": "",
        "type": "",
        "model_number": "",
        "output_capacity_kg_hr": None,
        "power_kw": None,
        "dimensions_mm": "",
        "weight_kg": None,
        "automation_level": "",
        "price_usd": None,
        "material": "",
        "control_type": "",
        "image_url": "",
        "video_url": "",
        "spec_sheet_pdf_url": "",
        "description": ""
    },
    "Manufacturer": {
        "id": None,
        "name": "",
        "country": "",
        "website": "",
        "email": "",
        "phone": "",
        "address": "",
        "logo_url": ""
    },
    "MachineManufacturerMap": {
        "id": None,
        "machine_id": None,
        "manufacturer_id": None
    }
}

# Extend for full system schema (optional)
DEFAULT_FULL_SCHEMA = deepcopy(DEFAULT_FULL_SCHEMA_BASE)
DEFAULT_FULL_SCHEMA.update({
    "Inquiry": {
        "id": None,
        "machine_id": None,
        "user_name": "",
        "user_email": "",
        "user_phone": "",
        "user_message": "",
        "inquiry_status": "",
        "timestamp": ""
    },
    "User": {
        "id": None,
        "name": "",
        "email": "",
        "password_hash": "",
        "role": "",
        "industry_focus": "",
        "created_at": ""
    },
    "SearchHistory": {
        "id": None,
        "user_id": None,
        "search_term": "",
        "filters_applied": "",
        "timestamp": ""
    },
    "ComparisonList": {
        "id": None,
        "user_id": None,
        "machine_id": None
    },
    "AIRecommendations": {
        "id": None,
        "user_id": None,
        "industry_type_id": None,
        "input_params": "",
        "recommended_machine_ids": "",
        "timestamp": ""
    }
})

# --- JSON SCHEMA FOR VALIDATION ---
MACHINE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "IndustryType": {
            "type": "object",
            "properties": {
                "id": {"type": ["integer", "null"]},
                "name": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["id", "name", "description"]
        },
        "PlantCapacity": {
            "type": "object",
            "properties": {
                "id": {"type": ["integer", "null"]},
                "industry_type_id": {"type": ["integer", "null"]},
                "capacity_label": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["id", "industry_type_id", "capacity_label", "description"]
        },
        "ProcessFlow": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": ["integer", "null"]},
                    "plant_capacity_id": {"type": ["integer", "null"]},
                    "step_number": {"type": ["integer", "null"]},
                    "step_name": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["id", "step_name"]
            }
        },
        "Product": {
            "type": "object",
            "properties": {
                "id": {"type": ["integer", "null"]},
                "industry_type_id": {"type": ["integer", "null"]},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "image_url": {"type": "string"}
            },
            "required": ["id", "name", "image_url"]
        },
        "ProductRecipe": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": ["integer", "null"]},
                    "product_id": {"type": ["integer", "null"]},
                    "step_number": {"type": ["integer", "null"]},
                    "ingredient": {"type": "string"},
                    "quantity": {"type": "string"},
                    "unit": {"type": "string"},
                    "process_flow_step_id": {"type": ["integer", "null"]}
                },
                "required": ["id", "ingredient"]
            }
        },
        "Machine": {
            "type": "object",
            "properties": {
                "id": {"type": ["integer", "null"]},
                "process_flow_id": {"type": ["integer", "null"]},
                "industry_type_id": {"type": ["integer", "null"]},
                "name": {"type": "string"},
                "type": {"type": "string"},
                "model_number": {"type": "string"},
                "output_capacity_kg_hr": {"type": ["number", "null"]},
                "power_kw": {"type": ["number", "null"]},
                "dimensions_mm": {"type": "string"},
                "weight_kg": {"type": ["number", "null"]},
                "automation_level": {"type": "string"},
                "price_usd": {"type": ["number", "null"]},
                "material": {"type": "string"},
                "control_type": {"type": "string"},
                "image_url": {"type": "string"},
                "video_url": {"type": "string"},
                "spec_sheet_pdf_url": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["id", "name"]
        },
        "Manufacturer": {
            "type": "object",
            "properties": {
                "id": {"type": ["integer", "null"]},
                "name": {"type": "string"},
                "country": {"type": "string"},
                "website": {"type": "string"},
                "email": {"type": "string"},
                "phone": {"type": "string"},
                "address": {"type": "string"},
                "logo_url": {"type": "string"}
            },
            "required": ["id", "name"]
        }
    },
    "required": ["IndustryType", "PlantCapacity", "ProcessFlow", "Product", "ProductRecipe", "Machine", "Manufacturer"]
}

def validate_json_structure(json_data: Dict) -> (bool, str):
    """
    Validates a JSON object against the machine schema.
    Returns (is_valid, error_message)
    """
    try:
        jsonschema.validate(instance=json_data, schema=MACHINE_JSON_SCHEMA)
        return True, ""
    except jsonschema.exceptions.ValidationError as e:
        return False, str(e)

# --- DEFAULT SCHEMAS ---
DEFAULT_JSON_STRUCTURE = {
    "IndustryType": {"id": None, "name": "", "description": ""},
    "PlantCapacity": {"id": None, "industry_type_id": None, "capacity_label": "", "description": ""},
    "ProcessFlow": [],
    "Product": {"id": None, "industry_type_id": None, "name": "", "description": "", "image_url": ""},
    "ProductRecipe": []
}


# --- HELPER FUNCTIONS ---
def get_metadata_value(metadata: Dict, candidates: List[str]) -> str:
    """
    Retrieve value from metadata using case-insensitive key matching.
    """
    if not isinstance(metadata, dict):
        return ""
    for key in metadata:
        key_str = str(key).lower()
        if any(candidate.lower() in key_str for candidate in candidates):
            value = metadata[key]
            return str(value).strip() if value is not None else ""

    return ""


def extract_process_flow(product: Dict) -> List[Dict]:
    """
    Extract process steps from tables or text.
    Returns a list of process steps.
    """
    flow = []
    # From tables
    for table in product.get("tables", []):
        if isinstance(table, dict):
            for col_name, cell_value in table.items():
                if "process" in col_name.lower() and isinstance(cell_value, str) and cell_value.strip():
                    flow.append({
                        "id": len(flow) + 1,
                        "plant_capacity_id": 1,
                        "step_number": len(flow) + 1,
                        "step_name": cell_value.strip(),
                        "description": ""
                    })
    # From text
    text = product.get("text", "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    process_indicators = ["step", "phase", "stage", "process"]
    for line in lines:
        if any(indicator in line.lower() for indicator in process_indicators) and len(line) < 100:
            flow.append({
                "id": len(flow) + 1,
                "plant_capacity_id": 1,
                "step_number": len(flow) + 1,
                "step_name": line,
                "description": ""
            })
    return flow

def format_machine_hierarchy_report(grouped_machines, consolidated_list, doc_name=""):
    lines = []
    if doc_name:
        lines.append(f"📋 Machine Hierarchy Report - {doc_name}\n")

    lines.append("🔗 Grouped Machine Hierarchy (from product context):")
    for idx, group in enumerate(grouped_machines, 1):
        if not group["main_machine"]:
            continue
        lines.append(f"✅ {idx}. {group['main_machine']} (Supplier: {group['supplier']})")
        if group["description"]:
            lines.append(f"   {group['description']}")
        if group["sub_machines"]:
            lines.append("   Sub-Machines/Components:")
            for sm in group["sub_machines"]:
                lines.append(f"    - {sm}")
        if group["related_machines"]:
            lines.append("   Related Machines:")
            for rm in group["related_machines"]:
                lines.append(f"    - {rm}")
        lines.append("")

    lines.append("🔚 Final Consolidated List of All Machines (Alphabetical Order)")
    for item in consolidated_list:
        if item["description"]:
            lines.append(f"{item['machine']} - {item['description']}")
        else:
            lines.append(f"{item['machine']}")
    return "\n".join(lines)

def extract_recipe_from_tables(tables: List[Dict]) -> List[Dict]:
    """
    Extract recipe from tables containing ingredient info.
    """
    recipe = []
    for table in tables:
        if isinstance(table, dict):
            ingredient = get_metadata_value(table, ["ingredient", "material", "component"])
            quantity = get_metadata_value(table, ["quantity", "amount"])
            unit = get_metadata_value(table, ["unit", "uom"])
            if ingredient:
                recipe.append({
                    "id": len(recipe) + 1,
                    "product_id": 1,
                    "step_number": len(recipe) + 1,
                    "ingredient": ingredient,
                    "quantity": quantity,
                    "unit": unit,
                    "process_flow_step_id": None
                })
    return recipe


# --- MAIN MAPPING FUNCTIONS ---
def map_product_to_schema(product: Dict, user_query: Optional[str] = None) -> Dict:
    """
    Map a product dict to the simplified JSON schema.
    """
    result = json.loads(json.dumps(DEFAULT_JSON_STRUCTURE))  # deep copy
    metadata = product.get("metadata", {})
    raw_text = product.get("text", "")
    images = product.get("images", [])
    tables = product.get("tables", [])

    # Product Name
    product_name = get_metadata_value(metadata, [
        "Name of machine", "Product name", "Title", "product name", "machine"
    ])
    if not product_name:
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        for line in lines:
            if len(line.split()) >= 3 and len(line) <= 100 and re.search(r'\b(machine|line|system)\b', line.lower()):
                product_name = line
                break
    result["Product"]["name"] = product_name

    # Description
    description = get_metadata_value(metadata, ["Description", "Function", "Product Description"])
    if not description and product_name:
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        try:
            idx = next(i for i, l in enumerate(lines) if product_name in l)
            if idx + 1 < len(lines):
                next_line = lines[idx + 1]
                if len(next_line.split()) > 4:
                    description = next_line
        except StopIteration:
            pass
    result["Product"]["description"] = description or raw_text[:300].strip()

    # Capacity
    capacity = get_metadata_value(metadata, ["Capacity", "machinery capacity", "output"])
    if not capacity:
        for table in tables:
            if isinstance(table, dict):
                capacity = get_metadata_value(table, ["Capacity"])
                if capacity:
                    break
    if not capacity:
        match = re.search(r"capacity[:\s]*([0-9]+\.?[0-9]*\s*[kK][gG]\/[hHrR]+)", raw_text)
        if match:
            capacity = match.group(1).strip()
    result["PlantCapacity"]["capacity_label"] = capacity

    # Industry Type
    combined_text = f"{product_name} {description}".lower()
    industry_map = {
        "chocolate": "Chocolate",
        "confectionery": "Chocolate",
        "candy": "Chocolate",
        "food": "Food Processing",
        "snack": "Snack Production",
        "beverage": "Beverage Processing",
        "bakery": "Bakery"
    }
    for keyword, industry in industry_map.items():
        if keyword in combined_text:
            result["IndustryType"]["name"] = industry
            break

    # Image URL
    if images:
        img = images[0]
        result["Product"]["image_url"] = img if isinstance(img, str) else img.get("path", "")

    # Process Flow
    result["ProcessFlow"] = extract_process_flow(product)

    return result


def map_document_to_full_schema(product: Dict, user_query: Optional[str] = None) -> Dict:
    """
    Map a product/document dict to the full relational schema.
    Always includes the original text as 'text'.
    Ensures all fields are present and correctly typed.
    """
    result = json.loads(json.dumps(DEFAULT_FULL_SCHEMA))  # deep copy
    try:
        if not isinstance(product, dict):
            logger.warning("Invalid product: not a dict")
            return result

        metadata = product.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        raw_text = product.get("text", "")
        images = product.get("images", [])
        tables = product.get("tables", [])

        result['text'] = raw_text  # Always include original text

        # --- IndustryType ---
        result["IndustryType"]["name"] = get_metadata_value(metadata, ["Industry", "Category"])
        result["IndustryType"]["description"] = get_metadata_value(metadata, ["Industry Description"])

        # --- PlantCapacity ---
        result["PlantCapacity"]["capacity_label"] = get_metadata_value(metadata, ["Capacity", "Output"])
        result["PlantCapacity"]["description"] = get_metadata_value(metadata, ["Capacity Description"])

        # --- Product ---
        product_name = get_metadata_value(metadata, ["Product Name", "Name of machine", "Title"])
        if not product_name:
            lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
            for line in lines:
                if len(line.split()) > 2 and line[0].isupper() and any(kw in line.lower() for kw in ["machine", "line"]):
                    product_name = line
                    break
        result["Product"]["name"] = product_name
        result["Product"]["description"] = get_metadata_value(metadata, ["Product Description"]) or raw_text[:300]
        if images:
            img = images[0]
            result["Product"]["image_url"] = img if isinstance(img, str) else img.get("path", "")

        # --- Machine ---
        result["Machine"]["name"] = get_metadata_value(metadata, ["Machine Name"]) or result["Product"]["name"]
        result["Machine"]["type"] = get_metadata_value(metadata, ["Machine Type", "Type"])
        result["Machine"]["model_number"] = get_metadata_value(metadata, ["Model Number", "Model"])
        try:
            result["Machine"]["output_capacity_kg_hr"] = float(metadata.get("Output Capacity (kg/hr)", 0)) or None
        except (ValueError, TypeError):
            result["Machine"]["output_capacity_kg_hr"] = None
        try:
            result["Machine"]["power_kw"] = float(metadata.get("Power (kW)", 0)) or None
        except (ValueError, TypeError):
            result["Machine"]["power_kw"] = None
        result["Machine"]["dimensions_mm"] = get_metadata_value(metadata, ["Dimensions (mm)", "Size"])
        try:
            result["Machine"]["weight_kg"] = float(metadata.get("Weight (kg)", 0)) or None
        except (ValueError, TypeError):
            result["Machine"]["weight_kg"] = None
        result["Machine"]["automation_level"] = get_metadata_value(metadata, ["Automation Level"])
        try:
            result["Machine"]["price_usd"] = float(metadata.get("Price (USD)", 0)) or None
        except (ValueError, TypeError):
            result["Machine"]["price_usd"] = None
        result["Machine"]["material"] = get_metadata_value(metadata, ["Material"])
        result["Machine"]["control_type"] = get_metadata_value(metadata, ["Control Type"])
        result["Machine"]["image_url"] = result["Product"]["image_url"]
        result["Machine"]["description"] = get_metadata_value(metadata, ["Machine Description"]) or result["Product"]["description"]

        # --- Manufacturer ---
        result["Manufacturer"]["name"] = get_metadata_value(metadata, ["Manufacturer", "Company", "Supplier"])
        result["Manufacturer"]["country"] = get_metadata_value(metadata, ["Country", "Origin"])
        result["Manufacturer"]["website"] = get_metadata_value(metadata, ["Website", "URL"])
        result["Manufacturer"]["email"] = get_metadata_value(metadata, ["Email"])
        result["Manufacturer"]["phone"] = get_metadata_value(metadata, ["Phone"])
        result["Manufacturer"]["address"] = get_metadata_value(metadata, ["Address"])
        result["Manufacturer"]["logo_url"] = get_metadata_value(metadata, ["Logo URL"])

        # --- MachineManufacturerMap ---
        if result["Machine"]["name"] and result["Manufacturer"]["name"]:
            result["MachineManufacturerMap"]["machine_id"] = 1
            result["MachineManufacturerMap"]["manufacturer_id"] = 1

        # --- ProcessFlow ---
        process_flow = None
        for idx, table in enumerate(tables):
            if isinstance(table, dict) and any("process" in k.lower() for k in table.keys()):
                process_flow = {
                    "id": idx + 1,
                    "plant_capacity_id": 1,
                    "step_number": idx + 1,
                    "step_name": str(table.get("Process Step", "")),
                    "description": str(table.get("Description", ""))
                }
                break
        result["ProcessFlow"] = process_flow or {
            "id": None, "plant_capacity_id": None, "step_number": None, "step_name": "", "description": ""
        }

        # --- ProductRecipe ---
        product_recipe = None
        for idx, table in enumerate(tables):
            if isinstance(table, dict) and any("ingredient" in k.lower() for k in table.keys()):
                product_recipe = {
                    "id": idx + 1,
                    "product_id": 1,
                    "step_number": idx + 1,
                    "ingredient": str(table.get("Ingredient", "")),
                    "quantity": str(table.get("Quantity", "")),
                    "unit": str(table.get("Unit", "")),
                    "process_flow_step_id": None
                }
                break
        result["ProductRecipe"] = product_recipe or {
            "id": None, "product_id": None, "step_number": None, "ingredient": "", "quantity": "", "unit": "", "process_flow_step_id": None
        }

        return result

    except Exception as e:
        logger.warning(f"map_document_to_full_schema: error extracting schema: {e}")
        return result


# --- STRUCTURED EXTRACTION ---
def extract_and_validate_json(response: Any) -> Dict:
    """
    Parses and normalizes the LLM JSON output.
    """
    try:
        if isinstance(response, str):
            data = json.loads(response)
        else:
            data = response
        # Normalize keys if needed
        if isinstance(data, dict):
            data = normalize_keys(data)
        return data
    except Exception as e:
        logger.error(f"Failed to parse/normalize JSON: {e}")
        return {"error": str(e), "raw": str(response)[:200]}


def normalize_keys(data: Any) -> Any:
    """Recursively normalizes all keys in nested dictionaries."""
    if isinstance(data, dict):
        return {
            re.sub(r'[\s\-()]+', '_', k.strip().lower()).replace('__', '_').rstrip('_'): normalize_keys(v)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [normalize_keys(item) for item in data]
    else:
        return data
# --- Add this function to json_utils.py ---
def merge_with_default(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges parsed JSON data with DEFAULT_FULL_SCHEMA.
    Ensures arrays are lists, objects are dicts, scalars use null/"".
    """
    from copy import deepcopy

    merged = deepcopy(DEFAULT_FULL_SCHEMA)

    def recursive_merge(base, update):
        if isinstance(base, list):
            # If schema expects a list, ensure update is a list
            if isinstance(update, dict):
                # Wrap single dict in list if it's meant to be an array of objects
                return [update]  # Most common case: user gave one recipe instead of [recipe]
            elif isinstance(update, list):
                return update
            else:
                return []
        elif isinstance(base, dict):
            if not isinstance(update, dict):
                return base  # Keep default if update is invalid
            for key, value in update.items():
                if key in base:
                    if isinstance(base[key], list):
                        # Always replace list with list (or wrap dict in list)
                        if isinstance(value, dict):
                            base[key] = [value]
                        elif isinstance(value, list):
                            base[key] = value
                        else:
                            base[key] = []
                    elif isinstance(base[key], dict):
                        recursive_merge(base[key], value)
                    else:
                        # Scalar: string, number, null
                        base[key] = value
            return base
        else:
            # Base is scalar
            return update if update is not None else base

    recursive_merge(merged, parsed_data)
    return merged

def extract_structured_json_from_docx(docx_path: str, user_query: Optional[str] = None) -> List[Dict]:
    """
    Extract structured JSON from a DOCX file.
    """
    try:
        from document_processing import extract_products_from_docx
        products = extract_products_from_docx(docx_path)
        results = [map_product_to_schema(p, user_query=user_query) for p in products]
        return results
    except Exception as e:
        logger.error(f"Error extracting structured JSON from {docx_path}: {e}")
        return []


# --- USER INTERFACE UTILS ---
def get_user_input() -> str:
    try:
        return input("\n💬 You: ").strip()
    except EOFError:
        return ""


def handle_command(user_input: str, chains: dict, retriever, session_id: str) -> bool:
    """
    Handles custom command prefixes like 'ocr:', 'json:', etc.
    Returns True if command was handled, False otherwise.
    """
    if user_input.lower().startswith("json:"):
        query = user_input[5:].strip()
        print("\n🔄 Processing JSON query...")
        response = chains["json"].invoke({"query": query}).get("result", "")
        print(f"\n🧾 JSON:\n{response}")
        return True
    elif user_input.lower().startswith("jsonreverse:"):
        query = user_input[len("jsonreverse:"):].strip()
        print("\n🔎 Performing reverse engineering with JSON output...")
        response = chains["jsonreverse"].invoke({"query": query}).get("result", "")
        print(f"\n🧩 Reverse Engineering JSON:\n{response}")
        return True
    elif user_input.lower().startswith("reverse:"):
        query = user_input[len("reverse:"):].strip()
        print("\n🧠 Performing reverse engineering analysis...")
        response = chains["reverse"].invoke({"query": query}).get("result", "")
        print(f"\n🧠 Analysis:\n{response}")
        return True
    return False