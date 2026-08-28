import json
from pathlib import Path
from typing import Dict, Any

class BFRIRagEngine:
    def __init__(self, doc_path: str = "datasets/bfri_docs/bfri_guidelines.json"):
        self.doc_path = Path(doc_path)
        self.knowledge_base = self._load_docs()

    def _load_docs(self) -> Dict[str, Any]:
        if not self.doc_path.exists():
            return {}
        # Fixed: utf-8-sig strips Windows PowerShell BOM headers automatically
        with open(self.doc_path, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    def retrieve_treatment(self, disease_name: str) -> Dict[str, Any]:
        treatment = self.knowledge_base.get(disease_name)
        if treatment:
            return {
                "found": True,
                "disease": disease_name,
                "protocol": treatment
            }
        return {
            "found": False,
            "disease": disease_name,
            "protocol": {
                "symptoms": "General aquatic stress symptoms detected.",
                "bfri_treatment": "Consult local BFRI extension officer or district fisheries officer.",
                "water_remediation": "Perform partial water exchange and test pH, DO, and ammonia."
            }
        }

if __name__ == "__main__":
    rag = BFRIRagEngine()
    res = rag.retrieve_treatment("EUS")
    print("\n--- BFRI RAG Query Result (EUS) ---")
    print(json.dumps(res, indent=2))
