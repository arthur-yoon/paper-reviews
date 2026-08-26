import re
from enum import Enum
from typing import List, Dict, Set, Optional
import numpy as np

class ConstraintType(Enum):
    TYPE = "type"
    RELATION = "relation"
    EXCLUSION = "exclusion"

class VerificationStatus(Enum):
    SATISFIED = "Satisfied"
    VIOLATED = "Violated"
    UNKNOWN = "Unknown"

class KnowledgeGraph:
    def __init__(self):
        # entity_id -> set of types
        self.entity_types: Dict[str, Set[str]] = {}
        # (entity_a, relation, entity_b) -> True
        self.triples: Set[tuple] = set()

    def add_entity(self, entity_id: str, types: List[str]):
        if entity_id not in self.entity_types:
            self.entity_types[entity_id] = set()
        self.entity_types[entity_id].update(types)

    def add_triple(self, head: str, relation: str, tail: str):
        # Ensure entities exist
        self.add_entity(head, [])
        self.add_entity(tail, [])
        self.triples.add((head, relation, tail))
        
        # For symmetric relations or bidirectional checks in some KGs, 
        # but for this implementation, we treat directed edges.
        # If the relation is symmetric (like 'interacts_with'), one might add both.
        # Here we stick to directed unless specified, but for verification, 
        # we might want to check existence.
        # For 'side_effect', Drug -> SideEffect is standard.

    def get_entity_types(self, entity_id: str) -> Optional[Set[str]]:
        return self.entity_types.get(entity_id, None)

    def has_relation(self, head: str, relation: str, tail: str) -> Optional[bool]:
        """
        Returns True if relation exists, False if relation is known to be absent 
        (in open world, 'False' is tricky. Usually, if not in graph, it's Unknown.
        However, for 'Violated' in type constraints, if entity exists and type is different, it's Violated.
        For relations, if the edge is not present, it is typically 'Unknown' in open world,
        UNLESS we have negative facts. 
        
        The prompt says:
        "Violated: graph data confirms constraint is NOT met (e.g. asking for 'disease' but candidate is 'protein' is confirmed)."
        "Unknown: graph lacks info."
        
        So for relations:
        If edge (A, rel, B) exists -> Satisfied.
        If edge does NOT exist -> Unknown (unless we have explicit negative facts, which standard KGs don't).
        
        For Types:
        If entity is in KG:
           If required type is in entity's types -> Satisfied.
           If required type is NOT in entity's types -> Violated (assuming closed world for types of existing entities? 
           The prompt example says: "Disease_B: graph confirms Disease_B is 'disease' -> Violated (for protein constraint)".
           This implies that if an entity's type is explicitly known to be X, and we ask for Y (Y!=X), it's Violated.
           If entity type is not specified at all -> Unknown.
        """
        return (head, relation, tail) in self.triples

class Constraint:
    def __init__(self, ctype: ConstraintType, value: str, source_entity: Optional[str] = None):
        self.ctype = ctype
        self.value = value
        self.source_entity = source_entity # For relation constraints, the other end

class ConstraintExtractor:
    """
    Simple rule-based extractor to simulate the 'symbolic parsing' step.
    In a real system, this might use NER or LLM to extract these, but the prompt 
    says "without LLM reasoning process" and "lightweight symbolic constraints".
    We will define a simple parser for specific patterns.
    """
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def extract(self, question: str, llm_candidates: List[str]) -> List[Constraint]:
        constraints = []
        q_lower = question.lower()
        
        # 1. Type Constraints
        # Pattern: "... [noun] of [entity]?" or "What is the [type] of [entity]?"
        # Common biomedical types: protein, disease, drug, side_effect, symptom
        
        # Check for "side effect" or "disease" or "protein" in the question
        type_keywords = {
            'protein': ['protein'],
            'disease': ['disease', 'disorder', 'condition'],
            'drug': ['drug', 'medication'],
            'side_effect': ['side effect', 'adverse effect', 'side_effect']
        }
        
        # We need to identify the target type from the question.
        # Heuristic: Look for keywords in the question.
        detected_type = None
        source_entity = None
        
        # Extract source entity (often the main subject, e.g., "Cytarabine")
        # This is a very rough heuristic. In reality, NER would be used.
        # For the demo, we assume the first capitalized word or specific known entities are the source.
        # Let's look for a known entity in the KG that appears in the question.
        for eid in self.kg.entity_types.keys():
            if eid.lower() in q_lower:
                source_entity = eid
                break
        
        # Determine required type
        for type_name, keywords in type_keywords.items():
            if any(kw in q_lower for kw in keywords):
                detected_type = type_name
                break
                
        if detected_type and source_entity:
            constraints.append(Constraint(ConstraintType.TYPE, detected_type))
            # Relation constraint is often implicit. 
            # "Side effect of X" implies relation 'side_effect' or 'has_side_effect' between X and candidate.
            # Let's assume if 'side effect' is mentioned, we also check for relation.
            if detected_type == 'side_effect':
                constraints.append(Constraint(ConstraintType.RELATION, 'side_effect', source_entity))
            elif detected_type == 'disease':
                # "Disease related to X" -> relation 'associated_with' or 'treats'? 
                # Let's stick to the example: "Protein related to Disease".
                # If asking for disease of protein, relation might be 'causes' or 'associated_with'.
                # For simplicity in this demo, we'll rely on Type primarily unless specific relation is asked.
                pass

        # 2. Exclusion Constraints
        # Pattern: "not [X]", "exclude [X]", "other than [X]"
        exclusion_pattern = r'(?:not|exclude|other than|without)\s+([A-Za-z0-9_]+)'
        exclusions = re.findall(exclusion_pattern, question, re.IGNORECASE)
        for ex in exclusions:
            constraints.append(Constraint(ConstraintType.EXCLUSION, ex))

        return constraints

class ConstraintVerifier:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def verify(self, candidate: str, constraints: List[Constraint]) -> VerificationStatus:
        """
        Implements 3-valued logic.
        If ANY constraint is VIOLATED, the candidate is VIOLATED.
        If NO constraints are VIOLATED, and AT LEAST ONE is SATISFIED, is it Satisfied?
        Or do we need ALL to be satisfied?
        
        The prompt says:
        "Satisfied: graph data confirms constraints are met."
        "Violated: graph data confirms constraints are NOT met."
        
        If I have 2 constraints (Type and Relation):
        If Type is Satisfied and Relation is Unknown -> Overall?
        The prompt example only uses one constraint (Type) in the first example, and Type+Relation in the second.
        In the second example:
        Leukopenia: Confirmed side_effect -> Satisfied.
        
        Let's assume:
        - If any constraint is clearly VIOLATED -> Result is VIOLATED.
        - Else, if any constraint is clearly SATISFIED -> Result is SATISFIED? 
          Or should we be more conservative? 
          "Satisfied" implies positive evidence. 
          If Type is Satisfied but Relation is Unknown, is it Satisfied?
          Usually, in filtering, if one hard constraint fails, it's out. 
          If one passes and others are unknown, it's a candidate.
          
          Let's look at the ranking part: 
          "Satisfied ... higher rank ... Unknown ... lower rank".
          So we can classify a candidate as 'Satisfied' if it meets at least one positive constraint and violates none?
          Or does it need to meet ALL defined constraints?
          
          If I ask "What is the protein X?", and I have Type=Protein and Relation=InteractsWith_Y.
          If Candidate A is Protein (Satisfied) but Relation unknown (Unknown).
          Candidate B is not Protein (Violated).
          
          If I require ALL constraints to be satisfied for 'Satisfied', then A is Unknown.
          If I require ANY constraint to be satisfied (and none violated), then A is Satisfied.
          
          The prompt says: "Satisfied: graph data confirms constraints are met." (plural).
          This suggests ALL constraints should be satisfied.
          However, in Open World, 'Unknown' for one constraint doesn't mean 'Violated'.
          If I can't confirm the relation, I can't say it's 'Satisfied' in the strict sense of "all constraints met".
          So:
          - Any Violated -> VIOLATED
          - All Satisfied -> SATISFIED
          - Else (mix of Satisfied and Unknown, or just Unknown) -> UNKNOWN
          
          Let's implement:
          1. Check all constraints.
          2. If any is VIOLATED, return VIOLATED.
          3. If all are SATISFIED, return SATISFIED.
          4. Otherwise (at least one UNKNOWN, none VIOLATED), return UNKNOWN.
        """
        statuses = []
        for c in constraints:
            status = self._verify_single_constraint(candidate, c)
            statuses.append(status)
            
            # Short circuit: if any is violated, the whole thing is violated.
            if status == VerificationStatus.VIOLATED:
                return VerificationStatus.VIOLATED
        
        # No violations.
        if all(s == VerificationStatus.SATISFIED for s in statuses):
            return VerificationStatus.SATISFIED
        else:
            return VerificationStatus.UNKNOWN

    def _verify_single_constraint(self, candidate: str, constraint: Constraint) -> VerificationStatus:
        if constraint.ctype == ConstraintType.TYPE:
            entity_types = self.kg.get_entity_types(candidate)
            if entity_types is None:
                # Entity not in KG at all -> Unknown
                return VerificationStatus.UNKNOWN
            # Entity is in KG.
            if constraint.value in entity_types:
                return VerificationStatus.SATISFIED
            else:
                # Entity is in KG, but does not have the required type.
                # In many KGs, if type is not listed, it's unknown. 
                # But the prompt example says: "Disease_B is 'disease' -> Violated (for protein constraint)".
                # This implies if we KNOW it's a Disease, and we ask for Protein, it's Violated.
                # What if it has no types listed? Unknown.
                # What if it has types ['disease'] and we ask for 'protein'? Violated.
                # What if it has types ['protein', 'disease']? Satisfied.
                # So, if the specific type is missing, is it Violated or Unknown?
                # If the entity has a definitive type list that excludes the target, it's Violated.
                # If the entity has no type info, it's Unknown.
                # If the entity has SOME types, but not the target one? 
                # Usually, if a type is not asserted, it's not known to be absent (Open World).
                # HOWEVER, the prompt's "Violated" example relies on the fact that it IS a Disease.
                # So, if 'protein' is not in the set of known types, is it Violated?
                # If the KG is closed-world for types (i.e., if it's a protein, it's labeled), then missing label = not protein.
                # Given the context of "Lightweight symbolic constraints", assuming that if an entity has types defined, 
                # and the target type is not among them, it's a violation is a common heuristic in such filtering tasks 
                # to ensure precision. If we treat it as Unknown, we lose the filtering power for "Type" constraints 
                # unless we have explicit negative facts.
                # Let's assume: If entity types are known (non-empty) and target is not in them -> VIOLATED.
                # If entity types are empty (or unknown) -> UNKNOWN.
                
                if len(entity_types) > 0:
                    return VerificationStatus.VIOLATED
                else:
                    return VerificationStatus.UNKNOWN

        elif constraint.ctype == ConstraintType.RELATION:
            # Check if (source, relation, candidate) or (candidate, relation, source) exists?
            # Usually directed. Let's check (source -> candidate) and (candidate -> source) to be safe?
            # The prompt example: "Cytarabine" -> 'side_effect' -> 'Leukopenia'.
            # So source is Cytarabine, candidate is Leukopenia.
            # We check (source, rel, candidate).
            
            head = constraint.source_entity
            tail = candidate
            rel = constraint.value
            
            # Check both directions just in case the KG is undirected or symmetric in nature,
            # but for "side_effect", it's directional.
            if self.kg.has_relation(head, rel, tail):
                return VerificationStatus.SATISFIED
            
            # Check reverse?
            if self.kg.has_relation(tail, rel, head):
                return VerificationStatus.SATISFIED
            
            # If not found, it's UNKNOWN in Open World.
            # Unless we have specific logic for "negative facts", which we don't.
            return VerificationStatus.UNKNOWN

        elif constraint.ctype == ConstraintType.EXCLUSION:
            if candidate.lower() == constraint.value.lower():
                return VerificationStatus.VIOLATED
            else:
                # If it's not the excluded entity, the constraint is satisfied (it passes the exclusion).
                # Or is it Unknown? Exclusion is a hard filter. If it's not the excluded one, it's valid.
                return VerificationStatus.SATISFIED

        return VerificationStatus.UNKNOWN

def ces_pk_framework(kg: KnowledgeGraph, question: str, llm_candidates: List[str]) -> List[Dict]:
    extractor = ConstraintExtractor(kg)
    verifier = ConstraintVerifier(kg)
    
    constraints = extractor.extract(question, llm_candidates)
    
    results = []
    for candidate in llm_candidates:
        status = verifier.verify(candidate, constraints)
        results.append({
            "candidate": candidate,
            "status": status
        })
    
    # Ranking:
    # Satisfied: Highest score
    # Unknown: Middle score
    # Violated: Removed (or lowest score, but prompt says "Remove")
    
    # Filter out Violated
    filtered_results = [r for r in results if r["status"] != VerificationStatus.VIOLATED]
    
    # Assign scores for ranking
    score_map = {
        VerificationStatus.SATISFIED: 2.0,
        VerificationStatus.UNKNOWN: 1.0
    }
    
    for r in filtered_results:
        r["score"] = score_map.get(r["status"], 0.0)
        
    # Sort by score descending
    filtered_results.sort(key=lambda x: x["score"], reverse=True)
    
    return filtered_results

def main():
    # 1. Data Preparation: Load Hetionet-like data
    kg = KnowledgeGraph()
    
    # Entities
    kg.add_entity("Cytarabine", ["drug"])
    kg.add_entity("Leukopenia", ["side_effect", "disease"]) # Leukopenia is both a side effect and a disease condition
    kg.add_entity("Nausea", ["side_effect", "disease"])
    kg.add_entity("Hepatotoxicity", ["side_effect", "disease"])
    kg.add_entity("Protein_X", ["protein"])
    kg.add_entity("Unknown_D", []) # No types defined
    
    # Relations (Triples)
    kg.add_triple("Cytarabine", "side_effect", "Leukopenia")
    kg.add_triple("Cytarabine", "side_effect", "Nausea")
    # Note: Hepatotoxicity is NOT linked to Cytarabine in this mock KG.
    # Maybe linked to another drug?
    kg.add_triple("Some_Other_Drug", "side_effect", "Hepatotoxicity")
    
    # 2. Question & LLM Candidates
    question = "What is the side effect of Cytarabine?"
    # Simulated LLM output
    llm_candidates = ["Leukopenia", "Nausea", "Hepatotoxicity", "Protein_X", "Unknown_D"]
    
    # 3. Execution
    print(f"Question: {question}")
    print(f"LLM Candidates: {llm_candidates}")
    print("-" * 30)
    
    # Extract constraints for debugging
    extractor = ConstraintExtractor(kg)
    constraints = extractor.extract(question, llm_candidates)
    print("Extracted Constraints:")
    for c in constraints:
        print(f"  - {c.ctype.value}: {c.value} (Source: {c.source_entity})")
    print("-" * 30)
    
    results = ces_pk_framework(kg, question, llm_candidates)
    
    print("Final Ranked Results:")
    for res in results:
        print(f"  {res['candidate']}: {res['status'].value} (Score: {res['score']})")
        
    # Verify expected behavior:
    # Leukopenia: Type side_effect (Satisfied), Relation side_effect from Cytarabine (Satisfied) -> Satisfied
    # Nausea: Type side_effect (Satisfied), Relation side_effect from Cytarabine (Satisfied) -> Satisfied
    # Hepatotoxicity: Type side_effect (Satisfied), Relation side_effect from Cytarabine (Unknown - not in graph) -> Unknown
    # Protein_X: Type protein (Violated for side_effect constraint? 
    #            Wait, my extractor extracted Type="side_effect". 
    #            Protein_X types: ["protein"]. "side_effect" is not in ["protein"]. 
    #            So Type constraint is Violated. -> Removed.
    # Unknown_D: No types. Type constraint -> Unknown. Relation constraint -> Unknown. -> Unknown.

if __name__ == "__main__":
    main()
