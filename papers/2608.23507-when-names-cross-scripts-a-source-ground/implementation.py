import re
import string
from typing import List, Dict, Any, Tuple, Optional

class PersonRecord:
    def __init__(self, id: str, name_variants: List[str], source_evidence: List[str]):
        self.id = id
        self.name_variants = name_variants
        self.source_evidence = source_evidence

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def normalize_text(text: str) -> str:
    text = text.lower()
    text = ''.join(c for c in text if c.isalnum() or c in ' \n')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Simple transliteration table for demonstration
TRANSLIT_TABLE = {
    "temujin": "genghis khan",
    "chinghis khan": "genghis khan",
    "tamurlan": "tamerlane",
    "timur": "tamerlane",
    "mongke": "mongke", # Self-mapping, handled by other logic
    "batu": "batu"
}

def extract_years(text: str) -> List[int]:
    years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', text)
    return [int(y) for y in years]

def check_name_similarity(record_a: PersonRecord, record_b: PersonRecord, threshold: int = 3) -> bool:
    for name_a in record_a.name_variants:
        name_a_norm = normalize_text(name_a)
        for name_b in record_b.name_variants:
            name_b_norm = normalize_text(name_b)
            
            # Exact match
            if name_a_norm == name_b_norm:
                return True
                
            # Levenshtein distance
            if levenshtein_distance(name_a_norm, name_b_norm) <= threshold:
                return True
                
            # Transliteration match
            if name_a_norm in TRANSLIT_TABLE and name_b_norm in TRANSLIT_TABLE:
                if TRANSLIT_TABLE[name_a_norm] == TRANSLIT_TABLE[name_b_norm]:
                    return True
            elif name_a_norm in TRANSLIT_TABLE and TRANSLIT_TABLE[name_a_norm] == name_b_norm:
                return True
            elif name_b_norm in TRANSLIT_TABLE and name_b_norm == name_a_norm:
                return True
    return False

def analyze_source_conflicts(source_a: List[str], source_b: List[str]) -> Tuple[str, List[str]]:
    """
    Returns (status, reasons) where status is 'CONFLICT', 'SUPPORT', 'NEUTRAL', or 'INSUFFICIENT'
    """
    text_a = " ".join(source_a).lower()
    text_b = " ".join(source_b).lower()
    
    years_a = extract_years(text_a)
    years_b = extract_years(text_b)
    
    conflict_reasons = []
    support_reasons = []
    
    # Check for explicit death/birth year conflicts
    # Simple heuristic: If both have dates and they are far apart, it's a conflict
    if years_a and years_b:
        min_a = min(years_a)
        max_a = max(years_a)
        min_b = min(years_b)
        max_b = max(years_b)
        
        # If ranges don't overlap and are significantly different (e.g., > 30 years apart)
        if (max_a < min_b - 30) or (max_b < min_a - 30):
            conflict_reasons.append(f"Temporal context contradiction: {min_a}-{max_a} vs {min_b}-{max_b}")
            
    # Check for specific role/title contradictions if dates are not conclusive or as additional evidence
    # Keywords indicating distinct roles/statuses
    role_a = set(re.findall(r'\b(khan|emperor|merchant|soldier|official|founder|steppe|horde)\b', text_a))
    role_b = set(re.findall(r'\b(khan|emperor|merchant|soldier|official|founder|steppe|horde)\b', text_b))
    
    # If roles are mutually exclusive or clearly different in a way that implies different persons
    # For this simplified logic, if we have specific roles and they don't overlap, and dates don't support same person
    exclusive_pairs = {('merchant', 'khan'), ('soldier', 'emperor'), ('minor official', 'khan')}
    
    # Simplified logic: If we detected a temporal conflict, mark as CONFLICT
    if conflict_reasons:
        return 'CONFLICT', conflict_reasons
    
    # If no conflicts, check if there's supporting evidence
    # Supporting evidence: overlapping years, or shared specific keywords
    if years_a and years_b:
        # If there is any overlap or proximity, it's not a conflict
        if not (max_a < min_b - 30 or max_b < min_a - 30):
            support_reasons.append("Temporal contexts are compatible")
            
    common_roles = role_a.intersection(role_b)
    if common_roles:
        support_reasons.append(f"Shared context markers: {', '.join(sorted(common_roles))}")
    
    if support_reasons:
        return 'SUPPORT', support_reasons
    
    if not source_a or not source_b:
        return 'INSUFFICIENT', []
        
    return 'NEUTRAL', []

def match_persons(record_a: PersonRecord, record_b: PersonRecord, mode: str = "name_only") -> Dict[str, Any]:
    result = {
        "decision": None,
        "reason": None
    }
    
    if mode == "name_only":
        if check_name_similarity(record_a, record_b):
            result["decision"] = "SAME"
            result["reason"] = "Name variants matched via similarity or transliteration."
        else:
            result["decision"] = "DIFFERENT"
            result["reason"] = "Name variants did not match."
            
    elif mode == "source_grounded":
        # First check names. If names are clearly different and not transliterations, it's likely different.
        # But we allow "SAME" if names are similar OR if sources strongly support same person despite name diff?
        # The prompt says: "overcoming name surface mismatch" if sources align.
        
        name_match = check_name_similarity(record_a, record_b)
        
        status, reasons = analyze_source_conflicts(record_a.source_evidence, record_b.source_evidence)
        
        if status == 'CONFLICT':
            result["decision"] = "DIFFERENT"
            result["reason"] = "Source evidence contradicts temporal/contextual info: " + "; ".join(reasons)
            
        elif status == 'SUPPORT':
            if name_match:
                result["decision"] = "SAME"
                result["reason"] = "Names match and source evidence supports identity."
            else:
                # Names don't match, but sources support. 
                # The prompt Case 1 says: "overcoming name surface mismatch".
                result["decision"] = "SAME"
                result["reason"] = "Source evidence indicates historical context aligns, overcoming name surface mismatch."
                
        elif status == 'NEUTRAL':
            if name_match:
                result["decision"] = "SAME"
                result["reason"] = "Names match. Source evidence is neutral but does not contradict."
            else:
                result["decision"] = "ABSTAIN"
                result["reason"] = "Names do not match and source evidence does not provide sufficient linking or contradicting high-level identity markers."
                
        elif status == 'INSUFFICIENT':
            if name_match:
                 # If names match but sources are empty? 
                 # Prompt Case 3: "Insufficient provenance... names identical... ABSTAIN"
                 # Wait, Case 3 says "names identical" but source "merchant" vs "soldier" -> ABSTAIN.
                 # My logic above puts that in NEUTRAL (no conflict, no strong support). 
                 # Let's refine: If names are identical, but sources are "weak" (no strong support like dates/roles overlap), we might abstain?
                 # The prompt Case 3 logic: "names identical but sources do not provide linking or explicitly contradicting high-level identity markers."
                 # In my analyze_source_conflicts, 'merchant' vs 'soldier' yields NEUTRAL (no conflict, no common role keyword match in my simple set? 
                 # Actually 'merchant' and 'soldier' are in the regex set, so role_a={'merchant'}, role_b={'soldier'}, intersection is empty.
                 # years are empty. So it returns NEUTRAL.
                 # So for Case 3, we have name_match=True, status=NEUTRAL. 
                 # My current logic returns SAME. But prompt says ABSTAIN.
                 # I need to distinguish between "Strong Name Match + Neutral Source" (SAME) vs "Exact Name Match + Weak/Non-linking Source" (ABSTAIN)?
                 # Or maybe "Weak Name Match + Neutral Source" (ABSTAIN)?
                 
                 # Let's re-read Case 3: "Qutlugh" vs "Qutlugh". Exact name match.
                 # Source A: "Qutlugh, merchant". Source B: "Qutlugh, soldier".
                 # Prompt says: ABSTAIN. Reason: "Insufficient provenance... names identical...".
                 
                 # Case 1: "Genghis Khan" vs "Temujin". Name match = False (via transliteration it is True in my code? 
                 # TRANSLIT_TABLE maps 'temujin' -> 'genghis khan' and 'genghis khan' is the key? 
                 # No, my table has "temujin": "genghis khan" and "chinghis khan": "genghis khan".
                 # So check_name_similarity will return True for Genghis Khan and Temujin.
                 # Source A: "Chinghis Khan of the Steppe". Source B: "Temujin, founder of the empire".
                 # analyze_source_conflicts: 
                 #   years: none.
                 #   roles_a: {'khan', 'steppe'}
                 #   roles_b: {'founder'} (Wait, 'temujin' is a name, not in role set. 'founder' is in set.)
                 #   Intersection is empty.
                 #   Returns NEUTRAL.
                 # So Case 1: name_match=True, status=NEUTRAL.
                 # My current logic returns SAME. Prompt says SAME. This is consistent.
                 
                 # Case 3: name_match=True, status=NEUTRAL.
                 # My current logic returns SAME. Prompt says ABSTAIN.
                 
                 # How to distinguish Case 1 and Case 3?
                 # Case 1: Names are different surface forms (Genghis Khan vs Temujin) but linked by transliteration. Sources have no conflict.
                 # Case 3: Names are identical (Qutlugh vs Qutlugh). Sources have no conflict but also no strong linking evidence (like dates or shared specific roles that imply same person).
                 
                 # The prompt implies that if names are identical, you need *positive* evidence from sources to say SAME? 
                 # Or if names are different, you need *absence of conflict*?
                 
                 # Let's adjust the logic for NEUTRAL status:
                 # If names are different (or only transliteration match) -> SAME (absence of conflict is enough to bridge gap if transliteration links them).
                 # If names are identical -> We need stronger evidence? Or is "identical name + neutral source" considered "insufficient" because it could be a homonym without dates?
                 
                 # Actually, in Case 2 (Batu), name is identical. Source has dates. Conflict detected -> DIFFERENT.
                 # In Case 3 (Qutlugh), name is identical. Source has no dates. Neutral. -> ABSTAIN.
                 
                 # So, if name_match is True AND status is NEUTRAL:
                 # Check if the name match was "Exact" or "Fuzzy/Translit".
                 # If Exact Name Match: We are more suspicious of homonyms. If no positive support (dates/roles), ABSTAIN.
                 # If Fuzzy/Translit Name Match: We rely on transliteration. If no conflict, SAME.
                 
                 is_exact_match = False
                 for n1 in record_a.name_variants:
                     for n2 in record_b.name_variants:
                         if normalize_text(n1) == normalize_text(n2):
                             is_exact_match = True
                             
                 if is_exact_match:
                     # If names are exactly the same, we need positive support to avoid homonym error?
                     # But what if it's just a common name and no extra info? The prompt says ABSTAIN.
                     result["decision"] = "ABSTAIN"
                     result["reason"] = "Names are identical but source evidence lacks high-level identity markers (dates, roles) to resolve potential homonyms."
                 else:
                     # Transliteration or fuzzy match. No conflict.
                     result["decision"] = "SAME"
                     result["reason"] = "Source evidence does not contradict identity, and names are linked via transliteration/similarity."
                     
        else:
            result["decision"] = "ABSTAIN"
            result["reason"] = "Unknown status."
            
    else:
        result["decision"] = "ERROR"
        result["reason"] = f"Unknown mode: {mode}"
        
    return result

def run_simulation():
    print("--- MHER Source-Grounded Entity Reconciliation Simulator ---\n")
    
    # Case 1: Different names, same person (Genghis Khan / Temujin)
    person_a_1 = PersonRecord(
        id="A1",
        name_variants=["Genghis Khan", "Chinghis Khan"],
        source_evidence=["Chinghis Khan of the Steppe, united the tribes."]
    )
    person_b_1 = PersonRecord(
        id="B1",
        name_variants=["Temujin"],
        source_evidence=["Temujin, founder of the empire, born in 1162."]
    )
    
    res_name_1 = match_persons(person_a_1, person_b_1, mode="name_only")
    res_src_1 = match_persons(person_a_1, person_b_1, mode="source_grounded")
    
    print("Case 1: Genghis Khan vs Temujin")
    print(f"  Name-Only: {res_name_1['decision']} ({res_name_1['reason']})")
    print(f"  Source-Grounded: {res_src_1['decision']} ({res_src_1['reason']})")
    print()
    
    # Case 2: Same name, different person (Batu)
    person_a_2 = PersonRecord(
        id="A2",
        name_variants=["Batu"],
        source_evidence=["Batu, Khan of the Golden Horde, died 1255."]
    )
    person_b_2 = PersonRecord(
        id="B2",
        name_variants=["Batu"],
        source_evidence=["Batu, a minor official in the 1300s, recorded in 1310."]
    )
    
    res_name_2 = match_persons(person_a_2, person_b_2, mode="name_only")
    res_src_2 = match_persons(person_a_2, person_b_2, mode="source_grounded")
    
    print("Case 2: Batu (Khan) vs Batu (Official)")
    print(f"  Name-Only: {res_name_2['decision']} ({res_name_2['reason']})")
    print(f"  Source-Grounded: {res_src_2['decision']} ({res_src_2['reason']})")
    print()
    
    # Case 3: Same name, insufficient evidence (Qutlugh)
    person_a_3 = PersonRecord(
        id="A3",
        name_variants=["Qutlugh"],
        source_evidence=["Qutlugh, merchant in the Silk Road trade."]
    )
    person_b_3 = PersonRecord(
        id="B3",
        name_variants=["Qutlugh"],
        source_evidence=["Qutlugh, soldier in the guard."]
    )
    
    res_name_3 = match_persons(person_a_3, person_b_3, mode="name_only")
    res_src_3 = match_persons(person_a_3, person_b_3, mode="source_grounded")
    
    print("Case 3: Qutlugh (Merchant) vs Qutlugh (Soldier)")
    print(f"  Name-Only: {res_name_3['decision']} ({res_name_3['reason']})")
    print(f"  Source-Grounded: {res_src_3['decision']} ({res_src_3['reason']})")
    print()

if __name__ == "__main__":
    run_simulation()
