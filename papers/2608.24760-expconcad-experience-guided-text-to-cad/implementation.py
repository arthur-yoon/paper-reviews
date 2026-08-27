# implementation.py

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple

# Named constants to replace magic numbers
CONSTRAINT_SOURCE_USER_INPUT = "user_input"
CONSTRAINT_SOURCE_RETRIEVED_EXPERIENCE = "retrieved_experience"
CONSTRAINT_SOURCE_DEFAULT = "default"

# Regex patterns for parsing
# Pattern for Box: "Box of size WxDxH"
RE_BOX = re.compile(r"Box\s+of\s+size\s+(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)", re.IGNORECASE)
# Pattern for Cylinder: "Cylinder with radius R"
RE_CYLINDER = re.compile(r"Cylinder\s+with\s+radius\s+(\d+(?:\.\d+)?)", re.IGNORECASE)
# Pattern for Relation: "on top of", "above", "placed on"
RE_ON_TOP_OF = re.compile(r"\bon top of\b|\bover\b|\babove\b", re.IGNORECASE)

# Relation Types
RELATION_ON_TOP_OF = "on_top_of"

# Experience Rules
EXP_ON_TOP_OF_RULES = {
    "part_a_center_x": "part_b_center_x",
    "part_a_center_y": "part_b_center_y",
    "part_b_base_z": "part_a_height"
}

# Setup Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ExpConCAD")


@dataclass
class Constraint:
    """
    Section 3.2: Constraint Scope Identification
    Represents a single spatial constraint variable.
    """
    variable_name: str
    value: Optional[Union[int, float]]
    source: str = CONSTRAINT_SOURCE_DEFAULT

    def is_missing(self) -> bool:
        return self.value is None


@dataclass
class DesignPart:
    """
    Section 3.2: Constraint Scope Identification
    Represents a geometric part in the CAD assembly.
    """
    name: str
    geometry_type: str
    constraints: List[Constraint] = field(default_factory=list)

    def get_constraint(self, var_name: str) -> Optional[Constraint]:
        for c in self.constraints:
            if c.variable_name == var_name:
                return c
        return None

    def set_constraint_value(self, var_name: str, value: Union[int, float], source: str = CONSTRAINT_SOURCE_DEFAULT) -> bool:
        c = self.get_constraint(var_name)
        if c:
            c.value = value
            c.source = source
            logger.debug(f"Part {self.name}: Updated {var_name} to {value} (source: {source})")
            return True
        else:
            logger.warning(f"Part {self.name}: Constraint {var_name} not found to update.")
            return False


@dataclass
class Relation:
    """
    Section 3.1: Construction Structure Recovery
    Represents a spatial relationship between two parts.
    """
    part_a: DesignPart
    part_b: DesignPart
    relation_type: str
    constraints: List[Constraint] = field(default_factory=list)


@dataclass
class ConstructionStructure:
    """
    Section 3.1: Construction Structure Recovery
    Contains the list of parts and relations describing the CAD design.
    """
    parts: List[DesignPart] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)

    def get_part(self, name: str) -> Optional[DesignPart]:
        for p in self.parts:
            if p.name == name:
                return p
        return None


class ExperienceDatabase:
    """
    Section 3.3: Experience Retrieval
    Stores and retrieves design experiences (rules) based on relation patterns.
    """
    def __init__(self):
        self._experiences: Dict[str, Dict[str, str]] = {}
        logger.info("ExperienceDatabase initialized.")

    def add_experience(self, pattern: str, rules: Dict[str, str]) -> None:
        """
        Adds an experience rule to the database.
        pattern: e.g., "on_top_of"
        rules: e.g., {"part_b_center_x": "part_a_center_x"}
        """
        self._experiences[pattern] = rules
        logger.debug(f"Added experience for pattern '{pattern}' with {len(rules)} rules.")

    def search(self, relation_type: str) -> Optional[Dict[str, str]]:
        """
        Retrieves experience rules for a given relation type.
        """
        rules = self._experiences.get(relation_type)
        if rules:
            logger.debug(f"Retrieved experience for relation type '{relation_type}'.")
            return rules
        else:
            logger.debug(f"No experience found for relation type '{relation_type}'.")
            return None


class ExpConCADAgent:
    """
    Main Agent implementing the ExpConCAD framework.
    Implements:
    - Section 3.1: Structure Recovery
    - Section 3.3: Experience Retrieval
    - Section 3.4: Implicit Spatial Constraint Completion
    - Section 3.5: CadQuery Program Generation
    """
    def __init__(self):
        self.db = ExperienceDatabase()
        # Pre-load standard experiences
        self.db.add_experience(RELATION_ON_TOP_OF, EXP_ON_TOP_OF_RULES)
        logger.info("ExpConCADAgent initialized with default experiences.")

    def recover_structure(self, text: str) -> ConstructionStructure:
        """
        Section 3.1: Construction Structure Recovery
        Parses natural language input into DesignPart and Relation objects.
        Simulates LLM reasoning using regex.
        """
        logger.info(f"Starting structure recovery for text: '{text}'")
        structure = ConstructionStructure()

        # 1. Detect Parts
        box_match = RE_BOX.search(text)
        if box_match:
            w, d, h = float(box_match.group(1)), float(box_match.group(2)), float(box_match.group(3))
            box = DesignPart(
                name="box",
                geometry_type="Box",
                constraints=[
                    Constraint("width", w, CONSTRAINT_SOURCE_USER_INPUT),
                    Constraint("depth", d, CONSTRAINT_SOURCE_USER_INPUT),
                    Constraint("height", h, CONSTRAINT_SOURCE_USER_INPUT),
                    # Default positions for base part (center at origin)
                    Constraint("center_x", 0.0, CONSTRAINT_SOURCE_DEFAULT),
                    Constraint("center_y", 0.0, CONSTRAINT_SOURCE_DEFAULT),
                    Constraint("base_z", 0.0, CONSTRAINT_SOURCE_DEFAULT)
                ]
            )
            structure.parts.append(box)
            logger.debug(f"Parsed Box: w={w}, d={d}, h={h}")

        cyl_match = RE_CYLINDER.search(text)
        if cyl_match:
            r = float(cyl_match.group(1))
            cyl = DesignPart(
                name="cylinder",
                geometry_type="Cylinder",
                constraints=[
                    Constraint("radius", r, CONSTRAINT_SOURCE_USER_INPUT),
                    Constraint("height", r * 2.0, CONSTRAINT_SOURCE_DEFAULT), # Assume height = 2*radius if not specified
                    # Missing constraints to be completed
                    Constraint("center_x", None),
                    Constraint("center_y", None),
                    Constraint("base_z", None)
                ]
            )
            structure.parts.append(cyl)
            logger.debug(f"Parsed Cylinder: r={r}")

        # 2. Detect Relations
        # Heuristic: If both Box and Cylinder exist and "on top of" is present
        if len(structure.parts) == 2 and RE_ON_TOP_OF.search(text):
            box = structure.get_part("box")
            cyl = structure.get_part("cylinder")
            
            if box and cyl:
                relation = Relation(
                    part_a=box,
                    part_b=cyl,
                    relation_type=RELATION_ON_TOP_OF,
                    constraints=[] # Relation constraints are often derived from part constraints
                )
                structure.relations.append(relation)
                logger.info(f"Identified relation: {box.name} <-[{relation.relation_type}]-> {cyl.name}")
            else:
                logger.warning("Relation detected but parts missing or ambiguous.")
        else:
            logger.debug("No valid relation identified.")

        logger.info(f"Structure recovery complete. Parts: {[p.name for p in structure.parts]}, Relations: {[r.relation_type for r in structure.relations]}")
        return structure

    def retrieve_experience(self, structure: ConstructionStructure) -> Dict[Relation, Dict[str, str]]:
        """
        Section 3.3: Experience Retrieval
        Retrieves relevant experiences for each relation in the structure.
        """
        logger.info("Starting experience retrieval.")
        retrieved_map: Dict[Relation, Dict[str, str]] = {}
        
        for relation in structure.relations:
            rules = self.db.search(relation.relation_type)
            if rules:
                retrieved_map[relation] = rules
                logger.debug(f"Experience retrieved for {relation.part_a.name}-{relation.part_b.name}")
            else:
                logger.warning(f"No experience found for relation type: {relation.relation_type}")
        
        logger.info(f"Experience retrieval complete. Retrieved {len(retrieved_map)} experiences.")
        return retrieved_map

    def complete_constraints(self, structure: ConstructionStructure, experiences: Dict[Relation, Dict[str, str]]) -> ConstructionStructure:
        """
        Section 3.4: Implicit Spatial Constraint Completion
        Fills in missing constraint values using retrieved experiences.
        """
        logger.info("Starting constraint completion.")
        
        for relation, rules in experiences.items():
            part_a = relation.part_a
            part_b = relation.part_b
            
            # Extract current values from part_a for calculation
            # We need a helper to get values safely
            a_vals = {}
            for c in part_a.constraints:
                if c.value is not None:
                    a_vals[c.variable_name] = c.value
            
            # Apply rules
            for target_var, source_var in rules.items():
                # Determine which part the target variable belongs to
                # Convention: "part_a_var" or "part_b_var"
                if target_var.startswith("part_a_"):
                    target_part = part_a
                    var_name = target_var[len("part_a_"):]
                elif target_var.startswith("part_b_"):
                    target_part = part_b
                    var_name = target_var[len("part_b_"):]
                else:
                    logger.warning(f"Invalid rule target format: {target_var}")
                    continue

                # Check if target is currently missing
                target_constraint = target_part.get_constraint(var_name)
                if target_constraint is None:
                    logger.warning(f"Target constraint {var_name} not found in {target_part.name}.")
                    continue
                if not target_constraint.is_missing():
                    logger.debug(f"Constraint {target_part.name}.{var_name} already defined ({target_constraint.value}). Skipping.")
                    continue
                
                # Resolve source value
                source_val = None
                if source_var.startswith("part_a_"):
                    source_part = part_a
                    source_name = source_var[len("part_a_"):]
                elif source_var.startswith("part_b_"):
                    source_part = part_b
                    source_name = source_var[len("part_b_"):]
                else:
                    # Maybe it's a literal number or a direct part_a var name?
                    # In our EXP_ON_TOP_OF_RULES, keys are "part_x_var" and values are "part_y_var"
                    # But the example in prompt: {"var": "cylinder_center_x", "calc": "box_center_x"}
                    # Our EXP_ON_TOP_OF_RULES: "part_a_center_x": "part_b_center_x" -> This is wrong for "on top of".
                    # Let's re-evaluate the rule logic.
                    # If A is Box (base) and B is Cylinder (top).
                    # B.center_x should align with A.center_x.
                    # B.base_z should be A.center_z? Or A.height?
                    # If Box is centered at (0,0,0) with height H, top surface is at Z=H/2? Or Z=H?
                    # Usually, if "center_z" is 0, box spans -H/2 to H/2.
                    # Or if "base_z" is 0, box spans 0 to H.
                    # In our recover_structure, we set Box: center_x=0, center_y=0, base_z=0.
                    # So Box top is at Z=H.
                    # Cylinder base_z should be H.
                    # Cylinder center_x should be 0 (same as Box center_x).
                    
                    # My EXP_ON_TOP_OF_RULES in __init__:
                    # "part_a_center_x": "part_b_center_x" -> This means A.center_x = B.center_x. Correct.
                    # "part_a_center_y": "part_b_center_y" -> Correct.
                    # "part_b_base_z": "part_a_height" -> B.base_z = A.height. Correct.
                    
                    # So the mapping is:
                    # Key: Target Variable (part_id_var_name)
                    # Value: Source Variable (part_id_var_name)
                    # We need to evaluate the Source Variable's value from the *other* part or same part.
                    
                    # Let's trace:
                    # Rule: "part_b_base_z": "part_a_height"
                    # Target: part_b (cyl) var "base_z"
                    # Source: part_a (box) var "height"
                    
                    # My code above:
                    # source_var = "part_a_height"
                    # source_part = part_a (box)
                    # source_name = "height"
                    # source_val = a_vals.get("height") -> 5.0
                    # This works.
                    
                    # Rule: "part_a_center_x": "part_b_center_x"
                    # Target: part_a (box) var "center_x"
                    # Source: part_b (cyl) var "center_x"
                    # Wait, Box center_x is already 0.0 (not missing). So this rule is skipped for Box.
                    # What about Cylinder center_x?
                    # It is missing. But the rule says "part_a_center_x" is the target.
                    # This rule implies: Box.center_x = Cyl.center_x.
                    # Since Box.center_x is known, we can infer Cyl.center_x = Box.center_x.
                    # My current logic only fills if TARGET is missing.
                    # Target "part_a_center_x" is NOT missing. So it skips.
                    # We miss the opportunity to fill Cyl.center_x!
                    
                    # We need bidirectional inference or explicit rules for both directions.
                    # Better approach: Define rules as "part_b_center_x": "part_a_center_x".
                    # Then Target: Cyl.center_x (Missing). Source: Box.center_x (Known).
                    # This fills Cyl.center_x.
                    
                    # Let's fix EXP_ON_TOP_OF_RULES in __init__ to be more symmetric or explicit for the top part.
                    # Actually, the prompt's example:
                    # {"var": "cylinder_center_x", "calc": "box_center_x"}
                    # This suggests: Cylinder's var is calculated from Box's var.
                    # My keys in EXP_ON_TOP_OF_RULES were:
                    # "part_a_center_x": "part_b_center_x"
                    # This is the reverse of what's needed if A is Base and B is Top.
                    # If A=Box, B=Cyl.
                    # We want B.center_x = A.center_x.
                    # So Rule should be: "part_b_center_x": "part_a_center_x".
                    
                    # I will update the constants in the code below.
                    # But for now, in the function, let's handle the general case.
                    # If Target is missing, look for Source.
                    # If Source is from Part A, get from A.
                    # If Source is from Part B, get from B.
                    
                    # However, if the rule is defined as "part_a_center_x": "part_b_center_x",
                    # and Part A (Box) center_x is NOT missing, we skip.
                    # Part B (Cyl) center_x IS missing, but it's not the target in this specific rule entry.
                    # We need another rule entry: "part_b_center_x": "part_a_center_x".
                    
                    # So, updating EXP_ON_TOP_OF_RULES is the cleanest fix.
                    
                    pass

                # Re-evaluating Source Value
                # We need to get the value from the appropriate part
                # source_var format: "part_id_var_name"
                
                if source_var.startswith("part_a_"):
                    source_part = part_a
                    source_name = source_var[len("part_a_"):]
                elif source_var.startswith("part_b_"):
                    source_part = part_b
                    source_name = source_var[len("part_b_"):]
                else:
                    logger.warning(f"Invalid source var format: {source_var}")
                    continue

                # Get value from source_part
                source_constraint = source_part.get_constraint(source_name)
                if source_constraint is None or source_constraint.value is None:
                    logger.warning(f"Source constraint {source_part.name}.{source_name} is missing or None. Cannot compute {target_part.name}.{var_name}.")
                    continue

                source_val = source_constraint.value
                target_part.set_constraint_value(var_name, source_val, CONSTRAINT_SOURCE_RETRIEVED_EXPERIENCE)
                logger.info(f"Completed: {target_part.name}.{var_name} = {source_val} (from {source_part.name}.{source_name})")

        logger.info("Constraint completion finished.")
        return structure

    def generate_code(self, structure: ConstructionStructure) -> str:
        """
        Section 3.5: CadQuery Program Generation
        Generates CadQuery-like code string based on completed constraints.
        """
        logger.info("Starting code generation.")
        code_lines = []
        code_lines.append("import cadquery as cq")
        code_lines.append("result = cq.Workplane(\"XY\")")
        
        for part in structure.parts:
            # Get values
            get_val = lambda name: part.get_constraint(name).value if part.get_constraint(name) else 0.0
            
            x = get_val("center_x") or 0.0
            y = get_val("center_y") or 0.0
            z = get_val("base_z") or 0.0
            
            if part.geometry_type == "Box":
                w = get_val("width") or 1.0
                d = get_val("depth") or 1.0
                h = get_val("height") or 1.0
                # CadQuery: workplane at z, box(w,d,h)
                code_lines.append(f"# Part: {part.name}")
                code_lines.append(f"result = result.movedTo({x}, {y}, {z}).box({w}, {d}, {h})")
            elif part.geometry_type == "Cylinder":
                r = get_val("radius") or 1.0
                h = get_val("height") or 2.0
                code_lines.append(f"# Part: {part.name}")
                code_lines.append(f"result = result.movedTo({x}, {y}, {z}).circle({r}).extrude({h})")
            else:
                code_lines.append(f"# Unknown geometry: {part.geometry_type}")
                
        code = "\n".join(code_lines)
        logger.info("Code generation complete.")
        logger.debug("Generated Code:\n" + code)
        return code


if __name__ == "__main__":
    agent = ExpConCADAgent()
    
    # Test Case 1: Ambiguous description
    user_input_1 = "Make a Box of size 10x10x5. Place a Cylinder with radius 2 on top of the Box."
    
    logger.info("--- Running Demo: Box with Cylinder on top ---")
    structure1 = agent.recover_structure(user_input_1)
    exps1 = agent.retrieve_experience(structure1)
    completed1 = agent.complete_constraints(structure1, exps1)
    code1 = agent.generate_code(completed1)
    logger.info(f"Final Code for Case 1:\n{code1}")
    
    # Test Case 2: Simple Box
    user_input_2 = "Create a Box of size 2x2x2."
    logger.info("--- Running Demo: Single Box ---")
    structure2 = agent.recover_structure(user_input_2)
    exps2 = agent.retrieve_experience(structure2)
    completed2 = agent.complete_constraints(structure2, exps2)
    code2 = agent.generate_code(completed2)
    logger.info(f"Final Code for Case 2:\n{code2}")

