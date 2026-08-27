# test_implementation.py

import unittest
from implementation import (
    ExpConCADAgent,
    Constraint,
    DesignPart,
    Relation,
    ConstructionStructure,
    CONSTRAINT_SOURCE_USER_INPUT,
    CONSTRAINT_SOURCE_RETRIEVED_EXPERIENCE,
    CONSTRAINT_SOURCE_DEFAULT
)


class TestExpConCADAgent(unittest.TestCase):
    """
    Tests for the ExpConCADAgent implementation.
    """

    def setUp(self):
        self.agent = ExpConCADAgent()

    def test_recover_structure_box_and_cylinder(self):
        """
        Test Section 3.1: Structure Recovery
        Verifies that parts and relations are correctly parsed.
        """
        text = "Make a Box of size 10x10x5. Place a Cylinder with radius 2 on top of the Box."
        structure = self.agent.recover_structure(text)

        self.assertEqual(len(structure.parts), 2)
        self.assertEqual(len(structure.relations), 1)

        box = structure.get_part("box")
        cyl = structure.get_part("cylinder")

        self.assertIsNotNone(box)
        self.assertIsNotNone(cyl)
        self.assertEqual(box.geometry_type, "Box")
        self.assertEqual(cyl.geometry_type, "Cylinder")

        # Check Box constraints
        self.assertEqual(box.get_constraint("width").value, 10.0)
        self.assertEqual(box.get_constraint("height").value, 5.0)

        # Check Cylinder constraints (missing initially)
        self.assertEqual(cyl.get_constraint("center_x").value, None)
        self.assertEqual(cyl.get_constraint("base_z").value, None)

        # Check Relation
        rel = structure.relations[0]
        self.assertEqual(rel.relation_type, "on_top_of")
        self.assertEqual(rel.part_a.name, "box")
        self.assertEqual(rel.part_b.name, "cylinder")

    def test_complete_constraints_on_top_of(self):
        """
        Test Section 3.4: Constraint Completion
        Verifies that missing constraints are filled correctly based on experience.
        """
        # Manually construct a structure to ensure specific initial states
        # Box at origin, height 5
        box = DesignPart(
            name="box",
            geometry_type="Box",
            constraints=[
                Constraint("center_x", 0.0, CONSTRAINT_SOURCE_USER_INPUT),
                Constraint("center_y", 0.0, CONSTRAINT_SOURCE_USER_INPUT),
                Constraint("base_z", 0.0, CONSTRAINT_SOURCE_USER_INPUT),
                Constraint("height", 5.0, CONSTRAINT_SOURCE_USER_INPUT)
            ]
        )
        # Cylinder with missing position
        cyl = DesignPart(
            name="cylinder",
            geometry_type="Cylinder",
            constraints=[
                Constraint("radius", 2.0, CONSTRAINT_SOURCE_USER_INPUT),
                Constraint("height", 4.0, CONSTRAINT_SOURCE_USER_INPUT),
                Constraint("center_x", None),
                Constraint("center_y", None),
                Constraint("base_z", None)
            ]
        )
        relation = Relation(part_a=box, part_b=cyl, relation_type="on_top_of")
        structure = ConstructionStructure(parts=[box, cyl], relations=[relation])

        # Retrieve experience
        experiences = self.agent.retrieve_experience(structure)
        
        # Complete constraints
        completed_structure = self.agent.complete_constraints(structure, experiences)
        
        # Verify completed constraints
        cyl_completed = completed_structure.get_part("cylinder")
        
        # Cylinder center_x should match Box center_x (0.0)
        cx = cyl_completed.get_constraint("center_x")
        self.assertIsNotNone(cx)
        self.assertEqual(cx.value, 0.0)
        self.assertEqual(cx.source, CONSTRAINT_SOURCE_RETRIEVED_EXPERIENCE)

        # Cylinder center_y should match Box center_y (0.0)
        cy = cyl_completed.get_constraint("center_y")
        self.assertIsNotNone(cy)
        self.assertEqual(cy.value, 0.0)
        self.assertEqual(cy.source, CONSTRAINT_SOURCE_RETRIEVED_EXPERIENCE)

        # Cylinder base_z should match Box height (5.0)
        bz = cyl_completed.get_constraint("base_z")
        self.assertIsNotNone(bz)
        self.assertEqual(bz.value, 5.0)
        self.assertEqual(bz.source, CONSTRAINT_SOURCE_RETRIEVED_EXPERIENCE)

    def test_generate_code_structure(self):
        """
        Test Section 3.5: Code Generation
        Verifies that the generated code string contains expected CadQuery commands.
        """
        text = "Make a Box of size 10x10x5. Place a Cylinder with radius 2 on top of the Box."
        structure = self.agent.recover_structure(text)
        experiences = self.agent.retrieve_experience(structure)
        completed_structure = self.agent.complete_constraints(structure, experiences)
        code = self.agent.generate_code(completed_structure)

        # Check for CadQuery import
        self.assertIn("import cadquery as cq", code)
        
        # Check for Box creation
        # Box should be at z=0, size 10x10x5
        self.assertIn("box(10.0, 10.0, 5.0)", code)
        
        # Check for Cylinder creation
        # Cylinder should be at z=5.0 (base_z completed to 5.0), radius 2, height 4.0 (default 2*radius)
        self.assertIn("circle(2.0).extrude(4.0)", code)
        
        # Check that cylinder is moved to z=5.0
        # The code uses movedTo(x, y, z). 
        # Box: movedTo(0.0, 0.0, 0.0)
        # Cyl: movedTo(0.0, 0.0, 5.0)
        self.assertIn("movedTo(0.0, 0.0, 5.0)", code)

    def test_independent_parts_no_completion(self):
        """
        Test edge case: Parts with no relation should remain unchanged.
        """
        text = "Make a Box of size 1x1x1. Make a Cylinder with radius 1."
        structure = self.agent.recover_structure(text)
        experiences = self.agent.retrieve_experience(structure)
        completed_structure = self.agent.complete_constraints(structure, experiences)
        
        cyl = completed_structure.get_part("cylinder")
        # center_x should still be None because no relation to complete it
        self.assertEqual(cyl.get_constraint("center_x").value, None)


if __name__ == "__main__":
    unittest.main()

