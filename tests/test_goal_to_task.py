# File: tests/test_goal_to_task.py

import unittest
from unittest.mock import MagicMock
import sys
import os

# Add tools/ to the Python path to import goal_to_task_converter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../tools")))

#import goal_to_task_converter  # now imported after setting path
import goal_to_task_runner as goal_to_task_converter  # renamed

MOCK_GOAL = {
    "id": "goal_001",
    "description": "Research emotion-influenced scheduling",
    "emotion": "curious",
    "tags": ["research", "emotion", "AI"],
    "converted_to_task": False
}

EXPECTED_TASK = {
    "type": "task",
    "description": "Research emotion-influenced scheduling",
    "emotion": "curious",
    "tags": ["research", "emotion", "AI"],
    "converted_from_goal": "goal_001"
}

class TestGoalToTaskConversion(unittest.TestCase):
    def test_goal_conversion_to_task(self):
        # Replace goal_to_task_converter functions with mocks
        goal_to_task_converter.save_memory = MagicMock()
        goal_to_task_converter.save_goals = MagicMock()
        goal_to_task_converter.load_goals = MagicMock(return_value=[MOCK_GOAL.copy()])

        # Run the actual logic
        goal_to_task_converter.convert_goals_to_tasks()

        # Validate the memory save call
        goal_to_task_converter.save_memory.assert_called_once()
        task = goal_to_task_converter.save_memory.call_args[0][0]
        self.assertEqual(task["description"], EXPECTED_TASK["description"])
        self.assertEqual(task["emotion"], EXPECTED_TASK["emotion"])
        self.assertEqual(task["tags"], EXPECTED_TASK["tags"])
        self.assertEqual(task["converted_from_goal"], EXPECTED_TASK["converted_from_goal"])
        self.assertEqual(task["type"], "task")

        # Confirm the updated goal is marked as converted
        updated_goal = goal_to_task_converter.load_goals.return_value[0]
        self.assertTrue(updated_goal["converted_to_task"])

        # Ensure the save_goals function was called
        goal_to_task_converter.save_goals.assert_called_once()

if __name__ == "__main__":
    unittest.main()
