# File: tests/test_task_executor.py

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add path to import TaskExecutor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../core/task_logic")))

import TaskExecutor


class TestTaskExecutor(unittest.TestCase):

    @patch("TaskExecutor.append_memory")
    @patch("TaskExecutor.save_memory")
    @patch("TaskExecutor.load_memory")
    def test_single_task_execution(self, mock_load_memory, mock_save_memory, mock_append_memory):
        """
        Test that TaskExecutor:
        - Detects 1 pending task
        - Marks it as executed
        - Generates a reflection
        """

        # === Step 1: Setup a fake task ===
        task = {
            "type": "task",
            "description": "Test execution system",
            "emotion": "focus",
            "tags": ["unit-test"],
            "executed": False
        }

        # === Step 2: Simulate memory load ===
        mock_load_memory.return_value = [task]

        # === Step 3: Run the execution logic ===
        executed_count = TaskExecutor.run_task_execution(verbose=True)

        # === Step 4: Assertions ===
        self.assertEqual(executed_count, 1)

        # Task should be updated and saved
        mock_save_memory.assert_called_once()
        updated_memory = mock_save_memory.call_args[0][0]
        self.assertTrue(updated_memory[0]["executed"])

        # Reflection should be added
        mock_append_memory.assert_called_once()
        reflection = mock_append_memory.call_args[0][0]
        self.assertEqual(reflection["type"], "reflection")
        self.assertIn("Executed task", reflection["content"])
        self.assertEqual(reflection["emotion"], "satisfaction")


if __name__ == "__main__":
    unittest.main()
