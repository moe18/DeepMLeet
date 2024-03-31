# How to Contribute

We welcome contributions to the project in various forms. There are primarily two ways you can contribute:

## 1. Enhancing the User Interface (UI)

If you're interested in improving the user interface, follow these steps:

- **Fork the repository**: Begin by forking the project repository to your GitHub account.
- **Clone your fork**: Clone your forked repository to your local machine to make your changes.
- **Make your changes**: Work on enhancing the UI in your local setup.
- **Submit a Pull Request (PR)**: Once you're satisfied with your changes, commit them and push to your fork on GitHub. Then, submit a Pull Request to the original repository. Your changes will be reviewed and, if approved, merged into the project.

## 2. Adding or Improving Problems

Contributing problems or solutions is another excellent way to help out. Here’s how to add a new problem:

1. **Fork and Clone**: Similar to UI enhancements, start by forking the repository and cloning it to your local system.
2. **Create the Problem**: Add a new problem to the appropriate directory. Here's the template you should follow:

```bash
"Name of Problem (difficulty)": {
    "description": "A brief description of the problem.",
    "example": """Example:
        Input:
        Output:
        Reasoning:""",
    "learn": "This section is designed to teach the user how to solve the problem without directly giving away the answer.",
    "starter_code": "def test_func(a: List[List[int|float]]) -> Set:\n    return (n, m)",
    "solution": """def test_func(a: List[List[int|float]]) -> Set:
    # The solution to the questions""",
    "test_cases": [
        {"test": "test_func(input_1)", "expected_output": "result_1"},
        {"test": "test_func(input_2)", "expected_output": "result_2"},
        {"test": "test_func(input_3)", "expected_output": "result_3"},
    ],
}
```
**Submit Your Changes:** After adding the new problem, commit your changes and push them to your fork. Submit a Pull Request to the main repository with your additions.
