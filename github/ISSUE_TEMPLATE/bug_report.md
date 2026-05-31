name: 🐛 Bug report
description: Create a report to help us improve CricScope
title: "bug: "
labels: ["bug"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  - type: textarea
    id: describe-bug
    attributes:
      label: Describe the bug
      description: A clear and concise description of what the bug is.
      placeholder: Describe the problem clearly...
    validations:
      required: true
  - type: textarea
    id: reproduction-steps
    attributes:
      label: Steps to reproduce
      description: List the exact steps to reproduce the problem.
      placeholder: |
        1. Run the app local server
        2. Go to pages/...
        3. Click on ...
        4. See error
    validations:
      required: true
  - type: textarea
    id: expected-behavior
    attributes:
      label: Expected Behavior
      description: What did you expect to happen instead?
      placeholder: Explain what should happen...
    validations:
      required: true
  - type: textarea
    id: environment
    attributes:
      label: Environment Info
      description: Python version, Streamlit version, OS, etc.
      placeholder: |
        - OS: Windows/Linux
        - Python: 3.10
        - Browser: Chrome/Safari
    validations:
      required: false
