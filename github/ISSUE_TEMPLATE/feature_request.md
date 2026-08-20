name: 🚀 Feature request
description: Suggest an idea or enhancement for CricScope
title: "feat: "
labels: ["enhancement"]
body:
  - type: markdown
    attributes:
      value: |
        We love new feature ideas! Share your proposal with us.
  - type: textarea
    id: feature-description
    attributes:
      label: Feature Description
      description: Describe the core enhancement idea.
      placeholder: What feature are you proposing?
    validations:
      required: true
  - type: textarea
    id: use-case
    attributes:
      label: Use Case & Motivation
      description: Explain why this is helpful for users or developers.
      placeholder: Why is this needed?
    validations:
      required: true
  - type: textarea
    id: alternative-solutions
    attributes:
      label: Alternative Solutions Considered
      description: Any alternative implementations you thought about.
      placeholder: Did you consider other options?
    validations:
      required: false
