# Security Policy

We take the security of CricScope seriously. If you believe you have found a security vulnerability in this project, we appreciate your help in disclosing it to us in a responsible manner.

## Supported Versions

Currently, security updates and patches are actively backported to the following branches:

| Version | Status |
|:---|:---|
| Main (1.x) | Supported |
| < 1.0.0 | Unsupported |

We strongly recommend that all users run the latest release of CricScope. If you are running an older release or an un-synced fork, please sync with the upstream repository to receive active patches.

## Reporting a Vulnerability

Please do **NOT** report security vulnerabilities publicly via GitHub Issues, Pull Requests, or public comments. Instead, report them privately using the procedure below.

### How to Report

To report a suspected security vulnerability:

1.  Send a private email to the project maintainer at **itsarnav.singh80@gmail.com**.
2.  Provide a clear and descriptive subject line (e.g., `Security Vulnerability Report: [Brief Description]`).
3.  Include as much detail as possible in your report to help us understand and resolve the issue quickly:
    *   **Description:** A detailed explanation of the vulnerability and its potential impact.
    *   **Steps to Reproduce:** Clear, step-by-step instructions or a minimal proof-of-concept (PoC) script.
    *   **Affected Components:** Specific files, routes, or functions involved (e.g., API key inputs or data loaders).
    *   **Mitigation Suggestions:** Any ideas you have on how to resolve the vulnerability.

## Response and Resolution Timeline

Once a report is received, the CricScope maintainers commit to the following response timeline:

1.  **Initial Acknowledgment:** Within 48 hours of receiving the email.
2.  **Investigation and Triage:** Within 5 business days, we will verify the vulnerability and determine its severity level (Low, Medium, High, or Critical).
3.  **Remediation:** We aim to implement, test, and deploy a security fix within 30 days of triage.
4.  **Coordinated Disclosure:** We will coordinate with the reporter to publish a security advisory alongside a patched release, ensuring credit is properly attributed.

## Out of Scope

The following areas are considered out of scope for security reports:

*   Standard limitations or issues related to Streamlit's underlying architecture (e.g., local execution vulnerabilities if the host server is already compromised).
*   Known, publicly documented behaviors of scikit-learn model storage (pickle format serialization limits).
*   Denial of Service (DoS) attacks or automated brute-forcing of local deployment ports.

Thank you for helping us keep the CricScope project and its community safe!