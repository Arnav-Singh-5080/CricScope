# Security Policy

<div align="center">

# CricScope Security Policy

Luxury-grade IPL analytics platform powered by Machine Learning.

![Security](https://img.shields.io/badge/Security-Policy-success?style=for-the-badge)
![Maintained](https://img.shields.io/badge/Maintained-Yes-gold?style=for-the-badge)
![Open Source](https://img.shields.io/badge/Open%20Source-GSSoC%20%7C%20NSoC-blueviolet?style=for-the-badge)

</div>

---

# Supported Versions

The following versions of CricScope currently receive security updates and active maintenance.

| Version | Supported | Status |
| ------- | ---------- | ------ |
| 2.x | ✅ Yes | Active development |
| 1.x | ⚠️ Partial | Critical fixes only |
| < 1.0 | ❌ No | Unsupported |

---

# Reporting a Vulnerability

If you discover a security vulnerability in CricScope, please report it responsibly.

We take all legitimate security reports seriously and appreciate responsible disclosure from contributors and researchers.

---

# Please DO NOT

- Open public GitHub issues for vulnerabilities
- Leak exploit details publicly
- Share sensitive credentials or datasets
- Perform destructive testing on deployments
- Abuse the Streamlit hosting environment
- Attempt unauthorized access to contributor accounts

---

# Responsible Disclosure Process

Please send a detailed report including:

- Vulnerability description
- Steps to reproduce
- Expected vs actual behavior
- Severity assessment
- Screenshots/logs (if applicable)
- Suggested mitigation (optional)

---

# Contact Information

## Project Admin

**Arnav Singh**

- Email: `itsarnav.singh80@gmail.com`
- GitHub: `@Arnav-Singh-5080`

---

# Response Timeline

We aim to respond within the following timeframe:

| Action | Estimated Time |
| ------ | --------------- |
| Initial acknowledgement | 24–48 hours |
| Vulnerability triage | 3–5 business days |
| Severity confirmation | Within 1 week |
| Patch / mitigation | Depends on severity |
| Public disclosure | After fix release |

---

# Security Scope

This policy applies to:

## Included Components

- Streamlit frontend
- Custom CSS UI system
- ML prediction pipeline
- scikit-learn preprocessing
- Dataset processing logic
- CSV handling
- Deployment configuration
- Open-source contribution workflow

---

# Out of Scope

The following are generally considered out of scope unless they lead to significant security impact:

- UI styling bugs
- Visual inconsistencies
- Minor typo/documentation issues
- Rate limiting concerns in local deployments
- Public dataset inaccuracies
- Streamlit framework vulnerabilities unrelated to CricScope code

---

# Security Best Practices

Contributors are encouraged to follow these practices:

## Code Safety

- Validate all external inputs
- Avoid unsafe file handling
- Never commit secrets or API keys
- Sanitize user-controlled values
- Keep dependencies updated

## ML/Data Safety

- Validate CSV schemas
- Prevent malformed dataset injection
- Handle NaN/infinite values safely
- Avoid unsafe model serialization

## Open Source Hygiene

- Use meaningful commit messages
- Review pull requests carefully
- Test locally before submitting PRs
- Follow repository coding standards

---

# Dependency Management

We recommend regularly updating:

- Streamlit
- scikit-learn
- pandas
- numpy
- plotly

To update dependencies:

```bash
pip install --upgrade -r requirements.txt
```

---

# Environment Variables

If future versions introduce secrets or API integrations:

## Recommended

Use `.env` files and environment variables.

## Never Commit

- API keys
- Authentication tokens
- Private credentials
- Deployment secrets

Example:

```env
API_KEY=your_secret_key
```

Add `.env` to `.gitignore`.

---

# Secure Development Guidelines

Before opening a pull request:

- Run the application locally
- Check for console errors
- Validate ML predictions
- Ensure no sensitive data is exposed
- Verify responsive layouts
- Test edge-case inputs

---

# Hall of Fame

We appreciate contributors who responsibly disclose security issues.

| Contributor | Contribution |
|-------------|--------------|
| Your Name | Example security finding |

---

# Legal

By reporting vulnerabilities responsibly, you agree:

- Not to exploit vulnerabilities maliciously
- Not to access unauthorized data
- Not to disrupt service availability
- To follow ethical disclosure practices

Failure to follow responsible disclosure guidelines may result in disqualification from acknowledgements.

---

# Acknowledgements

Special thanks to:

- Open-source contributors
- GSSoC '26 community
- NSoC 2026 contributors
- Security researchers supporting ethical disclosure

---

<div align="center">

## CricScope

Machine Learning • IPL Analytics • Luxury UI

Built with ❤️ for open source.

</div>