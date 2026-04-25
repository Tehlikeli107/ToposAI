# Security Policy

ToposAI is an experimental research framework. Please do not use it as a
security boundary, trading system, medical decision system, or autonomous
high-stakes decision system without independent review.

## Supported Versions

The repository is currently pre-release/alpha. Security fixes target the latest
`main` branch until tagged releases are established.

## Reporting a Vulnerability

Please report vulnerabilities through GitHub issues if the issue is not
sensitive. For sensitive reports, contact the maintainer privately before
publishing details.

Useful reports include:

- Affected commit or version.
- Steps to reproduce.
- Expected vs. actual behavior.
- Impact and suggested mitigation, if known.

## Scope

In scope:

- Code execution risks in package entry points.
- Unsafe dependency or packaging behavior.
- Bugs that could materially mislead users about benchmark or verification
  results.

Out of scope:

- Claims that experimental models perform poorly on real tasks.
- Results that differ across hardware unless there is a reproducibility bug.
- Vulnerabilities in third-party services used by optional demos.
