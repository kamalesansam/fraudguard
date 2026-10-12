# Security policy

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's
[private vulnerability reporting](https://github.com/kamalesansam/fraudguard/security/advisories/new)
for this repository. Expect an acknowledgement within 72 hours.

Useful details: affected file or endpoint, reproduction steps, and impact.

## Scope

In scope: the inference API, the model packaging and verification code, the CI
workflows, the container image and the Kubernetes manifests.

Out of scope: the synthetic training data (contains no real cardholder data)
and findings that require an already-compromised CI runner.

## Accepted risks

Known vulnerabilities in pipeline-only dependencies that have no fixed release
are listed with reasons and review dates in
[`security/pip-audit-ignore.txt`](security/pip-audit-ignore.txt). The serving
image is audited with no exceptions.
