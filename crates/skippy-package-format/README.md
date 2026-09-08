# skippy-package-format

Validated, versioned contracts for Skippy model packages.

This crate parses package manifests, derives their content identity, and resolves
stage admission data such as resident tensors and typed sidecar artifacts. It is
shared by packaging and runtime crates so both sides enforce the same package-v2
rules.
