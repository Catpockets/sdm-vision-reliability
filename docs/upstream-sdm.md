# Upstream SDM reference

The upstream `ReexpressAI/sdm_activations` repository is included as a pinned
Git submodule at `third_party/sdm_activations`.

The current pin is:

```text
06d295df31e0ab8ee13729244b9a7d4d5c4f91de
```

This repository contains SDM paper replication scripts and auxiliary data
processing code. The primary estimator implementation is maintained in the
[Reexpress MCP Server](https://github.com/ReexpressAI/reexpress_mcp_server).

## Checkout

Clone the project and initialize the pinned upstream checkout with:

```bash
git clone https://github.com/Catpockets/sdm-vision-reliability.git
cd sdm-vision-reliability
git submodule update --init --recursive
```

To inspect the exact upstream version currently used by this project:

```bash
git -C third_party/sdm_activations log -1 --oneline
```

Submodule updates should be deliberate: review the upstream release notes and
change the parent repository's submodule commit in a separate pull request.
