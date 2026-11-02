# ADR 0001: Use a compact U-Net

**Status:** accepted

**Context:** There are only 118 labelled images, and the demo must run on a small CPU instance.

**Decision:** Use a four-level U-Net with 16 base channels (about 0.48M parameters), rather than a large pretrained encoder.

**Consequences:** It is cheap to serve and low risk for overfitting through capacity alone. A pretrained encoder might score higher but would need more memory and a larger image.
