# Training workspace

V1 is inference-first. The preserved upstream HybridNets training scripts remain
at the repository root so the fork baseline and existing commands stay usable.
They retain their original dependency expectations and are not part of PR CI.

Future training modernization belongs here and must include:

- a pinned, independently installable training environment;
- dataset licensing and privacy documentation;
- reproducible train/validation splits and evaluation metrics;
- model-card and `models.lock.json` updates for released weights;
- no private dashcam footage or datasets committed to Git.
