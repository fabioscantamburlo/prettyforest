Here are the two use cases for beautiful-forest explained clearly, focusing on what they do and why they matter to an end-user.

Use Case 1: The Blueprint (Model Structure Only)
This use case focuses entirely on the architecture and logic of your trained model. It answers the question: “What rules did my AI model learn?”

What it shows: The "skeleton" or trunk of the tree. It maps out every decision split (e.g., Age ≤ 35), showing the exact pathways, features, and thresholds the model uses to make choices, down to the final leaf classifications or regression values.

The User Experience: The user gets a clean, interactive flow chart. If the tree is massive, they can click on parent nodes to collapse or expand deep branches, making a highly complex model suddenly look clean, organized, and digestible.

Best used for: * Auditing a model's logic to make sure it isn't picking up on biased or weird rules.

Sharing a high-level overview of a model's decision-making process with non-technical stakeholders or clients.

Use Case 2: The Flow (Model Structure + Real Data)
This use case takes the model's architecture from Use Case 1 and animates it with actual data. It answers the question: “How does my data actually travel through this model, and where does it end up?”

What it shows: The entire tree, but with your test or validation data flowing through it. Every node displays how many rows of data passed through it. More importantly, the final leaves are populated with visual distributions (like mini histograms or pie charts) showing the exact mix of data points that landed there.

The User Experience: Users can trace the exact path a specific group of samples took. They can visually see "heavy" paths where most of your data travels, versus "rare" paths. They can instantly see if a leaf node is "pure" (e.g., containing 100% fraud cases) or "muddy" (e.g., a 50/50 split of churn vs. no-churn, meaning the model struggled there).

Best used for:

Debugging and diagnosing model performance (finding exactly where the model is making mistakes).

Explaining specific predictions to stakeholders (e.g., "We rejected this loan application because the applicant fell into this specific high-risk leaf bucket, which historically has an 80% default rate").