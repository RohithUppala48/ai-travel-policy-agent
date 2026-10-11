# Prompt Engineering Experiments

## Objective

The objective is to design prompts that allow the Travel & Policy
Assistant to answer employee questions accurately while reducing
hallucination and providing policy source attribution.

---

## Experiment 1 — Basic Prompt

### Prompt Design

The basic prompt provides the policy context and user question.

### Problem

The model may provide an answer without clearly explaining its
reasoning or policy source.

### Observation

The response can be correct but may not consistently provide
structured policy information.

---

## Experiment 2 — P.T.C.F. Prompt

### Prompt Design

The prompt defines:

- Purpose
- Task
- Context
- Format
- Constraints

### Improvement

The model receives clearer instructions about its responsibility,
available context, expected response format, and restrictions.

### Expected Result

More consistent and policy-grounded responses.

---

## Experiment 3 — Role-Based Prompt

### Prompt Design

The model is assigned the role of a Corporate Travel Policy Assistant.

### Improvement

The role provides domain-specific context and establishes the expected
communication style.

### Expected Result

More business-friendly and relevant responses.

---

## Experiment 4 — Few-Shot Prompt

### Prompt Design

The prompt provides example questions and expected response styles
before presenting the actual user question.

### Improvement

The examples demonstrate the expected response behavior.

### Expected Result

More consistent answer formatting and tone.

---

## Experiment 5 — Structured Output

### Prompt Design

The response is divided into:

- Answer
- Policy Rule
- Source
- Confidence

### Improvement

The output becomes easier for the application to parse and display.

### Expected Result

Consistent responses that can later be integrated into the Flask
dashboard.

---

## Final Prompt

The final implementation will use a combination of:

- Role definition
- Clear task
- Retrieved context
- Explicit constraints
- Source attribution
- Structured response

This approach is intended to reduce hallucination and improve
consistency.