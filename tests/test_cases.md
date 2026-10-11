# AI Travel Policy Assistant — Test Cases

## 1. Policy Retrieval Tests

### TC01 — India Airport Policy
Question:
What is the airport reimbursement limit in India?

Expected:
- Tool: policy_search
- Answer mentions INR 2,000
- Source includes airport_policy.txt
- No hallucinated information

---

### TC02 — US Airport Policy
Question:
What is the airport reimbursement limit in the US?

Expected:
- Tool: policy_search
- Answer mentions USD 75
- Source includes travel_policy_us.txt or relevant policy source

---

### TC03 — Hotel Policy
Question:
What is the hotel reimbursement limit?

Expected:
- The assistant should not invent a hotel reimbursement amount.
- If the retrieved policies do not contain a hotel limit, clearly state that the available policy information does not specify one.

---

## 2. Follow-up / Memory Tests

### TC04 — Country Follow-up

First question:
What is the airport reimbursement limit in India?

Follow-up:
What about the US?

Expected:
- First answer: INR 2,000
- Second answer: USD 75
- The second question must be interpreted as:
  "What is the airport reimbursement limit in the US?"
- Tool: policy_search

---

### TC05 — Context Preservation

First question:
What is the airport reimbursement limit in India?

Follow-up:
Does this require approval?

Expected:
The assistant should understand that "this" refers to the airport reimbursement/travel limit discussed previously.

---

## 3. Employee Eligibility Tests

### TC06 — Eligible Employee

Question:
Is EMP001 eligible for travel?

Expected:
- Tool: check_employee_eligibility
- Employee: EMP001
- Eligible: True

---

### TC07 — Ineligible Employee

Question:
Is EMP004 eligible for travel?

Expected:
- Tool: check_employee_eligibility
- Employee: EMP004
- Eligible: False

---

### TC08 — Unknown Employee

Question:
Is EMP9999 eligible for travel?

Expected:
- Tool: check_employee_eligibility
- Graceful "employee not found" response
- No fabricated employee information

---

## 4. Trip Validation Tests

### TC09 — Trip Within Limit

Question:
Can EMP001 claim INR 1,500 for an airport trip?

Expected:
- Tool: validate_trip
- Result: Approved

---

### TC10 — Trip Above Limit

Question:
Can EMP001 claim INR 2,500 for an airport trip?

Expected:
- Tool: validate_trip
- Result: Needs Approval
- Excess amount: INR 500

---

### TC11 — Ineligible Employee Trip

Question:
Can EMP004 claim INR 1,500 for an airport trip?

Expected:
- Tool: validate_trip
- Result: Not Eligible

---

## 5. Tool Selection Tests

### TC12 — Required Information

Question:
What information is required to validate an employee trip?

Expected:
- Tool: policy_search
- Do NOT call validate_trip
- Do NOT invent employee ID, amount, or trip type

---

### TC13 — Missing Employee ID

Question:
Can I claim INR 2,500 for an airport trip?

Expected:
- Do not invent employee ID
- Ask for employee ID
- Do not execute validate_trip until employee ID is available

---

### TC14 — Missing Amount

Question:
Can EMP001 claim an airport trip?

Expected:
- Do not invent amount
- Ask for trip amount

---

## 6. Reimbursement Calculation

### TC15 — Amount Below Limit

Question:
Calculate reimbursement for INR 1,500 when the policy limit is INR 2,000.

Expected:
- Tool: calculate_reimbursement
- Reimbursable: INR 1,500
- Review amount: INR 0

---

### TC16 — Amount Above Limit

Question:
Calculate reimbursement for INR 2,500 when the policy limit is INR 2,000.

Expected:
- Tool: calculate_reimbursement
- Reimbursable: INR 2,000
- Review amount: INR 500

---

## 7. Complex Agent Tests

### TC17 — Employee + Policy + Validation

Question:
Can EMP001 claim INR 2,500 for an airport trip?

Expected:
The agent may combine:
1. Employee eligibility
2. Policy information
3. Trip validation

Expected final result:
Needs Approval

---

### TC18 — Late Night Trip

Question:
Can EMP001 claim INR 2,500 for an airport trip at 10:30 PM?

Expected:
- validate_trip
- Needs Approval
- late_night = True
- Correct policy-related explanation

---

## 8. Out-of-Scope Tests

### TC19 — Weather

Question:
What is the weather in Chennai?

Expected:
- Refuse because weather is outside the assistant's scope.
- Do not call policy tools unnecessarily.

---

### TC20 — General Knowledge

Question:
Who is the Prime Minister of India?

Expected:
- Politely state that the assistant only handles company travel and employee policy questions.

---

## 9. MCP Tests

### TC21 — MCP Policy Search

Expected:
search_company_policy is available and works.

### TC22 — MCP Employee Eligibility

Expected:
get_employee_eligibility is available and works.

### TC23 — MCP Trip Validation

Expected:
validate_employee_trip is available and works.

---

## 10. Flask API Tests

### TC24 — Health Check

GET /health

Expected:
HTTP 200

---

### TC25 — Ask Endpoint

POST /ask

Expected:
- HTTP 200
- answer returned
- sources returned
- tools_used returned

---

### TC26 — Clear Conversation

POST /clear

Expected:
- Conversation memory cleared successfully.