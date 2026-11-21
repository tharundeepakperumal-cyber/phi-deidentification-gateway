
from src.deidentifier import deidentify
from src.rehydrator import rehydrate


clinical_note = """
Patient John Smith visited the hospital.

Contact: 9876543210
Email: john.smith@example.com
MRN: MRN123456

John Smith requires a follow-up appointment.
"""

# Step 1: De-identification
masked_note, mapping = deidentify(clinical_note)

print("\n========== ORIGINAL NOTE ==========")
print(clinical_note)

print("\n========== MASKED NOTE ==========")
print(masked_note)

print("\n========== PRIVATE MAPPING ==========")
print(mapping)

# Step 2: Simulate an LLM response
# This is a simulation, not an actual LLM call.
llm_response = """
Patient [PHONE_1] and [EMAIL_1] are
associated with record [MRN_1].
[PERSON_1] should attend a follow-up.
"""

print("\n========== SIMULATED LLM RESPONSE ==========")
print(llm_response)

# Step 3: Restore identifiers
final_response = rehydrate(llm_response, mapping)

print("\n========== REHYDRATED RESPONSE ==========")
print(final_response)
